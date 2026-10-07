"""SDK 의 EdgeRuntime 을 백그라운드 이벤트 루프에서 돌리고, 비디오 루프와 이어준다.

OpenCV 창(imshow/waitKey)은 메인 스레드를 원하고, EdgeRuntime 은 asyncio 루프를
원한다. 그래서 런타임은 전용 스레드의 루프에서 돌리고, 비디오 루프는 이 클래스를
통해서만 런타임과 이야기한다. 비디오 루프 쪽 호출은 어느 것도 블로킹하지 않는다.

런타임이 대신 해주는 일:

* Central 등록(register)과 주기적 하트비트 -> Edge Fleet 화면에 디바이스가 보인다
* 추론 결과를 로컬 SQLite 큐에 쌓고, 링크가 살아 있으면 업로드 (오프라인이면 누적)
* 명령 롱폴링: ping / restart / resync / set_policy / pull_model
* 로컬 API (기본 127.0.0.1:8600) -> ``geo-mlops-edge status`` 로 상태 조회
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Coroutine, Optional

from geo_mlops_sdk.contracts.inference import InferenceOutput, InferenceRecord, ModelRef
from geo_mlops_sdk.edge.daemon import STOP_GRACE_S, ApiStartupError, serve
from geo_mlops_sdk.edge.events import EdgeEvent
from geo_mlops_sdk.edge.runtime import EdgeRuntime
from geo_mlops_sdk.edge.settings import EdgeSettings
from geo_mlops_sdk.edge.sync import INFERENCE_KIND

logger = logging.getLogger(__name__)

#: HUD 용 상태 갱신 주기. ``runtime.status()`` 는 모델 캐시 디렉터리까지 훑으므로
#: 매 프레임 부를 값은 아니다.
STATUS_INTERVAL_S = 1.0


class EdgeAgent:
    def __init__(self, settings: EdgeSettings) -> None:
        self.settings = settings
        self.runtime: Optional[EdgeRuntime] = None
        #: Central 이 이 디바이스에서 활성화한 모델의 (가중치, 이름/버전).
        #: 비디오 루프가 꺼내서 검출 모델을 교체한다.
        self.models: queue.Queue[tuple[Path, ModelRef]] = queue.Queue()
        #: Fleet 에서 restart 명령이 오면 설정된다. 재시작은 supervisor 의 몫이다.
        self.restart_requested = threading.Event()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="edge-runtime", daemon=True
        )
        self._status: dict = {}
        self._status_future: Optional[Future] = None
        self._status_at = 0.0

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()
        if self.runtime is None or not self.runtime.running:
            raise RuntimeError("edge runtime failed to start; see the log above")

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:  # 루프가 이미 닫혔다
                pass
        self._thread.join(STOP_GRACE_S + 5.0)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:  # noqa: BLE001 - 비디오는 계속 재생, 원인은 로그로
            logger.exception("edge runtime stopped with an error")
        finally:
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self.runtime = EdgeRuntime(self.settings)
        self.runtime.events.add_listener(self._on_event)
        try:
            await self.runtime.start()
            self._ready.set()
            if self.settings.api.enabled:
                try:
                    await serve(self.runtime, self.settings, self._stop)
                except ApiStartupError as exc:
                    # 같은 PC 에 geo-mlops-edge 서비스가 이미 떠 있으면 포트가 겹친다.
                    # 로컬 API 만 포기하고 Central 연동은 그대로 둔다.
                    logger.warning("local API disabled: %s", exc)
            await self._stop.wait()
        finally:
            await asyncio.wait_for(self.runtime.stop(), STOP_GRACE_S)

    def _on_event(self, event: EdgeEvent) -> None:
        """런타임 이벤트 리스너. 런타임 루프 스레드에서 동기로 불린다."""
        if event.kind == "runtime" and event.name == "restart_requested":
            self.restart_requested.set()
        elif event.kind == "command":
            logger.info("command %s: %s", event.name, event.data)
        elif event.kind == "model" and event.name == "activated":
            self._offer_model(event.data["name"], event.data["version"])

    def _offer_model(self, name: str, version: str) -> None:
        assert self.runtime is not None
        model = self.runtime.models.get(name, version)
        if model is None or model.weights_path is None:
            return
        if model.framework != "yolo":
            logger.warning(
                "%s v%s is a %r model; this demo only plays YOLO models",
                name,
                version,
                model.framework,
            )
            return
        self.models.put((model.weights_path, ModelRef(name=name, version=version)))

    # --- called from the video loop ----------------------------------------

    def report(
        self,
        output: InferenceOutput,
        model: ModelRef,
        latency_ms: float,
        input_ref: str,
    ) -> None:
        """추론 결과 하나를 Central 행 큐에 넣는다. 기다리지 않는다."""
        record = InferenceRecord(
            id=str(uuid.uuid4()),
            ts=datetime.now(timezone.utc),
            model=model,
            input_ref=input_ref,
            output=output.model_dump(mode="json"),
            latency_ms=round(latency_ms, 3),
        )
        self._submit(self._enqueue(record))

    async def _enqueue(self, record: InferenceRecord) -> None:
        assert self.runtime is not None
        runtime = self.runtime
        # 큐 id 와 레코드 id 를 같게 둬야 재전송된 배치를 Central 이 중복으로 처리한다.
        await runtime.queue.enqueue_record(
            INFERENCE_KIND,
            record.model_dump(mode="json"),
            record_id=record.id,
            priority=record.priority,
        )
        await runtime.retention.enforce()
        await runtime.refresh_backlog()
        runtime.request_sync()

    def status(self) -> dict:
        """마지막으로 읽은 런타임 상태. 갱신은 백그라운드에서 하고 여기선 기다리지 않는다."""
        future = self._status_future
        if future is not None and future.done():
            if future.exception() is None:
                self._status = future.result()
            self._status_future = future = None

        now = time.monotonic()
        if future is None and now - self._status_at >= STATUS_INTERVAL_S:
            self._status_at = now
            self._status_future = self._submit(self._read_status())
        return self._status

    async def _read_status(self) -> dict:
        assert self.runtime is not None
        return self.runtime.status()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Optional[Future]:
        if self._loop is None or self._loop.is_closed():
            coro.close()
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:  # 종료 중
            coro.close()
            return None
        future.add_done_callback(_log_failure)
        return future


def _log_failure(future: Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.warning("edge runtime call failed: %s", future.exception())

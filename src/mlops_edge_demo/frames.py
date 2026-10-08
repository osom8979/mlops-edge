"""재생 중인 프레임을 Central 로 수집하는 collector. ``edge.yaml`` 에 ``type: frames`` 로 선언한다.

비디오 루프가 넘긴 프레임을 JPEG 으로 SDK 큐에 넣고, ``labels`` 가 켜져 있으면 현재
모델의 검출 결과를 같은 이름의 LabelMe JSON 으로 함께 넣는다. 오프라인 누적, 청크
업로드, 재전송은 SDK 업로더가 맡는다.

* ``dataset_id`` 를 주면 Central 은 파일을 그 데이터셋에 바로 넣고 검증한다. LabelMe
  JSON 은 같은 stem 의 이미지와 매칭되고, 2점 ``rectangle`` 은 YOLO 학습에서 박스가 된다.
* 비우면 Central 의 "수집 데이터" 에 따로 쌓인다. 그곳에서 데이터셋으로 옮기는 기능은
  Central 에 아직 없다.

반복 재생되는 파일은 매 바퀴 같은 프레임이 다시 오므로, 큐에 넣은 프레임 위치를
``{data_dir}/frames/{name}.seen`` 에 남겨 재시작 후에도 다시 올리지 않는다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from geo_mlops_sdk.contracts.inference import InferenceOutput
from geo_mlops_sdk.edge.collectors import CollectorBase, Sink

TYPE = "frames"


class FrameCollector(CollectorBase):
    type_name = TYPE

    def __init__(
        self,
        name: str,
        *,
        priority: int,
        ledger: Path,
        every_n_frames: int = 30,
        dataset_id: str = "",
        labels: bool = True,
        label_conf: float = 0.5,
        jpeg_quality: int = 90,
        kind: str = "frame",
    ) -> None:
        super().__init__(name, priority=priority)
        self.every_n_frames = max(1, every_n_frames)
        self.dataset_id = dataset_id
        self.labels = labels
        self.label_conf = label_conf
        self.jpeg_quality = jpeg_quality
        self.kind = kind
        self._ledger = ledger
        self._seen: set[str] = set()
        self._sink: Optional[Sink] = None

    @classmethod
    def from_options(
        cls, *, name: str, priority: int, options: dict, state_dir: Path
    ) -> "FrameCollector":
        """SDK collector 레지스트리용 팩토리. ``state_dir`` 는 등록할 때 묶어 넘긴다."""
        return cls(
            name,
            priority=priority,
            ledger=state_dir / f"{name}.seen",
            every_n_frames=int(options.get("every_n_frames", 30)),
            dataset_id=str(options.get("dataset_id") or ""),
            labels=bool(options.get("labels", True)),
            label_conf=float(options.get("label_conf", 0.5)),
            jpeg_quality=int(options.get("jpeg_quality", 90)),
            kind=str(options.get("kind") or "frame"),
        )

    async def start(self, sink: Sink) -> None:
        if self._ledger.exists():
            self._seen = set(self._ledger.read_text(encoding="utf-8").splitlines())
        self._sink = sink
        self.state = "running"

    async def stop(self) -> None:
        self._sink = None
        self.state = "stopped"

    # --- 비디오 루프 스레드에서 불린다: 판정과 인코딩만 하고 큐에는 넣지 않는다 ----------

    def wants(self, source: str, index: int, seekable: bool) -> bool:
        if self.state != "running" or index % self.every_n_frames:
            return False
        return not seekable or _key(source, index) not in self._seen

    def encode(
        self,
        source: str,
        index: int,
        seekable: bool,
        frame: np.ndarray,
        output: InferenceOutput,
    ) -> list[tuple[str, bytes]]:
        """올릴 파일들 ``[(filename, bytes), ...]``. 이미지가 먼저, 라벨이 뒤다."""
        # 파일은 프레임 위치로 이름을 지어 같은 프레임이 같은 이름을 갖게 하고,
        # 위치가 없는 스트림은 시각으로 짓는다.
        suffix = (
            f"{index:06d}"
            if seekable
            else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        )
        stem = f"{Path(source).stem}_{suffix}"
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise ValueError(f"cannot encode frame {index} of {source}")
        files = [(f"{stem}.jpg", jpeg.tobytes())]
        if self.labels:
            label = labelme(output, f"{stem}.jpg", self.label_conf)
            files.append((f"{stem}.json", label))
        return files

    # --- 런타임 루프에서 실행된다 ---------------------------------------------------

    async def collect(
        self, files: list[tuple[str, bytes]], *, key: Optional[str], meta: dict
    ) -> None:
        sink = self._sink
        if sink is None:
            return
        if self.dataset_id:
            # SDK 업로더가 meta 의 dataset_id 를 업로드 요청에 실어 보낸다.
            meta = {**meta, "dataset_id": self.dataset_id}
        try:
            for filename, data in files:
                await sink.blob(
                    self.kind,
                    data,
                    filename=filename,
                    priority=self.priority,
                    meta=meta,
                )
        except Exception as exc:
            self.note_error(str(exc))
            raise

        if key is not None:
            self._seen.add(key)
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self._ledger, "a", encoding="utf-8") as file:
                file.write(key + "\n")
        self.note_emit(datetime.now(timezone.utc))


def frame_key(source: str, index: int, seekable: bool) -> Optional[str]:
    """중복 판정 키. 스트림은 같은 프레임이 다시 오지 않으므로 키가 없다."""
    return _key(source, index) if seekable else None


def _key(source: str, index: int) -> str:
    return f"{Path(source).name}#{index}"


def labelme(output: InferenceOutput, image_name: str, min_conf: float) -> bytes:
    """검출 결과를 LabelMe JSON 으로 만든다.

    Central 의 데이터셋 검증과 YOLO 학습 런타임 모두 2점 ``rectangle`` 을 박스로 읽고,
    ``imageWidth``/``imageHeight`` 로 정규화한다. 사람이 검수할 사전 라벨이라
    신뢰도는 ``description`` 에 남긴다.
    """
    width, height = output.width, output.height
    shapes: list[dict[str, Any]] = [
        {
            "label": detection.name,
            "points": [
                [
                    round(detection.bbox[0] * width, 2),
                    round(detection.bbox[1] * height, 2),
                ],
                [
                    round(detection.bbox[2] * width, 2),
                    round(detection.bbox[3] * height, 2),
                ],
            ],
            "group_id": None,
            "description": f"conf={detection.conf}",
            "shape_type": "rectangle",
            "flags": {},
        }
        for detection in output.detections
        if detection.conf >= min_conf and len(detection.bbox) == 4
    ]
    document: dict[str, Any] = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    return json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")

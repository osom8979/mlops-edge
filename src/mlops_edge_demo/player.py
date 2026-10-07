"""비디오 루프: 프레임 읽기 -> 검출 -> 그리기 -> 표시, 그리고 주기적으로 Central 에 보고."""

from __future__ import annotations

import logging
import queue
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from geo_mlops_sdk.contracts.inference import ModelRef
from geo_mlops_sdk.edge.commands import RESTART_EXIT_CODE

from mlops_edge_demo.detector import Detector, Prediction
from mlops_edge_demo.edge import EdgeAgent

logger = logging.getLogger(__name__)

WINDOW = "MLOps Edge Demo"

#: 헤드리스 모드에서 상태를 로그로 남기는 주기.
LOG_INTERVAL_S = 5.0

#: 링크 상태별 HUD 색 (BGR).
LINK_COLORS = {
    "online": (80, 200, 80),
    "probing": (0, 200, 255),
    "offline": (0, 200, 255),
    "auth_failed": (60, 60, 255),
}


def run(
    source: str,
    detector: Detector,
    agent: EdgeAgent,
    *,
    report_interval: float,
    loop: bool,
    show: bool,
) -> int:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_period = 1.0 / fps
    name = Path(source).name
    logger.info("playing %s at %.1f fps", source, fps)
    if show:
        # Qt 툴바 없이, 비율을 유지한 채 창 크기를 바꿀 수 있게 연다.
        cv2.namedWindow(
            WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
        )
        # WINDOW_NORMAL 은 작은 기본 크기로 열리므로 영상 원본 크기로 맞춘다.
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width and height:
            cv2.resizeWindow(WINDOW, width, height)

    fps_meter = 0.0
    last_report = last_log = 0.0
    rewound = False
    try:
        while not agent.restart_requested.is_set():
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                # 스트림은 되감기가 안 되므로, 되감은 직후에도 못 읽으면 끝낸다.
                if not loop or rewound:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                rewound = True
                continue
            rewound = False

            detector = _swap_model(agent, detector)
            prediction = detector.predict(frame)

            now = time.monotonic()
            if now - last_report >= report_interval:
                index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                agent.report(
                    prediction.output,
                    detector.ref,
                    prediction.latency_ms,
                    input_ref=f"{name}#frame={index}",
                )
                last_report = now

            status = agent.status()
            if now - last_log >= LOG_INTERVAL_S:
                logger.info(
                    " | ".join(_summary(detector.ref, prediction, fps_meter, status))
                )
                last_log = now

            if show:
                image = prediction.annotated
                _draw_hud(image, detector.ref, prediction, fps_meter, status)
                cv2.imshow(WINDOW, image)

            # 원본 FPS 에 맞춰 재생한다. 추론이 더 느리면 기다리지 않는다.
            remaining = frame_period - (time.monotonic() - started)
            if show:
                key = cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif remaining > 0:
                time.sleep(remaining)

            elapsed = time.monotonic() - started
            fps_meter = 0.9 * fps_meter + 0.1 / elapsed if fps_meter else 1.0 / elapsed
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if agent.restart_requested.is_set():
        logger.warning(
            "restart requested by Central; exiting with %d", RESTART_EXIT_CODE
        )
        return RESTART_EXIT_CODE
    return 0


def _swap_model(agent: EdgeAgent, detector: Detector) -> Detector:
    """Central 이 새 모델을 활성화했으면 그 모델로 갈아탄다."""
    try:
        weights, ref = agent.models.get_nowait()
    except queue.Empty:
        return detector

    try:
        swapped = Detector(weights, ref, conf=detector.conf)
    except Exception:  # noqa: BLE001 - 깨진 모델 때문에 데모를 멈추지 않는다
        logger.exception(
            "cannot load %s:%s; keeping %s", ref.name, ref.version, detector.ref.name
        )
        return detector
    logger.info(
        "switched model: %s:%s -> %s:%s",
        detector.ref.name,
        detector.ref.version,
        ref.name,
        ref.version,
    )
    return swapped


def _summary(
    ref: ModelRef, prediction: Prediction, fps: float, status: dict
) -> list[str]:
    detections = prediction.output.detections
    counts = Counter(d.name for d in detections)
    objects = ", ".join(f"{label} {n}" for label, n in counts.most_common(4))
    link = status.get("link", {}).get("state", "starting")
    backlog = (status.get("backlog") or {}).get("count", 0)
    sync = status.get("sync", {}).get("state", "-")
    registered = (
        "registered" if status.get("device", {}).get("registered") else "unregistered"
    )
    return [
        f"model {ref.name}:{ref.version} {prediction.latency_ms:.1f} ms",
        f"{fps:.1f} fps, {len(detections)} objects ({objects or '-'})",
        f"central {link} / {registered} / backlog {backlog} / sync {sync}",
    ]


def _draw_hud(
    image: np.ndarray, ref: ModelRef, prediction: Prediction, fps: float, status: dict
) -> None:
    device = status.get("device", {}).get("id", "")
    lines = [f"MLOps Edge Demo  {device}", *_summary(ref, prediction, fps, status)]
    link = status.get("link", {}).get("state", "")

    font, scale, thickness, line_height = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 20
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    height = line_height * len(lines) + 8

    # 반투명 배경 위에 글자를 올린다.
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width + 16, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)
    for row, line in enumerate(lines):
        color = (
            LINK_COLORS.get(link, (255, 255, 255))
            if row == len(lines) - 1
            else (255, 255, 255)
        )
        cv2.putText(
            image,
            line,
            (8, 18 + row * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

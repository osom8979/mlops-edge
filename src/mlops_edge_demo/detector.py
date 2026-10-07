"""OpenCV 프레임에 YOLO 를 돌리고, 결과를 SDK 의 추론 계약(InferenceOutput)으로 돌려준다.

SDK 의 ``YoloRunner`` 는 인코딩된 이미지 바이트를 받는다. 비디오 루프에서 매 프레임을
JPEG 으로 인코딩했다가 다시 디코딩할 이유는 없으므로 프레임(ndarray)을 직접 넣되,
출력 형태(정규화된 ``[x1, y1, x2, y2]`` 등)는 SDK 와 같게 맞춰 Central 쪽에서
엣지 추론과 서버 추론을 구분할 필요가 없도록 한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from geo_mlops_sdk.contracts.inference import Detection, InferenceOutput, ModelRef
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class Prediction:
    output: InferenceOutput  # Central 로 보낼 결과
    annotated: np.ndarray  # 화면에 그릴 프레임
    latency_ms: float


class Detector:
    def __init__(self, weights: str | Path, ref: ModelRef, *, conf: float) -> None:
        self.model = YOLO(str(weights))
        self.ref = ref
        self.conf = conf

    def predict(self, frame: np.ndarray) -> Prediction:
        started = time.perf_counter()
        # stream=False 이면 항상 list[Results] 다 (ultralytics 의 반환 타입은 union).
        results = cast(
            list[Results], self.model.predict(frame, conf=self.conf, verbose=False)
        )
        result = results[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Central 이 분류 모델을 배포한 경우처럼 박스가 없으면 검출 0건으로 본다.
        boxes = result.boxes
        rows = (
            zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
            if boxes is not None
            else []
        )
        detections = [
            Detection(
                cls=int(cls),
                name=str(result.names[int(cls)]),
                conf=round(conf, 4),
                bbox=[round(value, 6) for value in xyxyn],
            )
            for xyxyn, conf, cls in rows
        ]
        height, width = frame.shape[:2]
        output = InferenceOutput(
            task="object_detection", width=width, height=height, detections=detections
        )
        return Prediction(output=output, annotated=result.plot(), latency_ms=latency_ms)

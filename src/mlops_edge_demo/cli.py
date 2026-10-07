"""명령행 진입점: ``mlops-edge-demo [SOURCE] [options]``."""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path
from typing import Optional, Sequence

import httpx
from geo_mlops_sdk.contracts.inference import ModelRef
from geo_mlops_sdk.edge.daemon import configure_logging
from geo_mlops_sdk.edge.settings import EdgeSettings

from mlops_edge_demo import player, video
from mlops_edge_demo.detector import Detector
from mlops_edge_demo.edge import EdgeAgent

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mlops-edge-demo",
        description=(
            "데모 비디오에 YOLO 검출을 돌려 화면에 보여주고, "
            "결과를 MLOps Central Platform 의 Edge Fleet 으로 보고한다."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=video.DEFAULT_VIDEO_URL,
        help="비디오 URL(최초 1회 다운로드 후 캐시), 로컬 파일 또는 스트림 주소 (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        default="yolo26n.pt",
        help="YOLO 가중치. 경로 없이 이름만 주면 COCO 사전학습 모델을 캐시에 받는다. "
        "Central 이 이 디바이스에 모델을 배포하면 그 모델로 교체된다 (default: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="검출 신뢰도 임계값 (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("edge.yaml"),
        help="Edge 런타임 설정 파일 (default: %(default)s)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="GEO_EDGE_CENTRAL__TOKEN 등 비밀값을 담은 파일. 없으면 무시 (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="비디오/가중치 다운로드 캐시 (default: %(default)s)",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=1.0,
        help="Central 로 보낼 추론 결과 간격(초). 0 이면 매 프레임 (default: %(default)s)",
    )
    parser.add_argument(
        "--once", action="store_true", help="반복 재생하지 않고 영상 끝에서 종료"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="창 없이 실행 (헤드리스 Edge PC)"
    )
    return parser.parse_args(argv)


def load_settings(config: Path, env_file: Path) -> EdgeSettings:
    """우선순위: 환경변수 > env 파일 > 설정 파일 > 기본값 (SDK 규칙과 같다)."""
    if not config.is_file():
        raise SystemExit(f"config file not found: {config}")
    data = EdgeSettings.read_config_file(config)
    return EdgeSettings(_env_file=env_file, **data)  # type: ignore[call-arg]


def resolve_weights(weights: str, cache_dir: Path) -> Path:
    """``yolo26n.pt`` 처럼 이름만 주면 작업 디렉터리 대신 캐시에 받도록 경로를 바꾼다."""
    path = Path(weights)
    if path.exists() or path.parent != Path("."):
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / path.name


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config, args.env_file)
    configure_logging(settings.log_level)

    if not settings.central.configured:
        logger.warning(
            "central.base_url or GEO_EDGE_CENTRAL__TOKEN is not set: "
            "results are queued locally and nothing reaches the Edge Fleet"
        )

    try:
        source = video.resolve(args.source, args.cache_dir / "videos")
    except httpx.HTTPError as exc:
        raise SystemExit(f"cannot download {args.source}: {exc}")
    weights = resolve_weights(args.weights, args.cache_dir / "weights")
    detector = Detector(
        weights, ModelRef(name=weights.stem, version="pretrained"), conf=args.conf
    )

    agent = EdgeAgent(settings)
    agent.start()
    # systemd 의 SIGTERM 도 Ctrl-C 처럼 처리해 런타임을 정상 종료시킨다.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        return player.run(
            source,
            detector,
            agent,
            report_interval=args.report_interval,
            loop=not args.once,
            show=not args.no_show,
        )
    except KeyboardInterrupt:
        return 0
    finally:
        # 정리 도중의 두 번째 Ctrl-C 가 런타임 종료(큐 정리, DB 닫기)를 끊지 않게 한다.
        # agent.stop() 은 제한 시간 안에 반드시 돌아온다.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        agent.stop()

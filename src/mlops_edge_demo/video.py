"""데모 비디오 소스: URL 은 한 번만 내려받고 이후엔 캐시를 재사용한다."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

logger = logging.getLogger(__name__)

#: KITTI 주행 영상 (ultralytics/assets 배포본, 17초, 6 MB). 후보 영상들을 yolo26n 으로
#: 비교했을 때 빈 프레임 없이 프레임당 약 6개, COCO 8개 클래스(car, traffic light, bus,
#: person, truck, bicycle, motorcycle, stop sign)가 잡혀 COCO 검출을 보여주기에 가장 좋았다.
DEFAULT_VIDEO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "kitti-inference-vid.mp4"
)


def is_url(source: str) -> bool:
    return urlsplit(source).scheme in ("http", "https")


def cached_path(url: str, cache_dir: Path) -> Path:
    """``url`` 의 캐시 위치. 이름이 같은 다른 URL 과 섞이지 않도록 해시를 붙인다."""
    name = Path(unquote(urlsplit(url).path)).name or "video.mp4"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}-{name}"


def fetch(url: str, cache_dir: Path) -> Path:
    """``url`` 을 ``cache_dir`` 에 내려받는다. 이미 있으면 그대로 쓴다.

    ``.part`` 에 받은 뒤 끝까지 받았을 때만 rename 하므로, 중간에 끊긴 다운로드가
    완성된 캐시처럼 남지 않는다.
    """
    target = cached_path(url, cache_dir)
    if target.exists():
        logger.info("using cached video %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    logger.info("downloading %s", url)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            written, logged_at = 0, time.monotonic()
            with open(partial, "wb") as file:
                for chunk in response.iter_bytes(64 * 1024):
                    file.write(chunk)
                    written += len(chunk)
                    # 느린 회선에서 멈춘 것처럼 보이지 않도록 5초마다 진행률을 남긴다.
                    if time.monotonic() - logged_at >= 5.0:
                        logged_at = time.monotonic()
                        percent = f" ({written * 100 // total}%)" if total else ""
                        logger.info("  %.1f MB%s", written / 1e6, percent)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

    logger.info("cached %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def resolve(source: str, cache_dir: Path) -> str:
    """HTTP(S) URL 은 캐시 파일 경로로 바꾸고, 그 외(로컬 파일, RTSP 등)는 그대로 OpenCV 에 넘긴다."""
    return str(fetch(source, cache_dir)) if is_url(source) else source

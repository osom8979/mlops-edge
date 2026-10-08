# MLOps Edge Demo

`uv` + [`geo-mlops-sdk`](https://pypi.org/project/geo-mlops-sdk/) + YOLO + OpenCV 비디오 플레이어로 구성한
**Edge PC 데모**입니다. 데모 비디오에서 COCO 객체를 검출해 화면에 보여주고, 그 PC 를
[MLOps Central Platform](https://mlops.unwiki.net) 의 **Edge Fleet** 디바이스로 연결해
상태와 추론 결과를 보고하고, 재생 중인 프레임을 학습용 데이터로 수집합니다.

```
┌──────────────────────────── Edge PC: mlops-edge-demo ────────────────────────────┐
│                                                                                  │
│  video URL ─► cache/videos/ ─► cv2.VideoCapture ─► YOLO ─► cv2.imshow (+HUD)     │
│                                                     │                            │
│                  1초마다 InferenceRecord · 30프레임마다 JPEG + LabelMe 사전 라벨  │
│                                                     ▼                            │
│  EdgeRuntime (geo-mlops-sdk, 백그라운드 asyncio 스레드)                            │
│    로컬 큐(SQLite) · 업로더 · 하트비트 · 명령 롱폴링 · frames collector            │
│    로컬 API 127.0.0.1:8600                                                       │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │ HTTPS, X-Edge-Token
                                       ▼
                 Central API  https://mlops-api-dev.unwiki.net
                 Web UI       https://mlops.unwiki.net  →  Edge Fleet
```

## Central 과 주고받는 것

| 방향 | 내용 |
|---|---|
| Edge → Central | **등록**: 시작할 때와 링크가 돌아올 때마다 (멱등) |
| | **하트비트** (10초): CPU/GPU/MEM/DISK, 백로그, 동기화 상태, 보유 모델 |
| | **추론 결과**: 1초에 1건. 오프라인이면 `data/edge` 큐에 쌓였다가 재연결 시 업로드 |
| | **프레임 수집**: `frames` collector 가 30프레임마다 JPEG 과 LabelMe 사전 라벨을 청크 업로드. `dataset_id` 를 주면 그 데이터셋으로 바로 들어감 |
| Central → Edge | **명령**: `ping`, `restart`(종료 코드 3), `resync`, `set_policy`, `pull_model` |
| | `pull_model` 로 YOLO 모델을 보내면 내려받아 활성화하고, **재생 중인 화면의 모델도 즉시 교체**됩니다 |

## 준비

- [uv](https://docs.astral.sh/uv/) (Python 3.12 는 `.python-version` 에 따라 uv 가 맞춥니다)
- (선택) NVIDIA GPU — 없으면 CPU 로 동작합니다

```bash
uv sync
```

Linux 에서는 CUDA 빌드 torch 와 `nvidia-*` 휠(합계 약 3 GB)이 함께 설치됩니다. 회선이 느려
`Failed to download distribution due to network timeout` 이 나면 타임아웃을 늘려 다시 실행하세요.

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

### Edge Fleet 에 디바이스 등록

1. <https://mlops.unwiki.net> → **Edge Fleet** → **디바이스 등록**
2. 디바이스 ID 에 `edge-demo-01` 입력 (`edge.yaml` 의 `device.id` 와 같게), 스코프 선택
3. 발급된 토큰을 복사 (창을 닫으면 다시 볼 수 없습니다)
4. `.env` 에 저장

```bash
cp .env.example .env
# GEO_EDGE_CENTRAL__TOKEN=<발급받은 토큰>
```

토큰이 없어도 데모는 돌아갑니다. 결과는 로컬 큐에 쌓이기만 하고, 토큰을 넣고 다시 실행하면 그때 올라갑니다.

## 실행

데모 비디오 URL 을 인자로 줍니다. 처음 한 번만 `cache/videos/` 에 내려받고 이후에는 캐시를 씁니다.

```bash
uv run mlops-edge-demo https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti-inference-vid.mp4
```

인자를 생략하면 위 URL 이 기본값입니다. 도심 주행 영상(17초, 반복 재생)으로, 후보 영상들을
`yolo26n` 으로 비교했을 때 빈 프레임 없이 프레임당 약 6개, COCO 8개 클래스(car, traffic light,
bus, person, truck, bicycle, motorcycle, stop sign)가 검출되어 기본값으로 골랐습니다. 그 밖의 예:

```bash
# 도로 항공 영상: 프레임당 차량 20여 대
uv run mlops-edge-demo https://github.com/ultralytics/assets/releases/download/v0.0.0/dashboard_sample.mp4

# 실내 영상: chair, potted plant, laptop, dining table, bottle, cup 등 7개 클래스 (5초)
uv run mlops-edge-demo https://github.com/ultralytics/assets/releases/download/v0.0.0/home-objects-compressed.mp4

# 로컬 파일 / RTSP 스트림은 다운로드 없이 그대로 OpenCV 로 엽니다
uv run mlops-edge-demo ./my-video.mp4

# 헤드리스 Edge PC: 창 없이, 5초마다 상태를 로그로
uv run mlops-edge-demo --no-show
```

창에서 `q` 또는 `ESC` 로 종료합니다. 화면 왼쪽 위 HUD 에 모델·추론 시간·FPS·검출 수와
Central 링크 상태(초록 `online` / 노랑 `offline`·`probing` / 빨강 `auth_failed`), 백로그가 표시됩니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `SOURCE` | KITTI 주행 영상 URL | HTTP(S) URL 은 다운로드 후 캐시, 그 외는 OpenCV 에 그대로 전달 |
| `--weights` | `yolo26n.pt` | COCO 사전학습 YOLO. 이름만 주면 `cache/weights/` 에 받습니다 |
| `--conf` | `0.25` | 검출 신뢰도 임계값 |
| `--config` | `edge.yaml` | Edge 런타임 설정 |
| `--env-file` | `.env` | 토큰 등 비밀값 |
| `--cache-dir` | `cache` | 비디오·가중치 캐시 |
| `--report-interval` | `1.0` | Central 로 보낼 추론 결과 간격(초), `0` 이면 매 프레임 |
| `--once` | | 반복 재생하지 않고 영상 끝에서 종료 |
| `--no-show` | | 창 없이 실행 |

## 상태 확인

데모가 띄운 임베드 런타임은 SDK 데몬과 같은 로컬 API 를 열기 때문에, SDK CLI 로 그대로 조회할 수 있습니다.

```bash
uv run geo-mlops-edge status --config edge.yaml        # 링크, 백로그, 모델, 조치 필요 사항
uv run geo-mlops-edge queue --config edge.yaml --limit 5
```

## 수집 → 학습 → Edge 배포

재생 중인 프레임으로 Central 에서 데이터셋을 만들고, 거기서 학습한 YOLO 를 이 PC 로 다시 보내는 순서입니다.

### 1. 프레임 수집

`edge.yaml` 의 `collectors:` 에 선언된 `frames` collector 가 `every_n_frames` 마다 프레임을 JPEG 으로,
그 프레임의 검출 결과를 **같은 이름의 LabelMe JSON**(2점 `rectangle`)으로 SDK 큐에 넣습니다.
업로드는 SDK 가 청크 단위로 하며, 오프라인이면 쌓아 두었다가 재연결 시 이어서 보냅니다.

1. Central 웹 UI → **데이터셋**에서 수집용 데이터셋을 만들고 ID 를 복사합니다.
   수집하는 동안에는 freeze 하지 마세요. freeze 된 데이터셋은 파일이 들어올 때마다 매니페스트 전체를 다시 계산합니다.
2. `edge.yaml` → `collectors` → `options.dataset_id` 에 넣고 데모를 실행합니다.
   HUD 마지막 줄과 5초 로그의 `frames N` 이 이번 실행에서 큐에 넣은 프레임 수입니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `every_n_frames` | `30` | 30fps 영상이면 초당 1장. KITTI 영상(17초)은 한 바퀴에 17장 |
| `dataset_id` | `""` | 비우면 Central 의 "수집 데이터" 에 따로 쌓입니다. **거기서 데이터셋으로 옮기는 기능은 Central 에 아직 없습니다** |
| `labels` | `true` | LabelMe 사전 라벨을 함께 올릴지 |
| `label_conf` | `0.5` | 이 신뢰도 이상인 검출만 사전 라벨로. 신뢰도는 각 shape 의 `description` 에 남깁니다 |
| `jpeg_quality` | `90` | |

- 파일 이름은 `<영상 이름>_<프레임 번호 6자리>.jpg` / `.json` 입니다. Central 은 같은 stem 으로 이미지와 라벨을 짝짓고,
  YOLO 학습 런타임은 사각형을 `cls cx cy w h` 라벨로 바꿉니다.
- 반복 재생해도, 다시 실행해도 이미 큐에 넣은 프레임은 다시 올리지 않습니다. 기록은 `data/edge/frames/<collector 이름>.seen` 에 있고,
  이 파일을 지우면 처음부터 다시 수집합니다.
- 사전 라벨은 지금 모델의 예측입니다. 그대로 학습하면 모델이 자기 오류를 다시 배우므로 **학습 전에 사람이 검수**해야 합니다.
  Central 에는 아직 라벨링 도구가 없습니다.
- 디바이스 토큰에 `data:write` 스코프가 필요합니다 (디바이스 등록 시 기본 포함).

### 2. 학습

데이터셋을 freeze 한 뒤 **학습** 위저드에서 그 데이터셋과 클래스를 고르고, 마지막 단계에서
**모델 레지스트리 등록**을 켜고 **등록 모델 이름**을 고정합니다(예: `edge-demo-yolo`).
등록을 켜지 않으면 Edge 가 내려받을 수 있는 모델 목록에 나타나지 않습니다.
이름에는 `/` 나 `'` 를 넣지 마세요. Edge 다운로드 경로와 레지스트리 조회가 깨집니다.

### 3. Edge 로 보내기

Edge Fleet → `edge-demo-01` → 명령에서 `pull_model` 에 등록 모델 이름/버전을 넣어 보내면,
디바이스가 다음 폴링 때 받아서 `data/edge/models/` 에 캐시하고 활성화합니다. 데모는 그 이벤트를
받아 재생 중인 검출 모델을 교체하고, 이후 추론 결과·사전 라벨은 새 모델 이름/버전으로 보고됩니다.
캐시된 모델은 다음 실행 때도 자동으로 다시 활성화됩니다. (YOLO 프레임워크 모델만 화면에 반영합니다.)

지금 Central 은 **승인(Production 승격)이나 배포 화면에서 Edge 로 모델을 보내지 않습니다.** Edge 로 보내는 방법은 위의 수동 `pull_model` 명령뿐입니다.

`restart` 명령을 받으면 종료 코드 3 으로 끝나므로, systemd 같은 감시자 아래에서 돌리면 새로 뜹니다.

## 파일 구성

```
edge.yaml                      Edge 런타임 설정 (Central 주소, device id, 주기, 보존 정책)
.env.example                   토큰 템플릿
src/mlops_edge_demo/
  cli.py                       명령행 인자, 설정 로드, 조립
  video.py                     URL 다운로드 + 캐시 (.part → rename)
  detector.py                  YOLO 추론 → SDK InferenceOutput
  edge.py                      EdgeRuntime 을 백그라운드 루프에서 실행, 보고·모델 교체 브리지
  frames.py                    frames collector: 프레임 JPEG + LabelMe 사전 라벨 수집
  player.py                    OpenCV 재생 루프와 HUD
cache/                         다운로드 캐시 (git 제외)
data/edge/                     로컬 큐 DB, 스풀, Central 모델 캐시, 수집 기록 (git 제외)
```

## 데모 비디오 출처

모두 [ultralytics/assets](https://github.com/ultralytics/assets/releases/tag/v0.0.0) 릴리스에서 받습니다.

- `kitti-inference-vid.mp4` — [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/) 영상
  (CC BY-NC-SA 3.0, 비상업적 용도). 상업 시연에는 다른 영상을 쓰세요.
- `dashboard_sample.mp4`, `home-objects-compressed.mp4` — Ultralytics 문서용 샘플

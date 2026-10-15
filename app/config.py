import os
import socket
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
DB_PATH = DATA_DIR / "records.db"
MODEL_PATH = BASE_DIR / "yolo26n.onnx"

CAMERA_INDEX = "auto"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
BURST_COUNT = 3
BURST_INTERVAL = 0.25

EDGE_MARGIN_RATIO = 0.02
MIN_PERSON_CONF = 0.5
MIN_PERSON_AREA_RATIO = 0.03
DETECT_IMGSZ = 320
POSE_ONLY_ON_EDGE = True
IGNORE_BOTTOM_EDGE = True
REQUIRE_FULL_BODY = False

POSE_MODEL_PATH = BASE_DIR / "yolo26n-pose.onnx"
ENABLE_POSE = True
KEYPOINT_CONF = 0.5

ENABLE_EYE_CHECK = True
EAR_THRESHOLD = 0.20
MAX_FACES = 10
FRONTAL_YAW_DEG = 30.0

AUTO_RETRY = True
MAX_ATTEMPTS = 3

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

ENABLE_VOICE = True
ENABLE_DETECTOR = True
MOCK_CAMERA = False

COS_REGION = ""
COS_BUCKET = ""
COS_PREFIX = "mubear"
PRESIGN_EXPIRE_SECONDS = 4 * 3600
CLEANUP_AGE_SECONDS = 4 * 3600
CLEANUP_INTERVAL_SECONDS = 600

_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

COS_SECRET_ID = os.environ.get("COS_SECRET_ID", "")
COS_SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
COS_REGION = os.environ.get("COS_REGION", "") or COS_REGION
COS_BUCKET = os.environ.get("COS_BUCKET", "") or COS_BUCKET


def local_ip() -> str:
    candidates = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in candidates:
                candidates.append(ip)
    except OSError:
        pass
    for prefix in ("192.168.", "10."):
        for ip in candidates:
            if ip.startswith(prefix):
                return ip
    for ip in candidates:
        if ip.startswith("172."):
            second = int(ip.split(".")[1])
            if 16 <= second <= 31:
                return ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def base_url() -> str:
    return f"http://{local_ip()}:{SERVER_PORT}"

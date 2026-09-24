import os

# ============================================================
#  Local JSON Watch Folder
# ============================================================
# aws_watcher.py watches this folder for JSON files written by the mmWave
# Visualizer's "Stream to isp3" feature (see MMWAVE visualizer/stream_writer.py).
# Default assumes the two projects sit side by side, e.g.:
#   Activity/
#     MMWAVE visualizer/radar_stream/   <- visualizer writes here every ~1s
#     isp3/rpi_pipeline/config.py        <- this file
# Override with the WATCH_DIR env var, or pass a path directly to
# aws_watcher.py, if a different isp3 checkout lives somewhere else — that is
# the one thing to repoint when swapping in another isp3 copy.
_DEFAULT_WATCH_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "MMWAVE visualizer", "radar_stream",
))
WATCH_DIR = os.getenv("WATCH_DIR", _DEFAULT_WATCH_DIR)

# ============================================================
#  Radar / Serial Settings
# ============================================================
# Common values on Raspberry Pi:
#   /dev/ttyACM0   (data port, IWR6843)
#   /dev/ttyACM1   (config port, IWR6843)
SERIAL_PORT_DATA = os.getenv("SERIAL_PORT_DATA", "COM8")
SERIAL_PORT_CFG  = os.getenv("SERIAL_PORT_CFG",  "COM7")
SERIAL_BAUD      = 921600

# ============================================================
#  Pipeline Settings
# ============================================================
WINDOW_SIZE   = 40    # frames per inference window (matches training)
STRIDE        = 5     # frames to slide window by
SNR_THRESHOLD = 10.0  # minimum SNR to keep a point
FRAME_DT      = 0.055 # seconds between frames (~18 fps)
NUM_FEATURES  = 20    # feature vector size (matches multi_class_model_best.pth)

# ============================================================
#  AWS EC2 API / Local Backend API
# ============================================================
# Endpoint for per-frame feature streaming.
# Example: http://<ec2-public-ip>/frame or http://127.0.0.1:8000/frame
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://127.0.0.1:8000/frame")
DEVICE_ID     = os.getenv("DEVICE_ID",     "rpi-1")
CLOUD_TIMEOUT = float(os.getenv("CLOUD_TIMEOUT", "1.5"))

# ============================================================
#  Class Labels (must match training order)
# ============================================================
CLASSES        = ["NO-FALL", "FALL"]
FALL_CLASS_IDS = {1}

#!/usr/bin/env bash
# One-time setup: virtual environment + Python packages for the edge.
# System packages (PyQt5, numpy, pyserial, PyOpenGL) are reused from the Pi's
# own Python via --system-site-packages, because PyQt5 builds very slowly
# from source on a Pi.
set -e
cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pyqt5 python3-pyqt5.qtopengl \
                        python3-numpy python3-serial python3-opengl

python3 -m venv --system-site-packages venv
./venv/bin/pip install --upgrade pip
# Versions matter here:
#   xgboost 3.x pulls a 305 MB NVIDIA CUDA library that a Pi can never use.
#   scikit-learn 1.7+ breaks xgboost 2.x model loading ("_estimator_type undefined").
./venv/bin/pip install --no-cache-dir "xgboost==2.1.4" "scikit-learn==1.6.1" joblib watchdog requests pyqtgraph json-fix

echo
echo "Checking imports..."
./venv/bin/python - <<'PY'
import PyQt5, pyqtgraph, serial, numpy, OpenGL, xgboost, sklearn, joblib, watchdog, requests
print("all edge dependencies import OK")
PY
echo "Setup done. Next: ./run_visualizer.sh  (and ./run_watcher.sh in a second terminal)"

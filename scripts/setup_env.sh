#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
TORCH_FLAVOR="${1:-auto}" # auto | cpu | cu124 | cu126

echo "[carphotoqa] root: ${ROOT_DIR}"
echo "[carphotoqa] venv: ${VENV_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "${TORCH_FLAVOR}" in
  cpu)
    echo "[carphotoqa] Installing CPU torch..."
    pip install "torch>=2.4,<3.0" "torchvision>=0.19,<1.0" \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu124)
    echo "[carphotoqa] Installing CUDA 12.4 torch..."
    pip install "torch>=2.4,<3.0" "torchvision>=0.19,<1.0" \
      --index-url https://download.pytorch.org/whl/cu124
    ;;
  cu126)
    echo "[carphotoqa] Installing CUDA 12.6 torch..."
    pip install "torch>=2.6,<3.0" "torchvision>=0.21,<1.0" \
      --index-url https://download.pytorch.org/whl/cu126
    ;;
  auto)
    echo "[carphotoqa] Installing torch from PyPI (auto CUDA-capable wheels when available)..."
    pip install "torch>=2.4,<3.0" "torchvision>=0.19,<1.0"
    ;;
  *)
    echo "Unknown torch flavor: ${TORCH_FLAVOR}"
    echo "Usage: scripts/setup_env.sh [auto|cpu|cu124|cu126]"
    exit 2
    ;;
esac

echo "[carphotoqa] Installing project dependencies..."
pip install -r "${ROOT_DIR}/requirements-dev.txt"

echo "[carphotoqa] Python: $(python --version)"
echo "[carphotoqa] Pip: $(pip --version)"

python - <<'PY'
import json
import sys

import torch
import ultralytics

report = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "ultralytics": ultralytics.__version__,
}

if report["torch_cuda_available"]:
    report["cuda_device_name"] = torch.cuda.get_device_name(0)

print("[carphotoqa] Sanity report:")
print(json.dumps(report, indent=2))
PY

echo
echo "[carphotoqa] Environment ready."
echo "Activate with:"
echo "  source .venv/bin/activate"

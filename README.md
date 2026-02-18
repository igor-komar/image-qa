# CarPhotoQA Environment Bootstrap

This repository is now prepared with a Python environment baseline for building the CompCars detector/segmenter pipeline.

## 1) Create and populate virtualenv

### Recommended (auto torch wheel selection)
```bash
./scripts/setup_env.sh auto
```

### NVIDIA CUDA 12.4 wheels (common stable option)
```bash
./scripts/setup_env.sh cu124
```

### CPU-only fallback
```bash
./scripts/setup_env.sh cpu
```

## 2) Activate environment
```bash
source .venv/bin/activate
```

## 3) Quick sanity checks
```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
python -c "import pycocotools, cv2, scipy, typer, rich; print('ok')"
```

## 4) (Optional) enable git hooks
```bash
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

## Installed dependency sets

- Runtime: `requirements.txt`
- Dev/test/lint: `requirements-dev.txt`
- Exact snapshot from this machine: `requirements-lock.txt`
- Project metadata and ruff/pytest config: `pyproject.toml`

## Notes for upcoming CompCars project

- Convention reminder for later angle classification: left/right refer to **vehicle** left/right, not image left/right.
- The environment is designed for PyTorch + Ultralytics YOLOv8 workflows and includes core libs needed for:
  - dataset parsing/conversion (`scipy`, `pycocotools`, `pandas`)
  - training/inference (`torch`, `torchvision`, `ultralytics`, `opencv-python`)
  - CLI/logging/config (`typer`, `rich`, `pydantic`, `pyyaml`)
  - testing/linting (`pytest`, `ruff`, `pre-commit`)

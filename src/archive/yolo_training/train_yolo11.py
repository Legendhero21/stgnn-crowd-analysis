"""
Train YOLOv11 model for drone-based person detection.

Usage (from project root):
    cd D:/stgnn_project
    .\.venv\Scripts\activate
    python -m src.train_yolo11
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

  
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    print(f"[FATAL] Failed to import ultralytics: {exc}")
    print("Install with: pip install ultralytics")
    sys.exit(1)


# ---------------------------------------------------------------------------#
# Logging setup
# ---------------------------------------------------------------------------#

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_yolo11")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


# ---------------------------------------------------------------------------#
# Configuration
# ---------------------------------------------------------------------------#

@dataclass
class TrainConfig:
    # Paths (adapt these to your layout)
    project_root: Path = Path("D:/stgnn_project").resolve()
    data_yaml: Path = Path("D:/stgnn_project/data/yolo_from_mat/data.yaml").resolve()

    # Model and run naming
    base_model: str = "yolo11n.pt"  # options: yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.
    run_project: Path = Path("D:/stgnn_project/runs/detect").resolve()
    run_name: str = "yolo11_person_drone"

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 4
    img_size: int = 640
    device: Union[int, str, List[int], None] = 0  # 0 for first GPU, "cpu" for CPU
    patience: int = 20
    seed: int = 42

    # Augmentations (drone‑specific)
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    mosaic: float = 0.5

    # Misc
    workers: int = 8
    exist_ok: bool = False  # if False, run_name will be auto‑incremented if it exists


# ---------------------------------------------------------------------------#
# Validation helpers
# ---------------------------------------------------------------------------#

def validate_paths(cfg: TrainConfig) -> bool:
    ok = True

    if not cfg.data_yaml.is_file():
        LOGGER.error("data.yaml not found at: %s", cfg.data_yaml)
        ok = False
    else:
        LOGGER.info("Using dataset config: %s", cfg.data_yaml)

    if not cfg.run_project.exists():
        try:
            cfg.run_project.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Created run project directory: %s", cfg.run_project)
        except Exception as exc:
            LOGGER.error("Failed to create run project directory %s: %s", cfg.run_project, exc)
            ok = False

    return ok


def log_config(cfg: TrainConfig) -> None:
    LOGGER.info("=== YOLOv11 Training Configuration ===")
    for field_name, value in cfg.__dict__.items():
        LOGGER.info("  %s: %s", field_name, value)
    LOGGER.info("=======================================")


# ---------------------------------------------------------------------------#
# Training entry point
# ---------------------------------------------------------------------------#

def train(cfg: Optional[TrainConfig] = None):
    """
    Train YOLOv11 on the configured dataset.

    Args:
        cfg: Optional custom TrainConfig. If None, default is used.
    """
    cfg = cfg or TrainConfig()

    # Validate inputs
    if not validate_paths(cfg):
        LOGGER.error("Configuration validation failed. Aborting training.")
        return None

    log_config(cfg)

    try:
        LOGGER.info("Loading base model: %s", cfg.base_model)
        model = YOLO(cfg.base_model)
    except Exception as exc:
        LOGGER.error("Failed to load base model '%s': %s", cfg.base_model, exc)
        return None

    try:
        LOGGER.info("Starting training...")
        results = model.train(
            data=str(cfg.data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.img_size,
            batch=cfg.batch_size,
            device=cfg.device,
            patience=cfg.patience,
            project=str(cfg.run_project),
            name=cfg.run_name,
            exist_ok=cfg.exist_ok,
            workers=cfg.workers,
            seed=cfg.seed,

            # Augmentations
            hsv_h=cfg.hsv_h,
            hsv_s=cfg.hsv_s,
            hsv_v=cfg.hsv_v,
            degrees=cfg.degrees,
            translate=cfg.translate,
            scale=cfg.scale,
            mosaic=cfg.mosaic,

            # Recommended defaults (can be tuned later)
            optimizer="auto",
            verbose=True,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user.")
        return None
    except Exception as exc:
        LOGGER.error("Unhandled exception during training: %s", exc)
        return None

    # results.save_dir is where weights & logs were stored
    try:
        save_dir = getattr(results, "save_dir", None)
    except Exception:
        save_dir = None

    if save_dir:
        LOGGER.info("Training completed. Results saved to: %s", save_dir)
    else:
        LOGGER.warning("Training finished, but results directory not reported.")

    return results


# ---------------------------------------------------------------------------#
# Script entry
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    train()

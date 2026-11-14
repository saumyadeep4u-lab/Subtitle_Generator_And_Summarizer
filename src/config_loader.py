import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads and validates the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration structure: root must be a YAML mapping.")

    # Ensure all configured directories exist
    for _, dir_path in config.get("paths", {}).items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config


def set_deterministic_seed(seed: int) -> None:
    """Fixes random seeds across all libraries for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """Configures a structured logger writing to both console and logs/pipeline.log."""
    log_dir = Path(config["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

if __name__ == "__main__":
  cfg = load_config()
  set_deterministic_seed(cfg["project"]["random_seed"])
  log = get_logger("SmokeTest", cfg)
  log.info(f"Config loaded successfully: {cfg['project']['name']}")
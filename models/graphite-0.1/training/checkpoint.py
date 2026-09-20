from pathlib import Path
from typing import Any
import random

import torch


CHECKPOINT_VERSION = 2


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    step: int,
    run_id: str,
    scaler: Any | None = None,
) -> None:
    """Save a complete training checkpoint."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "run_id": run_id,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }

    if torch.cuda.is_available():
        checkpoint["rng"]["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore training state."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint: {path}"
        )

    model.load_state_dict(
        checkpoint["model"]
    )

    if (
        optimizer is not None
        and checkpoint.get("optimizer") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if (
        scheduler is not None
        and checkpoint.get("scheduler") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    if (
        scaler is not None
        and checkpoint.get("scaler") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    rng = checkpoint.get("rng", {})

    if "torch" in rng:
        torch.set_rng_state(
            rng["torch"]
        )

    if "python" in rng:
        random.setstate(
            rng["python"]
        )

    if (
        torch.cuda.is_available()
        and "cuda" in rng
    ):
        torch.cuda.set_rng_state_all(
            rng["cuda"]
        )

    return {
        "version": checkpoint.get(
            "version",
            1,
        ),
        "run_id": checkpoint.get(
            "run_id"
        ),
        "step": checkpoint.get(
            "step",
            0,
        ),
    }
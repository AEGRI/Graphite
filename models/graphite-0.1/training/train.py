import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from architecture.model import GraphiteModel
from training.trainer import Trainer


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class TextDataset(Dataset):
    """
    Minimal next-token prediction dataset.

    Expects a sequence of token IDs and produces fixed-length
    input/target pairs.
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        if len(token_ids) <= context_length:
            raise ValueError(
                "Dataset must contain more tokens than "
                "the context length."
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.context_length = context_length

    def __len__(self) -> int:
        return (
            len(self.token_ids)
            - self.context_length
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.token_ids[
            index:
            index + self.context_length
        ]

        targets = self.token_ids[
            index + 1:
            index + self.context_length + 1
        ]

        return input_ids, targets


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_learning_rate_ratio: float,
):
    def learning_rate_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(
                step / max(warmup_steps, 1),
                1e-8,
            )

        progress = (
            step - warmup_steps
        ) / max(
            max_steps - warmup_steps,
            1,
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0
            + torch.cos(
                torch.tensor(
                    progress * torch.pi
                )
            ).item()
        )

        return (
            min_learning_rate_ratio
            + (
                1.0
                - min_learning_rate_ratio
            )
            * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_lambda,
    )


def load_token_ids(
    path: Path,
) -> list[int]:
    """
    Load a prepared token-ID dataset.

    The file must contain one integer token ID per line.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Token dataset does not exist: {path}"
        )

    token_ids = [
        int(line.strip())
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not token_ids:
        raise ValueError(
            f"Token dataset is empty: {path}"
        )

    return token_ids


def main() -> None:
    project_root = Path(
        __file__
    ).resolve().parents[1]

    config_directory = (
        project_root / "config"
    )

    model_config = load_json(
        config_directory / "model.json"
    )

    training_config = load_json(
        config_directory / "training.json"
    )

    architecture = model_config[
        "architecture"
    ]

    training = training_config[
        "training"
    ]

    optimizer_config = training_config[
        "optimizer"
    ]

    scheduler_config = training_config[
        "scheduler"
    ]

    seed = training_config[
        "reproducibility"
    ]["seed"]

    set_seed(seed)

    device = select_device()

    print(f"Device: {device}")

    model = GraphiteModel(
        vocab_size=architecture[
            "vocab_size"
        ],
        context_length=architecture[
            "context_length"
        ],
        hidden_size=architecture[
            "hidden_size"
        ],
        num_layers=architecture[
            "num_layers"
        ],
        num_attention_heads=architecture[
            "num_attention_heads"
        ],
        intermediate_size=architecture[
            "intermediate_size"
        ],
        activation=architecture[
            "activation"
        ],
        normalization_epsilon=architecture[
            "normalization_epsilon"
        ],
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=tuple(
            optimizer_config["betas"]
        ),
        eps=optimizer_config["epsilon"],
        weight_decay=training[
            "weight_decay"
        ],
    )

    min_learning_rate = training[
        "min_learning_rate"
    ]

    min_learning_rate_ratio = (
        min_learning_rate
        / training["learning_rate"]
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=scheduler_config[
            "warmup_steps"
        ],
        max_steps=training[
            "max_steps"
        ],
        min_learning_rate_ratio=(
            min_learning_rate_ratio
        ),
    )

    token_dataset_path = (
        project_root
        / "data"
        / "processed"
        / "tokens.txt"
    )

    token_ids = load_token_ids(
        token_dataset_path
    )

    dataset = TextDataset(
        token_ids=token_ids,
        context_length=architecture[
            "context_length"
        ],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training[
            "batch_size"
        ],
        shuffle=True,
        drop_last=True,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    run_id = (
        f"run-{seed}"
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=training[
            "max_steps"
        ],
        gradient_accumulation_steps=training[
            "gradient_accumulation_steps"
        ],
        gradient_clip_norm=training[
            "gradient_clip_norm"
        ],
        checkpoint_every=training_config[
            "checkpointing"
        ]["save_every_steps"],
        checkpoint_directory=str(
            project_root
            / "training"
            / "runs"
        ),
        run_id=run_id,
        log_every=training_config[
            "logging"
        ]["log_every_steps"],
    )

    trainer.train(
        dataloader=dataloader,
    )


if __name__ == "__main__":
    main()
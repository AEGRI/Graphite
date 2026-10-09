import argparse
import json
import math
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


MODEL_ROOT = Path(__file__).resolve().parents[1]

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from architecture.model import GraphiteModel
from tokenizer.tokenizer import GraphiteTokenizer
from training.checkpoint import load_checkpoint
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
    """Create next-token prediction examples from a token stream."""

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        if context_length < 1:
            raise ValueError("Context length must be positive.")

        if len(token_ids) <= context_length:
            raise ValueError(
                f"Dataset contains {len(token_ids):,} tokens, but "
                f"at least {context_length + 1:,} are required "
                f"for context length {context_length:,}."
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.token_ids[
            index:index + self.context_length
        ]

        targets = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return input_ids, targets


def load_token_ids(
    path: Path,
    vocab_size: int,
) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Token dataset does not exist: {path}"
        )

    token_ids = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()

            if not value:
                continue

            try:
                token_id = int(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid token ID on line {line_number}: {value!r}"
                ) from error

            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"Token ID {token_id} on line {line_number} "
                    f"is outside the configured vocabulary range "
                    f"0–{vocab_size - 1}. Retrain the tokenizer, "
                    "then regenerate the processed token dataset."
                )

            token_ids.append(token_id)

    if not token_ids:
        raise ValueError(
            f"Token dataset is empty: {path}"
        )

    return token_ids


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
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(progress * math.pi)
        )

        return (
            min_learning_rate_ratio
            + (1.0 - min_learning_rate_ratio) * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_lambda,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Graphite."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the configured maximum optimizer steps.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a compatible training checkpoint.",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override the configured context length.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_directory = MODEL_ROOT / "config"

    model_config = load_json(
        config_directory / "model.json"
    )
    training_config = load_json(
        config_directory / "training.json"
    )

    architecture = model_config["architecture"]
    training = training_config["training"]
    optimizer_config = training_config["optimizer"]
    scheduler_config = training_config["scheduler"]

    vocab_size = architecture["vocab_size"]
    configured_context_length = architecture["context_length"]
    context_length = (
        args.context_length
        if args.context_length is not None
        else configured_context_length
    )

    if not 1 <= context_length <= configured_context_length:
        raise ValueError(
            f"Context length must be between 1 and "
            f"{configured_context_length:,}."
        )

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else training["max_steps"]
    )

    if max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")

    seed = training_config["reproducibility"]["seed"]
    set_seed(seed)

    device = select_device()
    print(f"Device: {device}")

    tokenizer_path = (
        MODEL_ROOT / "tokenizer" / "files" / "tokenizer.json"
    )
    tokenizer = GraphiteTokenizer.load(tokenizer_path)

    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"Vocabulary mismatch: model.json specifies {vocab_size:,} "
            f"tokens, but tokenizer.json contains {tokenizer.vocab_size:,}. "
            "Update the model configuration and regenerate the token "
            "dataset before training."
        )

    token_dataset_path = (
        MODEL_ROOT / "data" / "processed" / "tokens.txt"
    )
    token_ids = load_token_ids(
        token_dataset_path,
        vocab_size=vocab_size,
    )

    print(f"Token count: {len(token_ids):,}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Context length: {context_length:,}")

    dataset = TextDataset(
        token_ids=token_ids,
        context_length=context_length,
    )

    print(f"Training examples: {len(dataset):,}")

    batch_size = training["batch_size"]
    gradient_accumulation_steps = training[
        "gradient_accumulation_steps"
    ]

    if batch_size < 1:
        raise ValueError("Configured batch size must be positive.")

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "Gradient accumulation steps must be positive."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = GraphiteModel(
        vocab_size=vocab_size,
        context_length=configured_context_length,
        hidden_size=architecture["hidden_size"],
        num_layers=architecture["num_layers"],
        num_attention_heads=architecture["num_attention_heads"],
        intermediate_size=architecture["intermediate_size"],
        activation=architecture["activation"],
        normalization_epsilon=architecture["normalization_epsilon"],
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(f"Parameters: {parameter_count:,}")

    learning_rate = training["learning_rate"]
    min_learning_rate = training["min_learning_rate"]

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    if not 0 <= min_learning_rate <= learning_rate:
        raise ValueError(
            "Minimum learning rate must be between zero "
            "and the configured learning rate."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=training["weight_decay"],
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=min(
            scheduler_config["warmup_steps"],
            max_steps,
        ),
        max_steps=max_steps,
        min_learning_rate_ratio=min_learning_rate / learning_rate,
    )

    start_step = 0

    if args.resume is not None:
        checkpoint_path = Path(args.resume)

        if not checkpoint_path.is_absolute():
            checkpoint_path = MODEL_ROOT / checkpoint_path

        checkpoint_path = checkpoint_path.resolve()

        metadata = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

        start_step = metadata["step"]

        print(f"Resumed checkpoint: {checkpoint_path}")
        print(f"Resuming from step: {start_step}")

        if start_step >= max_steps:
            raise ValueError(
                f"Checkpoint is already at step {start_step}, "
                f"but the requested maximum is {max_steps}. "
                "Increase --max-steps to continue training."
            )

    run_id = f"run-{seed}"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=training["gradient_clip_norm"],
        checkpoint_every=training_config["checkpointing"][
            "save_every_steps"
        ],
        checkpoint_directory=str(MODEL_ROOT / "training" / "runs"),
        run_id=run_id,
        log_every=training_config["logging"]["log_every_steps"],
    )

    trainer.train(
        dataloader=dataloader,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()

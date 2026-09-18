import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_json(path: str | Path) -> dict:
    """
    Load a JSON configuration or metadata file.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate a language model using next-token prediction loss.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(
            dataloader
        ):
            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            input_ids, targets = batch

            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError(
            "Evaluation dataset produced no tokens."
        )

    average_loss = (
        total_loss / total_tokens
    )

    perplexity = math.exp(
        min(average_loss, 20.0)
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    model.train()

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
    }


def save_results(
    results: dict,
    output_path: str | Path,
) -> None:
    """
    Save evaluation results as a JSON artifact.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """
    Evaluation entry point.

    Model loading and dataset construction will be connected
    once the inference and checkpoint pipelines are integrated.
    """

    project_root = Path(
        __file__
    ).resolve().parents[1]

    results_directory = (
        project_root
        / "eval"
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "schema_version": 1,
        "status": "not_run",
        "message": (
            "Evaluation pipeline is ready. "
            "Model and dataset loading will be "
            "connected during integration."
        ),
    }

    output_path = (
        results_directory
        / "evaluation.json"
    )

    save_results(
        results,
        output_path,
    )

    print(
        f"Evaluation result written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
import json
import sys
from pathlib import Path

import torch


# Allow imports from the graphite-0.1 model directory.
MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from architecture.model import GraphiteModel


def load_config() -> dict:
    config_path = MODEL_ROOT / "config" / "model.json"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_model() -> None:
    config = load_config()
    architecture = config["architecture"]

    model = GraphiteModel(
        vocab_size=architecture["vocab_size"],
        context_length=architecture["context_length"],
        hidden_size=architecture["hidden_size"],
        num_layers=architecture["num_layers"],
        num_attention_heads=architecture["num_attention_heads"],
        intermediate_size=architecture["intermediate_size"],
        activation=architecture["activation"],
        normalization_epsilon=architecture[
            "normalization_epsilon"
        ],
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    batch_size = 2
    sequence_length = 16

    input_ids = torch.randint(
        low=0,
        high=architecture["vocab_size"],
        size=(batch_size, sequence_length),
        device=device,
    )

    print(f"Device: {device}")
    print(f"Input shape: {tuple(input_ids.shape)}")

    logits = model(input_ids)

    expected_shape = (
        batch_size,
        sequence_length,
        architecture["vocab_size"],
    )

    assert logits.shape == expected_shape, (
        f"Unexpected output shape: "
        f"{tuple(logits.shape)}; "
        f"expected {expected_shape}"
    )

    targets = torch.randint(
        low=0,
        high=architecture["vocab_size"],
        size=(batch_size, sequence_length),
        device=device,
    )

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    assert torch.isfinite(loss), (
        f"Loss is not finite: {loss.item()}"
    )

    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    gradients_found = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_found, (
        "No parameter gradients were produced."
    )

    print("Backward pass: OK")
    print("Model test: PASS")


if __name__ == "__main__":
    test_model()
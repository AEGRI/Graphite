import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from inference.generate import generate
from inference.runtime import GraphiteRuntime
from tokenizer.tokenizer import GraphiteTokenizer


MODEL_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_ROOT = Path(__file__).resolve().parent

MODEL_CONFIG_PATH = MODEL_ROOT / "config" / "model.json"
INFERENCE_CONFIG_PATH = MODEL_ROOT / "config" / "inference.json"
TOKENIZER_PATH = (
    MODEL_ROOT
    / "tokenizer"
    / "files"
    / "tokenizer.json"
)


def load_inference_config() -> dict[str, Any]:
    return json.loads(
        INFERENCE_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def resolve_checkpoint(
    config: dict[str, Any],
) -> Path:
    """
    Resolve the checkpoint path from configuration.

    Supports both the current Graphite configuration layout
    and the canonical inference configuration layout.
    """

    model_config = config.get("model", {})

    checkpoint = model_config.get(
        "checkpoint"
    )

    if checkpoint is None:
        checkpoint = config.get(
            "checkpoint"
        )

    if checkpoint is None:
        checkpoint = os.environ.get(
            "GRAPHITE_CHECKPOINT"
        )

    if not checkpoint:
        raise RuntimeError(
            "No Graphite checkpoint is configured. "
            "Set model.checkpoint in inference.json "
            "or GRAPHITE_CHECKPOINT in the environment."
        )

    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = MODEL_ROOT / checkpoint_path

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Graphite checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    return checkpoint_path


def load_runtime(
    config: dict[str, Any],
) -> GraphiteRuntime:
    runtime = GraphiteRuntime.from_config(
        MODEL_CONFIG_PATH,
        INFERENCE_CONFIG_PATH,
    )

    checkpoint_path = resolve_checkpoint(
        config
    )

    runtime.load_checkpoint(
        checkpoint_path
    )

    return runtime


def generate_text(
    prompt: str,
    config: dict[str, Any],
) -> str:
    tokenizer = GraphiteTokenizer.load(
        TOKENIZER_PATH
    )

    runtime = load_runtime(
        config
    )

    generation_config = config.get(
        "generation",
        config.get("inference", {}),
    )

    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
    )

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=runtime.device,
    )

    generated_ids = generate(
        runtime=runtime,
        input_ids=input_tensor,
        max_new_tokens=generation_config.get(
            "max_new_tokens",
            256,
        ),
        temperature=generation_config.get(
            "temperature",
            0.8,
        ),
        top_k=generation_config.get(
            "top_k",
            50,
        ),
        top_p=generation_config.get(
            "top_p",
            0.95,
        ),
        do_sample=generation_config.get(
            "do_sample",
            True,
        ),
        seed=generation_config.get(
            "seed",
        ),
    )

    generated_ids = generated_ids[
        0
    ].detach().cpu().tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def handle_request(
    request: dict[str, Any],
) -> dict[str, Any]:
    prompt = request.get(
        "prompt"
    )

    if not isinstance(prompt, str):
        raise ValueError(
            "request field 'prompt' must be a string."
        )

    if not prompt.strip():
        raise ValueError(
            "request field 'prompt' cannot be empty."
        )

    config = load_inference_config()

    text = generate_text(
        prompt,
        config,
    )

    return {
        "text": text,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: bridge.py '<json request>'",
            file=sys.stderr,
        )
        return 1

    try:
        request = json.loads(
            sys.argv[1]
        )

        if not isinstance(request, dict):
            raise ValueError(
                "request must be a JSON object."
            )

        response = handle_request(
            request
        )

        print(
            json.dumps(
                response,
                ensure_ascii=False,
            )
        )

        return 0

    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
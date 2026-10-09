
import json
import os
import sys
from pathlib import Path
from typing import Any

MODEL_ROOT = Path(__file__).resolve().parent.parent

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import torch

from inference.generate import generate
from inference.runtime import GraphiteRuntime
from tokenizer.tokenizer import GraphiteTokenizer


MODEL_CONFIG_PATH = MODEL_ROOT / "config" / "model.json"
INFERENCE_CONFIG_PATH = MODEL_ROOT / "config" / "inference.json"
TOKENIZER_PATH = MODEL_ROOT / "tokenizer" / "files" / "tokenizer.json"


class GraphiteBridge:
    """Persistent JSON-lines inference bridge for Graphite."""

    def __init__(self):
        self.config = self.load_inference_config()
        self.tokenizer = GraphiteTokenizer.load(TOKENIZER_PATH)
        self.runtime = self.load_runtime(self.config)
        self.generation_config = self.config.get(
            "generation",
            self.config.get("inference", {}),
        )

    @staticmethod
    def load_inference_config() -> dict[str, Any]:
        return json.loads(
            INFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
        )

    @staticmethod
    def resolve_checkpoint(config: dict[str, Any]) -> Path:
        # An environment override must take precedence over the config.
        checkpoint = os.environ.get("GRAPHITE_CHECKPOINT")

        if not checkpoint:
            checkpoint = config.get("model", {}).get("checkpoint")

        if not checkpoint:
            checkpoint = config.get("checkpoint")

        if not checkpoint:
            raise RuntimeError(
                "No checkpoint configured. Set model.checkpoint "
                "in inference.json or GRAPHITE_CHECKPOINT."
            )

        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.is_absolute():
            checkpoint_path = MODEL_ROOT / checkpoint_path

        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Graphite checkpoint does not exist: {checkpoint_path}"
            )

        return checkpoint_path

    @classmethod
    def load_runtime(cls, config: dict[str, Any]) -> GraphiteRuntime:
        checkpoint_path = cls.resolve_checkpoint(config)

        print(
            f"[Graphite] Loading checkpoint: {checkpoint_path}",
            file=sys.stderr,
            flush=True,
        )

        runtime = GraphiteRuntime.from_config(
            MODEL_CONFIG_PATH,
            INFERENCE_CONFIG_PATH,
        )
        runtime.load_checkpoint(checkpoint_path)
        return runtime

    def generate_text(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("request field 'prompt' must be a string.")

        if not prompt.strip():
            raise ValueError("request field 'prompt' cannot be empty.")

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.runtime.device,
        )

        generated = generate(
            runtime=self.runtime,
            input_ids=input_tensor,
            max_new_tokens=self.generation_config.get("max_new_tokens", 256),
            temperature=self.generation_config.get("temperature", 0.8),
            top_k=self.generation_config.get("top_k", 50),
            top_p=self.generation_config.get("top_p", 0.95),
            do_sample=self.generation_config.get("do_sample", True),
            seed=self.generation_config.get("seed"),
        )

        all_ids = generated[0].detach().cpu().tolist()
        response_ids = all_ids[len(input_ids):]

        # Keep terminal diagnostics short.
        preview_count = 32
        preview = response_ids[:preview_count]
        suffix = " ..." if len(response_ids) > preview_count else ""

        print(
            f"[Graphite] Generated {len(response_ids)} tokens: "
            f"{preview}{suffix}",
            file=sys.stderr,
            flush=True,
        )

        try:
            return self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )
        except (ValueError, KeyError) as error:
            print(
                f"[Graphite] Decoding failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            return (
                f"[Raw token output: {len(response_ids)} tokens; "
                f"first {len(preview)} IDs: {preview}{suffix}]"
            )

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object.")

        return {"text": self.generate_text(request.get("prompt"))}

    def run(self) -> int:
        for line in sys.stdin:
            line = line.strip()

            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response, ensure_ascii=False), flush=True)
            except Exception as error:
                print(
                    json.dumps({"error": str(error)}, ensure_ascii=False),
                    flush=True,
                )

        return 0


def main() -> int:
    try:
        return GraphiteBridge().run()
    except Exception as error:
        print(
            json.dumps({"error": str(error)}, ensure_ascii=False),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

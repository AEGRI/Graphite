import json
from pathlib import Path

import torch

from architecture.model import GraphiteModel


class GraphiteRuntime:
    """
    Runtime environment for loading and executing Graphite.
    """

    def __init__(
        self,
        model: GraphiteModel,
        device: torch.device,
    ):
        self.model = model
        self.device = device

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def select_device(
        requested_device: str = "auto",
    ) -> torch.device:
        """
        Select an available execution device.
        """

        if requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but no CUDA "
                    "device is available."
                )

            return torch.device("cuda")

        if requested_device == "cpu":
            return torch.device("cpu")

        return torch.device(
            requested_device
        )

    @classmethod
    def from_config(
        cls,
        model_config_path: str | Path,
        inference_config_path: str | Path,
    ) -> "GraphiteRuntime":
        """
        Construct a runtime from Graphite configuration files.
        """

        model_config_path = Path(
            model_config_path
        )

        inference_config_path = Path(
            inference_config_path
        )

        model_config = json.loads(
            model_config_path.read_text(
                encoding="utf-8"
            )
        )

        inference_config = json.loads(
            inference_config_path.read_text(
                encoding="utf-8"
            )
        )

        architecture = model_config[
            "architecture"
        ]

        runtime_config = inference_config[
            "runtime"
        ]

        device = cls.select_device(
            runtime_config.get(
                "device",
                "auto",
            )
        )

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

        return cls(
            model=model,
            device=device,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict:
        """
        Load model weights from a Graphite checkpoint.
        """

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model weights."
            )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.to(self.device)
        self.model.eval()

        return {
            "run_id": checkpoint.get(
                "run_id"
            ),
            "step": checkpoint.get(
                "step",
                0,
            ),
        }

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the model on token IDs and return logits.
        """

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        return self.model(
            input_ids
        )
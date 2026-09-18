import torch
import torch.nn as nn

from .embeddings import TokenEmbedding
from .normalization import RMSNorm
from .transformer import TransformerBlock


class GraphiteModel(nn.Module):
    """
    Graphite 0.1 decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        activation: str = "silu",
        normalization_epsilon: float = 1e-5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    context_length=context_length,
                    normalization_epsilon=normalization_epsilon,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            hidden_size,
            eps=normalization_epsilon,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize model parameters.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the model.

        Args:
            input_ids:
                Token IDs with shape [batch, sequence].

            position_ids:
                Optional position IDs used by RoPE.

        Returns:
            Logits with shape [batch, sequence, vocab_size].
        """

        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]."
            )

        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds "
                f"context length {self.context_length}."
            )

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(
                x,
                position_ids=position_ids,
            )

        x = self.final_norm(x)

        logits = self.output_projection(x)

        return logits
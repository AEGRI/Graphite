import torch

from .runtime import GraphiteRuntime


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample the next token from model logits.
    """

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0."
        )

    logits = logits / temperature

    if top_k > 0:
        top_k = min(
            top_k,
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            top_k,
        )

        threshold = values[..., -1, None]

        logits = torch.where(
            logits < threshold,
            torch.full_like(
                logits,
                torch.finfo(logits.dtype).min,
            ),
            logits,
        )

    if top_p < 1.0:
        if not 0.0 < top_p <= 1.0:
            raise ValueError(
                "top_p must be between 0 and 1."
            )

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = (
            torch.cumsum(
                sorted_probabilities,
                dim=-1,
            )
        )

        remove_tokens = (
            cumulative_probabilities
            > top_p
        )

        remove_tokens[..., 1:] = (
            remove_tokens[..., :-1].clone()
        )

        remove_tokens[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_tokens,
            torch.finfo(sorted_logits.dtype).min,
        )

        logits = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )

        logits.scatter_(
            -1,
            sorted_indices,
            sorted_logits,
        )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


@torch.no_grad()
def generate(
    runtime: GraphiteRuntime,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate new token IDs autoregressively.
    """

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative."
        )

    if input_ids.dim() != 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence]."
        )

    if seed is not None:
        generator = torch.Generator(
            device=runtime.device
        )

        generator.manual_seed(seed)
    else:
        generator = None

    generated_ids = input_ids.to(
        runtime.device
    )

    context_length = (
        runtime.model.context_length
    )

    for _ in range(max_new_tokens):
        model_input = generated_ids[
            :, -context_length:
        ]

        logits = runtime.forward(
            model_input
        )

        next_token_logits = logits[
            :, -1, :
        ]

        if do_sample:
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if generator is not None:
                probabilities = torch.softmax(
                    next_token_logits / temperature,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )

        else:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token,
            ],
            dim=-1,
        )

    return generated_ids
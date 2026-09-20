import time
from typing import Callable

import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint


class Trainer:
    """Training engine for Graphite.

    Handles forward passes, loss calculation, gradient accumulation,
    optimization, learning-rate scheduling, gradient clipping,
    logging, and checkpointing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        max_steps: int,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
        checkpoint_every: int = 500,
        checkpoint_directory: str = "runs",
        run_id: str = "default",
        log_every: int = 10,
        scaler=None,
        logger: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_steps = max_steps
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_every = checkpoint_every
        self.checkpoint_directory = checkpoint_directory
        self.run_id = run_id
        self.log_every = log_every
        self.scaler = scaler
        self.logger = logger or print
        self.step = 0

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate next-token prediction loss."""

        logits = self.model(input_ids)

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def _train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Perform one gradient-accumulation micro-step."""

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )

        if self.scaler is not None:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=True,
            ):
                loss = self._compute_loss(
                    input_ids,
                    targets,
                )

            scaled_loss = (
                loss
                / self.gradient_accumulation_steps
            )

            self.scaler.scale(
                scaled_loss
            ).backward()

        else:
            loss = self._compute_loss(
                input_ids,
                targets,
            )

            scaled_loss = (
                loss
                / self.gradient_accumulation_steps
            )

            scaled_loss.backward()

        return loss.detach().item()

    def _optimizer_step(self) -> None:
        """Apply accumulated gradients."""

        if self.scaler is not None:
            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

            self.optimizer.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        if self.scheduler is not None:
            self.scheduler.step()

    def train(
        self,
        dataloader,
        start_step: int = 0,
    ) -> None:
        """Train the model until max_steps is reached."""

        self.step = start_step

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        data_iterator = iter(dataloader)

        step_start_time = time.perf_counter()

        while self.step < self.max_steps:
            accumulated_loss = 0.0

            for _ in range(
                self.gradient_accumulation_steps
            ):
                try:
                    input_ids, targets = next(
                        data_iterator
                    )

                except StopIteration:
                    data_iterator = iter(
                        dataloader
                    )

                    input_ids, targets = next(
                        data_iterator
                    )

                accumulated_loss += (
                    self._train_step(
                        input_ids,
                        targets,
                    )
                )

            self._optimizer_step()

            self.step += 1

            if (
                self.step % self.log_every == 0
                or self.step == 1
            ):
                elapsed = (
                    time.perf_counter()
                    - step_start_time
                )

                average_loss = (
                    accumulated_loss
                    / self.gradient_accumulation_steps
                )

                learning_rate = (
                    self.optimizer.param_groups[0][
                        "lr"
                    ]
                )

                self.logger(
                    f"step={self.step} "
                    f"loss={average_loss:.4f} "
                    f"lr={learning_rate:.6g} "
                    f"time={elapsed:.2f}s"
                )

                step_start_time = (
                    time.perf_counter()
                )

            if (
                self.checkpoint_every > 0
                and self.step
                % self.checkpoint_every
                == 0
            ):
                self.save_checkpoint()

        # Always save the final state, even when the
        # run ends before checkpoint_every is reached.
        if (
            self.max_steps > 0
            and self.step % self.checkpoint_every != 0
        ):
            self.save_checkpoint()

    def save_checkpoint(self) -> str:
        """Save the current training state."""

        checkpoint_path = self._checkpoint_path()

        save_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            run_id=self.run_id,
            scaler=self.scaler,
        )

        self.logger(
            f"checkpoint={checkpoint_path}"
        )

        return checkpoint_path

    def _checkpoint_path(self) -> str:
        """Generate the checkpoint filename."""

        return (
            f"{self.checkpoint_directory}/"
            f"graphite-{self.run_id}-"
            f"step-{self.step}.ckpt"
        )
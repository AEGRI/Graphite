# Training

This directory contains the training system used to train Graphite 0.1.

## Purpose

The training system defines how Graphite's model architecture is optimized using prepared training data. It manages the training process, optimization, checkpoints, and experiment state.

## Contains

* Training implementation
* Training loops
* Optimizers and learning-rate schedules
* Loss functions
* Checkpointing and resume logic
* Distributed or multi-device training support
* Training metrics and logging
* Training utilities
* Experiment configurations specific to training

## Does Not Contain

* Model architecture implementation
* Tokenizer implementation
* Dataset definitions or raw data
* Evaluation benchmarks
* Inference/runtime systems
* General-purpose configuration

Those responsibilities belong to their respective directories.

## Conventions

Training should be reproducible from version-controlled code, configuration, data, tokenizer, and documented environment information.

Training runs should record relevant configuration, dataset and tokenizer versions, random seeds, dependency versions, checkpoints, and metrics.

Training code should not modify the model architecture implicitly. Architecture changes should be made explicitly within `architecture/`.

## Training-Run Reproducibility

Every training run should have a unique run identifier and record the information required to reproduce or investigate the run.

A run should record, where applicable:

* Graphite model version
* Source-code commit
* Model configuration
* Training configuration
* Dataset version or manifest
* Tokenizer version
* Random seeds
* Dependency and framework versions
* Hardware and relevant runtime settings
* Starting checkpoint, if any
* Training start time
* Training metrics and evaluation results
* Final checkpoint

Reproducibility-critical changes should result in a new run rather than silently modifying an existing run.

## Checkpoints

Checkpoints represent a saved state of the training process.

Checkpoint filenames should use the following format:

```text
{model}-{run_id}-step-{step}.ckpt
```

Example:

```text
graphite-0.1-run-0001-step-10000.ckpt
```

A checkpoint should contain, where applicable:

* Model weights
* Optimizer state
* Learning-rate scheduler state
* Current training step
* Current epoch
* Random-number-generator state
* Gradient-scaler state for mixed-precision training
* Model and training configuration references
* Metadata required to resume training

Checkpoints intended only for inference may omit training-only state and should be clearly identified.

Checkpoint metadata should identify the exact training run and source configuration that produced it.

## Checkpoint Retention

Checkpoint retention policy should be defined by the training configuration.

At minimum, training systems should support retaining:

* The latest checkpoint
* The best checkpoint according to a defined metric
* Explicitly marked checkpoints intended for long-term preservation

Large checkpoint files should be stored outside normal source-code version control unless specifically intended for repository storage.

## Interactions

```text
config/ ──────────┐
data/ ────────────┤
tokenizer/ ───────┤
architecture/ ────┤
                  ↓
              training/
                  ↓
             model weights
                  ↓
             eval/ / inference/
```

The training system consumes the model definition, configuration, tokenizer, and prepared data to produce trained model weights and training artifacts.

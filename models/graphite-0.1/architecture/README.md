# Architecture

This directory defines the neural network architecture used by Graphite 0.1.

## Purpose

The architecture determines how Graphite processes tokens and transforms them into model outputs. It contains the implementation and definitions of the model's neural network components.

## Contains

* Model architecture definitions
* Transformer and attention components
* Embedding layers
* Feed-forward/network blocks
* Model-specific neural network components
* Architecture-related utilities

## Does Not Contain

* Training logic
* Dataset processing
* Tokenizer implementation
* Inference/runtime code
* Evaluation logic
* General configuration

Those responsibilities belong to their respective directories.

## Dependencies

Architecture code may depend on:

* `config/` for architecture parameters
* The selected machine-learning framework and its dependencies
* Other components within `architecture/`

It should remain independent of training and application-specific logic.

## Conventions

Architecture code should define **what the model is**, not how it is trained or used.

Changes that alter the neural network structure should be treated as architecture changes and may require a new Graphite model version.

## Interactions

```text
config/
   ↓
architecture/
   ↓
training/ ──→ inference/
   ↓
eval/
```

The architecture is the central model definition used by training, inference, and evaluation.

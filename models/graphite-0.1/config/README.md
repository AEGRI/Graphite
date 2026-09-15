# Configuration

This directory contains the configuration files that define how Graphite 0.1 is built, trained, evaluated, and run.

## Purpose

Configuration separates model parameters and runtime settings from the code that uses them. This allows Graphite's architecture, training process, and inference system to be changed or reproduced without hardcoding settings into their implementations.

## Contains

* Model architecture parameters
* Training parameters
* Optimization settings
* Dataset and data-processing settings
* Runtime and inference settings
* Experiment-specific configuration
* Configuration schemas or defaults

## Does Not Contain

* Model architecture implementation
* Training implementation
* Dataset files
* Tokenizer implementation
* Evaluation implementation
* Inference implementation

Those responsibilities belong to their respective directories.

## Conventions

Configuration should contain **declarative settings**, not application logic.

Settings that affect the structure or behavior of Graphite should be explicitly defined and version-controlled so that experiments and model versions can be reproduced.

## Interactions

```text
config/
 ├──→ architecture/
 ├──→ training/
 ├──→ data/
 ├──→ inference/
 └──→ eval/
```

Other subsystems consume configuration rather than defining their own conflicting copies of model settings.

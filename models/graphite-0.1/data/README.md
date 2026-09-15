# Data

This directory contains the data pipeline and dataset definitions used by Graphite 0.1.

## Purpose

The data system prepares and organizes the information used to train and evaluate Graphite. It defines how raw data is collected, processed, validated, stored, and provided to the training system.

## Contains

* Dataset definitions
* Data preprocessing and cleaning
* Dataset manifests and metadata
* Data filtering and validation
* Training, validation, and test splits
* Data loading and processing utilities
* Dataset-specific documentation

## Does Not Contain

* Model architecture
* Tokenizer implementation
* Training logic
* Evaluation logic
* Inference/runtime code
* Raw datasets that are not intended to be version-controlled

Those responsibilities belong to their respective directories or external data storage.

## Conventions

Data processing should be reproducible and deterministic where practical.

Changes to preprocessing, filtering, dataset composition, or data splits should be documented because they can affect model behavior and training results.

Large datasets should not be committed directly to the repository unless explicitly intended.

## Interactions

```text
Raw Data
   ↓
data/
   ↓
tokenizer/
   ↓
training/
   ↓
eval/
```

The data pipeline provides processed training and evaluation data to the rest of Graphite 0.1.

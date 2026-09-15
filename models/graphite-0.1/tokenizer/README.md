# Tokenizer

This directory contains the tokenizer used by Graphite 0.1 to convert raw text into tokens that can be processed by the model.

## Purpose

The tokenizer defines how text is represented as a sequence of token IDs. It provides the interface between input data and the model's embedding layer.

## Contains

* Tokenizer implementation
* Vocabulary
* Tokenizer configuration
* Special tokens
* Tokenization and detokenization utilities
* Tokenizer training or generation scripts
* Tokenizer metadata

## Does Not Contain

* Model architecture
* Dataset processing
* Training logic
* Evaluation logic
* Inference logic unrelated to tokenization
* Raw training data

Those responsibilities belong to their respective directories.

## Conventions

The tokenizer should be versioned together with the Graphite model version that depends on it.

Changes to the vocabulary, tokenization rules, special tokens, or tokenizer configuration may make existing model weights incompatible and should therefore be treated as model-affecting changes.

## Reproducibility

A tokenizer must be reproducible from version-controlled source and configuration.

The repository should record enough information to recreate the exact tokenizer, including:

* Tokenizer algorithm and implementation version
* Vocabulary and token IDs
* Special tokens and their IDs
* Normalization and preprocessing rules
* Training data version or dataset manifest
* Tokenizer training parameters
* Random seed where applicable
* Relevant dependency versions

Given the same source, configuration, dataset version, and dependencies, rebuilding the tokenizer should produce an identical tokenizer or a documented equivalent.

Changes to any reproducibility-critical component must be documented and associated with the Graphite model version that uses it.

## Interactions

```text
data/
   ↓
tokenizer/
   ↓
architecture/
   ↓
training/
   ↓
eval/ / inference/
```

The tokenizer converts processed data into token IDs for training and converts model output token IDs back into readable text.

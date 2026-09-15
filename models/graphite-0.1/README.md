# Graphite 0.1

Graphite 0.1 is the first model generation of the Graphite project, developed under AEGRI.

## Purpose

Graphite is intended to become a capable general-purpose AI model with a strong emphasis on interpretation, reasoning, programming, debugging, and technical problem solving.

Graphite 0.1 serves as the foundation for experimentation with the model architecture, training process, data pipeline, tokenizer, inference system, and evaluation methodology.

## Model Goals

Graphite 0.1 is being developed with the following goals:

* Strong understanding and interpretation of technical information
* Strong programming and debugging capabilities
* Ability to reason across large amounts of information
* Ability to understand relationships between files, systems, and concepts
* Foundation for future tool-use and agentic capabilities
* Foundation for future multimodal capabilities
* Architecture and training processes that can be improved and scaled over time

## Components

### Architecture

Contains the implementation of Graphite 0.1's neural network architecture and its individual model components.

### Config

Contains configuration used to define and train Graphite 0.1.

### Data

Contains the data-processing components used to prepare training and evaluation data for Graphite 0.1.

### Evaluation

Contains benchmarks, evaluation methods, and other systems used to measure Graphite 0.1's capabilities.

### Inference

Contains the systems required to load and run Graphite 0.1 outside of training.

### Tokenizer

Contains Graphite 0.1's tokenizer, vocabulary, and related tokenization components.

### Training

Contains the training implementation and supporting components used to train Graphite 0.1.

## Development Status

Graphite 0.1 is an active research and development project.

The architecture, training methodology, datasets, and implementation may change substantially during development.

This version should be considered experimental until its training and evaluation systems have been established.

## Design Philosophy

Graphite is being developed as a model that improves through increasingly effective interpretation of information and examples.

Training decisions should therefore prioritize data quality, useful reasoning behavior, generalization, and measurable improvements rather than simply increasing model size.

Graphite 0.1 is also intended to provide a foundation that can be studied, modified, and improved in future model generations.

## Directory Structure

```text
graphite-0.1/
├── architecture/
├── config/
├── data/
├── eval/
├── inference/
├── tokenizer/
└── training/
```

Each directory contains its own README describing its specific responsibilities, conventions, and implementation details.

## Versioning

Graphite 0.1 represents a distinct model generation.

Changes that substantially alter the model architecture, training methodology, or other fundamental characteristics may be carried into a future Graphite version rather than silently changing the definition of Graphite 0.1.

## License

Graphite is distributed under the license specified in the repository's `LICENSE` file.

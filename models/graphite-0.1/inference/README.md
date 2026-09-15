# Inference

This directory contains the runtime system used to load and execute Graphite 0.1 for inference.

## Purpose

The inference system loads trained Graphite checkpoints, processes input through the tokenizer and model, and produces model outputs.

## Contains

* Model loading
* Checkpoint loading
* Text generation
* Token generation and sampling
* Inference configuration
* Runtime utilities
* Device and memory management
* Generation and decoding utilities
* Inference benchmarks

## Does Not Contain

* Model architecture implementation
* Training implementation
* Dataset preparation
* Tokenizer training
* Evaluation definitions

Those responsibilities belong to their respective directories.

## Conventions

Inference should use the exact model architecture and tokenizer versions expected by the loaded checkpoint.

Runtime settings that affect generation should be explicitly configurable and recorded when reproducibility matters.

Inference code should not modify model weights unless a separate, explicitly defined mechanism requires it.

## Reproducibility

Inference runs should record, where applicable:

* Model version
* Checkpoint identifier
* Tokenizer version
* Inference configuration
* Generation parameters
* Random seed
* Source-code commit
* Relevant dependency versions
* Hardware and runtime information

Deterministic generation should be supported where practical.

## Inference Configuration

Inference configuration should be declarative and version-controlled.

The canonical configuration schema is:

```json id="7s4k2m"
{
  "schema_version": "1.0",
  "model": {
    "name": "graphite",
    "version": "0.1",
    "checkpoint": "<checkpoint-id>"
  },
  "tokenizer": {
    "version": "<tokenizer-version>"
  },
  "generation": {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "do_sample": true,
    "seed": 0
  },
  "runtime": {
    "device": "<device>",
    "dtype": "<dtype>"
  }
}
```

Required fields are:

* `schema_version`
* `model`
* `tokenizer`
* `generation`
* `runtime`

Generation parameters must be recorded when they can affect output.

Additional configuration fields may be added without changing the meaning of existing fields.

## Result Artifacts

An inference run may produce a result artifact containing the generated output and the metadata required to reproduce the run.

The canonical result artifact uses JSON:

```json id="x5n8qp"
{
  "schema_version": "1.0",
  "inference_id": "0001",
  "model": {
    "name": "graphite",
    "version": "0.1",
    "checkpoint": "<checkpoint-id>"
  },
  "tokenizer": {
    "version": "<tokenizer-version>"
  },
  "configuration": "<inference-config>",
  "input": {
    "text": "<input>"
  },
  "output": {
    "text": "<generated-output>"
  },
  "runtime": {},
  "timestamp": "<timestamp>",
  "status": "completed"
}
```

Required fields are:

* `schema_version`
* `inference_id`
* `model`
* `tokenizer`
* `configuration`
* `input`
* `output`
* `runtime`
* `timestamp`
* `status`

For large inputs or outputs, the artifact may store references to external files instead of embedding the complete contents.

## Artifact Storage

Inference artifacts are stored under the model version's inference directory:

```text id="6m0q2d"
models/
└── graphite-0.1/
    └── inference/
        ├── results/
        ├── outputs/
        └── logs/
```

* `results/` contains structured inference result artifacts.
* `outputs/` contains large generated outputs.
* `logs/` contains inference execution logs.

Large generated artifacts should not be committed to Git by default.

## Naming

Inference artifacts should use:

```text id="h2z9wc"
{model}-{run_id}-inference-{inference_id}.{extension}
```

Example:

```text id="r7p1kx"
graphite-0.1-run-0001-inference-0001.json
```

The `inference_id` uniquely identifies an inference execution. Re-running an inference should create a new ID rather than silently replacing an existing result.

## Model Loading

The inference system should validate that the selected checkpoint is compatible with:

* The Graphite architecture version
* The tokenizer version
* Required configuration
* The inference runtime

Incompatible combinations should produce a clear error rather than silently proceeding.

## Interactions

```text id="p8c3vn"
checkpoint ───────┐
architecture/ ────┤
config/ ──────────┤
tokenizer/ ───────┤
                  ↓
              inference/
                  ↓
               output
```

The inference system provides the runtime interface between trained Graphite models and applications or users.

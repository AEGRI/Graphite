# Evaluation

This directory contains the evaluation system used to measure and analyze Graphite 0.1.

## Purpose

The evaluation system measures model performance, compares model versions and training runs, and provides reproducible results for determining whether changes improve Graphite.

## Contains

* Evaluation tasks and benchmarks
* Evaluation datasets and manifests
* Evaluation scripts
* Metrics and scoring methods
* Baselines and reference results
* Evaluation reports
* Model comparison utilities
* Evaluation-specific configuration

## Does Not Contain

* Model architecture implementation
* Training implementation
* Tokenizer implementation
* General dataset processing
* Inference implementation

Those responsibilities belong to their respective directories.

## Conventions

Evaluation should be reproducible and independent of the training process.

Each evaluation should record:

* Model version
* Model checkpoint
* Source-code commit
* Tokenizer version
* Evaluation dataset version
* Evaluation configuration
* Metrics and scoring configuration
* Relevant dependency versions
* Evaluation results

Changes to evaluation datasets, prompts, metrics, or scoring procedures should be documented because they can affect comparisons between results.

## Reproducibility

Evaluation results should be traceable to the exact model, data, tokenizer, configuration, and evaluation code used to produce them.

Results should not be considered directly comparable when evaluation conditions differ unless those differences are explicitly documented.

## Result Artifacts

Each evaluation run produces a result artifact containing the structured results and metadata for that evaluation.

### Artifact Schema

The canonical result artifact uses JSON and must contain:

```json
{
  "schema_version": "1.0",
  "evaluation_id": "0001",
  "model": {
    "name": "graphite",
    "version": "0.1",
    "checkpoint": "run-0001-step-10000"
  },
  "source": {
    "commit": "<git-commit>"
  },
  "tokenizer": {
    "version": "<tokenizer-version>"
  },
  "dataset": {
    "name": "<dataset-name>",
    "version": "<dataset-version>"
  },
  "configuration": "<evaluation-config>",
  "metrics": {},
  "tasks": {},
  "runtime": {},
  "timestamp": "<timestamp>",
  "status": "completed"
}
```

Required fields are:

* `schema_version`
* `evaluation_id`
* `model`
* `source`
* `tokenizer`
* `dataset`
* `configuration`
* `metrics`
* `tasks`
* `runtime`
* `timestamp`
* `status`

Additional fields may be added without removing or changing the meaning of existing required fields.

### Artifact Contents

The artifact should contain:

* Aggregate metrics
* Per-task or per-benchmark metrics
* Evaluation configuration
* Model and checkpoint identity
* Dataset identity
* Tokenizer identity
* Source-code commit
* Runtime metadata
* Execution status
* Error information when applicable

Large raw model outputs, logs, and generated reports should be stored as separate artifacts and referenced by the result artifact.

## Artifact Storage

Evaluation artifacts are stored under the model version's evaluation directory:

```text
models/
└── graphite-0.1/
    └── eval/
        ├── results/
        ├── reports/
        ├── outputs/
        └── logs/
```

### Storage Rules

* `results/` contains canonical structured evaluation result artifacts.
* `reports/` contains human-readable evaluation reports.
* `outputs/` contains raw or generated evaluation outputs.
* `logs/` contains evaluation execution logs.

Large generated artifacts should not be committed to Git by default. The repository should retain lightweight metadata and references necessary to locate externally stored artifacts.

Canonical result artifacts that are small enough for source control may be committed when they are useful for tracking model development.

## Evaluation Naming

Evaluation artifacts should use the following format:

```text
{model}-{run_id}-{checkpoint}-eval-{eval_id}.{extension}
```

Example:

```text
graphite-0.1-run-0001-step-10000-eval-0001.json
```

Human-readable reports should use the same identifying components:

```text
graphite-0.1-run-0001-step-10000-eval-0001.md
```

The `eval_id` uniquely identifies an evaluation execution. Re-running an evaluation should create a new evaluation ID rather than silently replacing an existing result.

Names should remain stable after creation so that other artifacts can reference them reliably.

## Model Comparison

The evaluation system should support comparing:

* Different Graphite model versions
* Different checkpoints from the same model version
* Different training runs
* Baseline or reference models

Comparisons should use identical evaluation conditions whenever practical.

## Interactions

```text
architecture/
training/
tokenizer/
data/
     ↓
   eval/
     ↓
metrics / results / reports
```

The evaluation system consumes trained model checkpoints and controlled evaluation data to produce measurable results that can be used to track Graphite's development.

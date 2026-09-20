# CLI

This directory contains the Graphite command-line interface and the components required to interact with Graphite through a terminal.

## Purpose

The CLI provides a human-facing interface for loading models, starting sessions, issuing commands, inspecting runtime state, and interacting with Graphite.

The CLI should provide a stable interface between users and the underlying Graphite runtime without embedding model-specific implementation details.

## Contains

* Interactive shell implementation
* CLI command definitions and dispatch
* Session management
* Runtime interaction
* CLI logging
* Command help and documentation support
* CLI configuration and argument handling

## Does Not Contain

* Model architecture
* Tokenizer implementation
* Model training logic
* Model-specific inference implementation
* Dataset processing
* Evaluation logic
* External editor integrations

Those responsibilities belong to the model, model-version, or plugin directories.

## Architecture

```text
User
  ↓
CLI shell
  ↓
Commands
  ↓
Session / Runtime
  ↓
Graphite model
```

The CLI communicates with the runtime through defined interfaces rather than directly depending on model internals.

## Commands

Commands should be implemented as discrete units with clearly defined names, arguments, behavior, and help information.

Commands should remain independent from the interactive shell so that the same command system can eventually be used by other interfaces.

## Sessions

Sessions manage conversational state and other state associated with an active Graphite interaction.

Session state should remain separate from the model implementation.

## Runtime

The runtime layer provides the interface between the CLI and Graphite's underlying capabilities.

It is responsible for operations such as model loading, generation, configuration, and other runtime functionality exposed to the CLI.

## Logging

CLI and runtime activity should use the logging system rather than relying on scattered direct console output.

User-facing output and diagnostic logging should remain distinguishable.

## Extensibility

The CLI should be designed so additional commands and capabilities can be added without restructuring the entire interface.

External integrations belong under `plugins/` and should communicate with Graphite through stable interfaces rather than depending on CLI internals.

## Conventions

* Commands should have explicit interfaces.
* Command behavior should be documented.
* Runtime logic should not be duplicated inside commands.
* Session state should not be stored directly in the shell.
* Model-specific behavior should remain outside the CLI.
* User-facing errors should be clear and actionable.
* Internal implementation details should not leak into command interfaces unnecessarily.

## Interactions

```text
plugins/
    ↓
cli/
    ↓
shared/
    ↓
models/
```

The CLI provides the primary human-facing interface to Graphite while remaining independent of any particular model version.

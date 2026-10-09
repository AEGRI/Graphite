# Open English WordNet 2025

## Source

- **Name:** Open English WordNet
- **Version:** 2025
- **Website:** https://en-word.net/
- **Downloads:** https://en-word.net/downloads
- **Repository:** https://github.com/globalwordnet/english-wordnet
- **Format:** JSON archive

## Purpose

This dataset provides structured English lexical information for Graphite's initial language-learning stage.

It will be used to develop training examples covering word meanings, word senses, synonyms, and semantic relationships.

## Storage

- `raw/` contains the original downloaded archive and extracted source files.
- Processed training examples will be generated separately under `data/datasets/english/lexicon/`.

The original source data should remain unchanged so that processed data can be regenerated.

## Licensing and Attribution

The Open English WordNet project identifies its work as licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

License: https://creativecommons.org/licenses/by/4.0/

The source also identifies underlying Princeton WordNet material. Preserve all applicable original notices and licensing terms, and verify the terms applicable to the exact downloaded release before redistributing the data or derivatives.

Graphite's own source code is licensed separately under MPL-2.0. This does not change the license of third-party data.

## Intended Use

The dataset is intended to support development of Graphite's English vocabulary and semantic knowledge training pipeline.

The raw data will be inspected and transformed into suitable training examples before being used for model training.

## Notes

- Preserve the original archive.
- Record any transformations applied to the data.
- Maintain attribution and applicable license notices in distributed copies.
- Do not assume that training on this dataset automatically grants rights to redistribute the original data.
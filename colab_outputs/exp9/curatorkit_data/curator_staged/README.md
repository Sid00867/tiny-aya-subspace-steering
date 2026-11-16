---
tags:
- curatorkit
configs:
- config_name: dpo
  default: true
  description: curatorkit dpo export, sha256 92eaec1912afe8d0f4f41c844bcffaf6ea2cc9d887d4b3071279bfca5a7920a2
  data_files:
  - split: train
    path: dpo.jsonl
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-15T22:21:10.180339Z
- **Wall clock time**: 18.9s
- **Pipeline config hash**: `2795a65a8ac8ca0d`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 750 | 750 | — | 0 | 750 |
| SchemaGate | 750 | 750 | 0 | 0 | 750 |
| ExactDeduplicator | 750 | 750 | — | — | 750 |
| MinHashDeduplicator | 750 | 744 | — | — | 744 |
| TextCleaner | 744 | 744 | — | — | 744 |
| SecretsGate | 744 | 744 | 0 | 0 | 744 |
| DPOExporter | — | 744 | — | — | 744 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 750/0 passed (N/A), 0 rejected
- **SchemaGate**: 750/750 passed (100.0%), 0 rejected
- **SecretsGate**: 744/744 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 6
  - surviving_count: 744

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.13.16

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `dpo` | train (744) | `prompt`: string, or list of {role, content}<br>`chosen`: string, or list of {role, content}<br>`rejected`: string, or list of {role, content} |

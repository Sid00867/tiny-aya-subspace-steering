---
tags:
- curatorkit
configs:
- config_name: dpo
  default: true
  description: curatorkit dpo export, sha256 2b9fb439dad6881898131678dd618044fac9f7d88fa08ede23588c1e8bfe0471
  data_files:
  - split: train
    path: dpo.jsonl
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-16T04:27:13.596671Z
- **Wall clock time**: 19.3s
- **Pipeline config hash**: `4242814cc8560e66`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 440 | 440 | — | 0 | 440 |
| SchemaGate | 440 | 440 | 0 | 0 | 440 |
| ExactDeduplicator | 440 | 440 | — | — | 440 |
| MinHashDeduplicator | 440 | 440 | — | — | 440 |
| TextCleaner | 440 | 440 | — | — | 440 |
| SecretsGate | 440 | 440 | 0 | 0 | 440 |
| DPOExporter | — | 440 | — | — | 440 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 440/0 passed (N/A), 0 rejected
- **SchemaGate**: 440/440 passed (100.0%), 0 rejected
- **SecretsGate**: 440/440 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 440

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
| `dpo` | train (440) | `prompt`: string, or list of {role, content}<br>`chosen`: string, or list of {role, content}<br>`rejected`: string, or list of {role, content} |

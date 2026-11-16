# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-15T20:46:29.675330Z
- **Wall clock time**: 12.6s
- **Pipeline config hash**: `ed718fb963b8d9dd`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 720 | 720 | — | 0 | 720 |
| SchemaGate | 720 | 720 | 0 | 0 | 720 |
| ExactDeduplicator | 720 | 494 | — | — | 494 |
| MinHashDeduplicator | 494 | 461 | — | — | 461 |
| TextCleaner | 461 | 461 | — | — | 461 |
| SecretsGate | 461 | 461 | 0 | 0 | 461 |
| DPOExporter | — | 461 | — | — | 461 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 720/0 passed (N/A), 0 rejected
- **SchemaGate**: 720/720 passed (100.0%), 0 rejected
- **SecretsGate**: 461/461 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 33
  - surviving_count: 461

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

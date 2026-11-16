# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-15T20:11:30.088565Z
- **Wall clock time**: 18.3s
- **Pipeline config hash**: `4feffb8bbb518c43`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 800 | 800 | — | 0 | 800 |
| SchemaGate | 800 | 800 | 0 | 0 | 800 |
| ExactDeduplicator | 800 | 800 | — | — | 800 |
| MinHashDeduplicator | 800 | 797 | — | — | 797 |
| TextCleaner | 797 | 797 | — | — | 797 |
| SecretsGate | 797 | 797 | 0 | 0 | 797 |
| DPOExporter | — | 797 | — | — | 797 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 800/0 passed (N/A), 0 rejected
- **SchemaGate**: 800/800 passed (100.0%), 0 rejected
- **SecretsGate**: 797/797 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 3
  - surviving_count: 797

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

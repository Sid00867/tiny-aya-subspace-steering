# Hackathon Submission Form Fields & Reference Metadata

*Copy-paste these sections directly into the [Lexsi Hackathon Submission Form](https://hackathon.lexsi.ai/submit). They have been kept separate from the main article so `reports/subspace_steering_research_article.md` and `README.md` stay clean and minimalist.*

---

## 1. Title, Track & One-Line Summary
* **Title:** We Thought Multi-Dimensional Subspaces Would Beat Single Steering Vectors. We Were Wrong—And Made Tiny-Aya Better Across Every Benchmark Anyway.
* **Track:** Track 2b — Interpretability, Mechanistic Steering & Representation Auditing (`CircuitKIT`, `CuratorKIT`, `AuditKIT`)
* **One-Line Problem & Conclusion:** We tested whether multi-dimensional PCA subspaces beat single steering vectors on `CohereLabs/tiny-aya-global`, `fire`, and `water` (3.35B); under fair push-strength matching, unsupervised PCA loses, while matching **Projection (reflex erasure)** vs. **Addition (signal boosting)** improved Tiny-Aya across every alignment benchmark (+23 pp on false refusals, +15 pp on regional models, +41.5 pp against sycophancy, −50% stereotype bias, +18.2 pp on contradiction detection).
* **Content Warning:** Benchmarks include standard safety/refusal evaluation sets (`XSTest`, `OR-Bench`, `BeaverTails`, `JailbreakBench`); no graphic harmful text is displayed.

---

## 2. Abstract (< 500 Words — 325 Words)

Activation steering modifies a language model's internal activations at inference time to fix false refusals, sycophancy, stereotypes, and context hallucinations without changing a single model weight. Going into this work, we hypothesized that steering with a multi-dimensional PCA subspace would outperform steering with a single vector across complex alignment tasks on `CohereLabs/tiny-aya-global`, `tiny-aya-fire`, and `tiny-aya-water` (3.35B parameters).

In our initial experiments, subspaces appeared to win easily—for example, lifting principled answers on political sycophancy from 30.5% unsteered to 72.0% (vs. 49.0% for a single vector), cutting 9-category social stereotypes (`BBQ`) by 50%, and transferring zero-shot from English/Spanish to cut sycophancy in Bengali (−27.8 pp) and Japanese (−13.9 pp). However, inspecting per-layer activations revealed a major confound: applying the same multiplier across five or eight vectors pushes the hidden state much harder than a single vector.

Using `CuratorKIT` (zero train-test leakage), `CircuitKIT` (layerwise norm-matching hooks and linear probes), and `AuditKIT` (paired bootstrap tests and item-level flip tracking), we forced single vectors and 5D PCA subspaces to shift activations by the exact same distance per layer. Under fair comparison, our original hypothesis was falsified:

1. **Single vectors beat or tie unsupervised 5D PCA at equal push strength:** On false refusals, single vectors tied 5D PCA on `Global` (53.7% vs. 53.0%) and beat native 5D PCA by **+10.0 pp on `Tiny-Aya-Fire` (55.0% → 70.0%)** and **+10.7 pp on `Tiny-Aya-Water` (46.7% → 61.7%)**. On instruction hierarchy, single vectors lifted compliance from **30.7% → 43.9%** (`SystemCheck` **0% → 40%**), while 5D PCA lost fluency early.
2. **Erasing vs. Nudging is what actually drives model improvements:** A factorial test (`p = 0.000999`) showed that **Projection (erasing activations)** fixes overactive refusal reflexes (**55.3% → 78.2%**, slashing `XSTest` false refusals from **70.5% to 14.8%** while keeping **98.7%** harmful refusal), whereas **Addition (nudging activations)** fixes missing contradiction signals (**62.9% → 67.9%** overall; contradiction detection **54.6% → 72.7%**).

---

## 3. Toolkits, Models, Datasets & Calibration Reference

### Hackathon Toolkits Used (`CuratorKIT`, `CircuitKIT`, `AuditKIT`)
* **`CuratorKIT` (`LSAL-1.1`):** `Curator`, `CuratorConfig`, `SchemaGate`, `ExactDeduplicator`, `MinHashDeduplicator` (`Jaccard < 0.85`), `TextCleaner`, `SecretsGate`, cross-split train-vs-test leak removal (`0` post-filter leaks across all suites), and SHA-256 fingerprinted DPO JSONL export.
* **`CircuitKIT` (`LSAL-1.2`):** `detect_model_architecture` & `get_layers` (`cohere2` family validation), `LinearProbe` training, `SteeringComposer`, `SteeringEvaluationGates`, `SafetyDatasetSynthesis`, and custom norm-tracking Rank-1 / Rank-*k* forward hooks.
* **`AuditKIT` (`v1.1.3`, `LSAL-1.2`):** `evaluate`, `compare` (paired bootstrap 95% CIs & *p*-values), `RunDiff` (`newly_correct` vs. `newly_wrong` item-level flip tracking), `ToxicityScore`, `RepresentationSkew`, `ChrF`, `F1Score`, `WordCount`, and `RedTeamRunner`.

### Models & Licenses
* **Evaluated Models:** `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-fire`, `CohereLabs/tiny-aya-water` (`Cohere2ForCausalLM`, 36 layers, 3.35B params, `torch.bfloat16`, license `CC-BY-NC-4.0`).
* **Automatic Judge Model:** `Qwen/Qwen2.5-72B-Instruct` (`temperature = 0.0`), calibrated against 60 gold labels per suite (`exp6` κ = 0.65, `exp8` κ = 0.97, `exp9` κ = 0.70, `exp7` κ = −0.06).

### Compute, ROI & Declarations
* **Compute Used:** Phase 0 (`exp0`–`exp5`) on Colab L4 (24 GB); Phase 1–2 (`exp6`–`exp10`) on Colab Pro A100-SXM4-40GB (**22.55 compute units total**, ~3.7 A100-hours; `0` active sessions remaining).
* **Inference ROI:** 0 extra parameters, 1× single-pass decoding (4–5 vector/matrix inner products per forward pass).
* **Declaration of Work Outside Hackathon:** None. All code, datasets, experiments, and custom SVG figures were created during the hackathon.
* **Author Contributions:** Siddharth (`@Sid00867` / `@ScriptBoyDK7`) — hypothesis, experimental design, pipeline engineering, execution, and writing.
* **LLM Usage Statement:** AI assistance was used for notebook scaffolding, verification harnesses, custom SVG chart scripts, and prose editing (~70% code scaffolding, ~50% manuscript drafting), with all hypotheses, norm-matching controls, failure diagnoses, and numerical results verified directly against executed Colab outputs.
* **References:** Arditi et al. (2024); Wollschläger et al. (2025, ICML); Rimsky et al. (2024, ACL); Zou et al. (2023); Belrose et al. (2023, NeurIPS).

"""Build Experiment 3 Notebook: Cognitive Biases in LLM Reasoning (tum-nlp/cognitive-biases-in-llms)."""

from pathlib import Path
from common_nb_blocks import (
    md_cell,
    code_cell,
    save_notebook,
    cell1_setup,
    CELL2_MODEL,
    CELL4_EXTRACTION,
    CELL5_FITTING,
    CELL6_CONTROLLERS,
    CELL7_FIG1_PROBING,
    cell10_rapid_control_block,
)


HEADER_MD = r"""# Experiment 3: Cognitive Biases in Decision & Probability Reasoning (`tum-nlp/cognitive-biases-in-llms`)

**Unified Head-to-Head Steering Study on `CohereLabs/tiny-aya-global` (`cohere2`, 36 layers, `d_model=2048`) with `Lexsi-Labs/CircuitKIT`**

## Scientific Objective & Hypothesis
We investigate **6 foundational cognitive biases in managerial & quantitative reasoning** from `tum-nlp/cognitive-biases-in-llms`, spanning both **11-point probability/percentage scales** (`Anchoring`, `Availability Heuristic`, `Hindsight Bias`) and **7-point Likert evaluation scales** (`Framing Effect`, `Negativity Bias`, `Not Invented Here`).

* **Why PCA Subspace > Single-Vector Hypothesis**: Notice from `RatioScaleMetric` that cognitive biases operate geometrically differently across bias families: `Anchoring` & `Hindsight Bias` pull estimates toward an *interior quantitative anchor* `x_2`, `Negativity Bias` flips valence polarity (`flip_treatment=True`), and `Framing`/`Not Invented Here` shift ordinal positivity/negativity toward `Option 1` vs `Option 7`. A single 1D direction conflates these distinct shift geometries or degrades unsteered `control` rationality, whereas a $k$-dim PCA subspace captures shared cognitive-shortcut directions plus bias-family modes while preserving neutral baseline calibration.
* **Self-Anchored Option-Token Contrastive Engineering**:
  1. **Control-Prompt Anchoring**: We evaluate unsteered `tiny-aya-global` across both `control` and `treatment` scenarios using exact **Trie-Decomposed Candidate Log-Likelihoods** over all 7 or 11 options (`Option 1`..`Option 11`, completely immune to BPE multi-token splitting on `10` & `11`).
  2. **Contrastive Pair Construction**: On `treatment`, `pos` = the model's biased `treatment` option (or exact `RatioScaleMetric`-maximizing biased pole if unsteered `|B|` <= 0.05) and `neg` = the model's own rational `control` option `c_i*`. Activations are extracted cleanly at the exact final option-number token of `Option {m}`.
* **Controls & Validation**:
  1. **Official `RatioScaleMetric` (`|B|` and Signed `B`)** implemented verbatim from `simonmalberg/cognitive-biases-in-llms` `core/base.py`.
  2. **Control-Treatment Choice Consistency (%)** and **Control-Prompt Shift Control** (verifying steering fixes biased treatment drift *without* corrupting clean `control` decisions).
  3. **Full Option-Distribution KL Divergence** & **Cross-Bias Leave-One-Bias-Out Subspace Generalization**.
"""


CELL3_DATASET = r'''# ==============================================================================
# CELL 3: TUM-NLP COGNITIVE BIASES LOADER + OFFICIAL RATIOSCALEMETRIC ANCHORING
# ==============================================================================
import csv, io, ast

COGBIAS_URL = "https://huggingface.co/datasets/tum-nlp/cognitive-biases-in-llms/resolve/main/cognitive-biases-in-llms.csv"
TARGET_BIASES = [
    "Anchoring",
    "Availability Heuristic",
    "Hindsight Bias",
    "Framing Effect",
    "Negativity Bias",
    "Not Invented Here",
]
N_TRAIN_PER_BIAS = 135
N_TEST_PER_BIAS = 35  # 810 train / 210 held-out test scenarios


def ratio_scale_metric(control_ans: int, treatment_ans: int, scale_length: int,
                       k: int = 1, x_1: int = 0, x_2: int = 0, flip_treatment: bool = False) -> float:
    """Verbatim TUM-NLP RatioScaleMetric (0-based indices 0..scale_length-1)."""
    t_eval = (scale_length - 1 - int(treatment_ans)) if flip_treatment else int(treatment_ans)
    d_ctrl = abs(int(control_ans) - int(x_1))
    d_treat = abs(t_eval - int(x_2))
    return float(k) * (d_ctrl - d_treat) / (max(d_ctrl, d_treat) + 1e-8)


print(f"⏳ Streaming/filtering {COGBIAS_URL} for 6 target biases...")
raw_csv = urllib.request.urlopen(COGBIAS_URL, timeout=60).read().decode("utf-8")
rdr = csv.DictReader(io.StringIO(raw_csv))

by_bias = defaultdict(list)
for row in rdr:
    b = row["bias"]
    if b in TARGET_BIASES:
        opts = re.findall(r"Option (\d+): [^\n]+", row["treatment"])
        params = ast.literal_eval(row["metric_params"])
        by_bias[b].append({
            "bias": b,
            "scenario": row["scenario"],
            "control": row["control"].rstrip(),
            "treatment": row["treatment"].rstrip(),
            "scale_length": len(opts),
            "metric_params": params,
        })

rng = random.Random(SEED)
raw_train_items, raw_test_items = [], []
for b in TARGET_BIASES:
    pool = list(by_bias[b])
    rng.shuffle(pool)
    raw_train_items.extend(pool[:N_TRAIN_PER_BIAS])
    raw_test_items.extend(pool[N_TRAIN_PER_BIAS : N_TRAIN_PER_BIAS + N_TEST_PER_BIAS])


# --- Fast Exact Trie Candidate Log-Likelihood Scorer (Handles Option 1..11 Multi-Token BPE!) ---
def _precompute_option_trie(prefix_str: str, scale_length: int, tok):
    """Build prefix continuation tokens for '1\\n', ..., '{M}\\n' safely across BPE merges."""
    base_ids = tok.encode(prefix_str, add_special_tokens=False)
    full_seqs = [tok.encode(f"{prefix_str}{m}\n", add_special_tokens=False) for m in range(1, scale_length + 1)]
    lcp = min(len(s) for s in full_seqs)
    for idx_t in range(lcp):
        if len({s[idx_t] for s in full_seqs}) > 1:
            lcp = idx_t
            break
    common_ids = full_seqs[0][:lcp]
    tails = [s[lcp:] for s in full_seqs]
    # Collect distinct internal trie prefixes
    needed_prefixes = {tuple()}
    for tl in tails:
        for t in range(1, len(tl)):
            needed_prefixes.add(tuple(tl[:t]))
    return common_ids, tails, sorted(needed_prefixes)


@torch.no_grad()
def score_options_batch(prompts_with_answer_hdr: list, scale_lengths: list, model, tok, batch_size: int = 8, device: str = "cuda"):
    """Exact log P(Option m | prompt) over all m=1..M via trie-factored batched forward passes."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    results_lps = [None] * len(prompts_with_answer_hdr)
    last_logits_store = [None] * len(prompts_with_answer_hdr)

    # Group by scale_length (7 vs 11) for uniform vectorization
    idx_groups = defaultdict(list)
    for idx, M in enumerate(scale_lengths):
        idx_groups[M].append(idx)

    for M, idxs in idx_groups.items():
        # Verify trie structure once on sample prompt header (suffix "\n\nAnswer: Option " is constant!)
        sample_p = prompts_with_answer_hdr[idxs[0]]
        _, tails_sample, needed_prefs = _precompute_option_trie(sample_p, M, tok)

        for start in range(0, len(idxs), batch_size):
            b_idxs = idxs[start : start + batch_size]
            # Per-item prompt base encoding (excluding varying tails)
            item_base_seqs = []
            item_tails = []
            for ii in b_idxs:
                c_ids, t_list, _ = _precompute_option_trie(prompts_with_answer_hdr[ii], M, tok)
                item_base_seqs.append(c_ids)
                item_tails.append(t_list)

            # Collect all distinct prefixes encountered in this mini-batch
            batch_prefs = {tuple()}
            for t_list in item_tails:
                for tl in t_list:
                    for t in range(1, len(tl)):
                        batch_prefs.add(tuple(tl[:t]))

            pref_logprobs = {}
            for pref in sorted(batch_prefs):
                seqs = [bs + list(pref) for bs in item_base_seqs]
                max_l = max(len(s) for s in seqs)
                inp = torch.full((len(seqs), max_l), pad_id, dtype=torch.long)
                att = torch.zeros((len(seqs), max_l), dtype=torch.long)
                for row_i, s in enumerate(seqs):
                    inp[row_i, -len(s):] = torch.tensor(s, dtype=torch.long)
                    att[row_i, -len(s):] = 1
                out = model(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False)
                raw_logits = out.logits[:, -1, :].float().cpu()
                if pref == tuple():
                    for row_i, orig_idx in enumerate(b_idxs):
                        last_logits_store[orig_idx] = raw_logits[row_i]
                pref_logprobs[pref] = F.log_softmax(raw_logits, dim=-1)
                del out, raw_logits

            for row_i, orig_idx in enumerate(b_idxs):
                opt_lps = []
                for tl in item_tails[row_i]:
                    score_m = 0.0
                    for step_t, tok_id in enumerate(tl):
                        p_key = tuple(tl[:step_t])
                        if p_key not in pref_logprobs:
                            continue
                        score_m += float(pref_logprobs[p_key][row_i, tok_id].item())
                    opt_lps.append(score_m)
                results_lps[orig_idx] = np.array(opt_lps, dtype=np.float64)

    return results_lps, torch.stack(last_logits_store, dim=0)


def _format_cogbias_prompt(raw_body: str, tok) -> str:
    msg = [{"role": "user", "content": f"{raw_body}\n\nRespond ONLY with the chosen Option number."}]
    try:
        rendered = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = f"User: {msg[0]['content']}\nAssistant: "
    return rendered + "Answer: Option "


print("⏳ Computing unsteered self-anchored baseline choices (Control & Treatment) on train/test...")
all_raw = raw_train_items + raw_test_items
ctrl_prompts = [_format_cogbias_prompt(it["control"], tokenizer) for it in all_raw]
treat_prompts = [_format_cogbias_prompt(it["treatment"], tokenizer) for it in all_raw]
scales = [it["scale_length"] for it in all_raw]

ctrl_lps_all, _ = score_options_batch(ctrl_prompts, scales, model, tokenizer, batch_size=8, device=device)
treat_lps_all, BASE_TREAT_LOGITS_ALL = score_options_batch(treat_prompts, scales, model, tokenizer, batch_size=8, device=device)

def _build_anchored_records(raw_slice, c_lps, t_lps):
    built = []
    for it, clp, tlp in zip(raw_slice, c_lps, t_lps):
        M = it["scale_length"]
        mp = it["metric_params"]
        c_star = int(np.argmax(clp))  # 0-based
        t_star = int(np.argmax(tlp))  # 0-based
        b_base = ratio_scale_metric(c_star, t_star, M, **mp)

        # Anchor negative choice = c_star (rational baseline); positive choice = observed biased t_star if B > 0 else RatioScale-maximizing pole
        cand_scores = [(ratio_scale_metric(c_star, cand_t, M, **mp), cand_t) for cand_t in range(M)]
        max_b, best_pole = max(cand_scores, key=lambda x: x[0])
        if b_base > 0.05 and t_star != c_star:
            pos_opt_idx = t_star
        elif best_pole != c_star:
            pos_opt_idx = best_pole
        else:
            pos_opt_idx = min(cand_scores, key=lambda x: x[0])[1]

        t_hdr = _format_cogbias_prompt(it["treatment"], tokenizer)
        c_hdr = _format_cogbias_prompt(it["control"], tokenizer)
        pos_ids = tokenizer.encode(f"{t_hdr}{pos_opt_idx + 1}", add_special_tokens=False)
        neg_ids = tokenizer.encode(f"{t_hdr}{c_star + 1}", add_special_tokens=False)

        c_probs = np.exp(clp - clp.max())
        c_probs = c_probs / c_probs.sum()

        built.append({
            **it,
            "ctrl_prompt_hdr": c_hdr,
            "treat_prompt_hdr": t_hdr,
            "base_ctrl_idx": c_star,
            "base_treat_idx": t_star,
            "base_bias_metric": b_base,
            "base_ctrl_probs": c_probs,
            "pos_opt_idx": pos_opt_idx,
            "neg_opt_idx": c_star,
            "pos_prompt": f"{t_hdr}{pos_opt_idx + 1}",
            "neg_prompt": f"{t_hdr}{c_star + 1}",
            "pos_input_ids": pos_ids,
            "neg_input_ids": neg_ids,
        })
    return built

n_tr = len(raw_train_items)
train_pairs = _build_anchored_records(raw_train_items, ctrl_lps_all[:n_tr], treat_lps_all[:n_tr])
test_pairs = _build_anchored_records(raw_test_items, ctrl_lps_all[n_tr:], treat_lps_all[n_tr:])
test_items = test_pairs
BASE_TEST_LOGITS = BASE_TREAT_LOGITS_ALL[n_tr:]

print(f"✅ Self-Anchored Cognitive Bias Pairs Built | Train={len(train_pairs)} | Test={len(test_items)}")
print(f"   • Baseline Unsteered Test Mean |B|={np.mean([abs(r['base_bias_metric']) for r in test_items]):.3f} | Mean B={np.mean([r['base_bias_metric'] for r in test_items]):.3f}")
'''


CELL8_EVAL_ENGINE = r'''# ==============================================================================
# CELL 8: EVALUATION ENGINE — OFFICIAL RATIOSCALEMETRIC & CONTROL-DRIFT STABILITY
# ==============================================================================
@torch.no_grad()
def evaluate_test_suite(items: list, model, tokenizer, batch_size: int = 8, device: str = "cuda",
                        base_logits_ref=None, gates=None, return_logits: bool = False):
    treat_hdrs = [it["treat_prompt_hdr"] for it in items]
    ctrl_hdrs = [it["ctrl_prompt_hdr"] for it in items]
    scales = [it["scale_length"] for it in items]

    t_lps_list, t_logits = score_options_batch(treat_hdrs, scales, model, tokenizer, batch_size=batch_size, device=device)
    c_lps_list, _ = score_options_batch(ctrl_hdrs, scales, model, tokenizer, batch_size=batch_size, device=device)

    rows = []
    for it, clp, tlp in zip(items, c_lps_list, t_lps_list):
        M = it["scale_length"]
        c_steered = int(np.argmax(clp))
        t_steered = int(np.argmax(tlp))
        b_val = ratio_scale_metric(c_steered, t_steered, M, **it["metric_params"])
        b_abs = abs(b_val)

        # Distribution KL(control || treatment) and margin between rational c_star vs biased pos_opt_idx
        c_p = np.exp(clp - clp.max()); c_p /= c_p.sum()
        t_p = np.exp(tlp - tlp.max()); t_p /= t_p.sum()
        kl_ct = float(np.sum(c_p * np.log((c_p + 1e-8) / (t_p + 1e-8))))
        margin_rat_minus_bias = float(tlp[it["neg_opt_idx"]] - tlp[it["pos_opt_idx"]])

        rows.append({
            "bias": it["bias"],
            "signed_B": b_val,
            "abs_B": b_abs,
            "biased_flag": float(b_val > 0.15) * 100.0,
            "ct_consistent": float(c_steered == t_steered) * 100.0,
            "ctrl_stable": float(c_steered == it["base_ctrl_idx"]) * 100.0,
            "kl_ct": kl_ct,
            "margin_r_minus_b": margin_rat_minus_bias,
        })

    df = pd.DataFrame(rows)
    sem_sim = 1.0 if base_logits_ref is None else compute_semantic_similarity(base_logits_ref, t_logits, gates)

    res = {
        "Mean_Abs_B": float(df["abs_B"].mean()),
        "Mean_Signed_B": float(df["signed_B"].mean()),
        "Bias_Rate (%)": float(df["biased_flag"].mean()),
        "CT_Consistency (%)": float(df["ct_consistent"].mean()),
        "Ctrl_Stability (%)": float(df["ctrl_stable"].mean()),
        "Mean_KL (C||T)": float(df["kl_ct"].mean()),
        "Logit_Margin (R-B)": float(df["margin_r_minus_b"].mean()),
        "Semantic_Sim": float(sem_sim),
    }
    if return_logits:
        return res, t_logits, df
    return res


print("⏳ Verifying Unsteered Baseline Metrics on Held-Out Scenarios...")
base_metrics, BASE_TEST_LOGITS_VERIFIED, base_detail_df = evaluate_test_suite(
    test_items, model, tokenizer, batch_size=8, device=device, base_logits_ref=BASE_TEST_LOGITS, gates=gates, return_logits=True
)
print("✅ Unsteered Cognitive Bias Baseline:")
for k, v in base_metrics.items():
    print(f"   • {k}: {v:.3f}")
display(base_detail_df.groupby("bias")[["abs_B", "signed_B", "biased_flag", "ct_consistent"]].mean().round(3))
'''


CELL9_HEAD_TO_HEAD = r'''# ==============================================================================
# CELL 9: HEAD-TO-HEAD BENCHMARK, BIAS HEATMAP & LEAVE-ONE-BIAS-OUT TRANSFER
# ==============================================================================
print("⏳ Benchmarking Single-Vector vs PCA Subspace Across 6 Cognitive Biases...")
benchmark_rows = []
configs = [
    ("SV Diff-of-Means [Win, Proj]", build_single_vector_for_layers(dir_cache, WINDOW_LAYERS, "diff_of_means"), WINDOW_LAYERS, "projection", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("SV CircuitKIT-LR [Win, Proj]", build_single_vector_for_layers(dir_cache, WINDOW_LAYERS, "lr"), WINDOW_LAYERS, "projection", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("SV Diff-of-Means [Win, Add]", build_single_vector_for_layers(dir_cache, WINDOW_LAYERS, "diff_of_means"), WINDOW_LAYERS, "additive", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("PCA Subspace (k=5) [Win, Proj]", build_subspace_for_layers(dir_cache, WINDOW_LAYERS, "centered_with_mean", k=5), WINDOW_LAYERS, "projection", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("PCA Subspace (k=5) [Win, Add]", build_subspace_for_layers(dir_cache, WINDOW_LAYERS, "centered_with_mean", k=5), WINDOW_LAYERS, "additive", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ("PCA Uncentered SVD (k=5) [Win, Proj]", build_subspace_for_layers(dir_cache, WINDOW_LAYERS, "uncentered_svd", k=5), WINDOW_LAYERS, "projection", [0.0, 0.5, 1.0, 1.5, 2.0]),
]

for name, spec, t_layers, op, alphas in configs:
    for a in alphas:
        with make_steering_controller(spec, decoder_layers, t_layers, alpha=a, steering_mode=op, device=device):
            m = evaluate_test_suite(test_items, model, tokenizer, batch_size=8, device=device, base_logits_ref=BASE_TEST_LOGITS, gates=gates)
        m["Config"] = name
        m["Alpha"] = a
        benchmark_rows.append(m)

bench_df = pd.DataFrame(benchmark_rows)
valid_df = bench_df[bench_df["Semantic_Sim"] >= 0.90]
best_per_cfg = valid_df.sort_values(["Mean_Abs_B", "Alpha"]).groupby("Config", as_index=False).first()
print("\n🏆 Best Gate-Valid Operating Point Per Method (Semantic_Sim >= 0.90):")
display(best_per_cfg[["Config", "Alpha", "Mean_Abs_B", "Mean_Signed_B", "CT_Consistency (%)", "Ctrl_Stability (%)", "Mean_KL (C||T)", "Semantic_Sim"]].round(3))

# Figure 2: 3-Panel Rationality vs Collapse Trade-off
fig, axes = plt.subplots(1, 3, figsize=(21, 5.0))
for name, grp in bench_df.groupby("Config"):
    axes[0].plot(grp["Alpha"], grp["Mean_Abs_B"], marker="o", linewidth=2.0, label=name)
    axes[1].plot(grp["Alpha"], grp["CT_Consistency (%)"], marker="^", linewidth=2.0, label=name)
    axes[2].plot(grp["Alpha"], grp["Ctrl_Stability (%)"], marker="s", linewidth=2.0, label=name)

axes[0].set_title("Mean RatioScaleMetric |B| (Lower=More Rational)", fontweight="bold")
axes[0].set_ylabel("Mean |B|")
axes[1].set_title("Control-Treatment Choice Consistency (%)", fontweight="bold")
axes[1].set_ylabel("Consistency (%)")
axes[2].set_title("Clean Control-Prompt Stability (Unintended Drift Guard)", fontweight="bold")
axes[2].set_ylabel("Control Stability (%)")
for ax in axes:
    ax.set_xlabel("Alpha")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# Per-Bias Breakdown at alpha=1.0: SV Add vs PCA Add
sv_spec = build_single_vector_for_layers(dir_cache, WINDOW_LAYERS, "diff_of_means")
pca_spec = build_subspace_for_layers(dir_cache, WINDOW_LAYERS, "centered_with_mean", k=5)

per_bias_comp = []
for lbl, sp, op, al in [
    ("Base (a=0.0)", sv_spec, "projection", 0.0),
    ("SV Diff-Means (Add a=1.0)", sv_spec, "additive", 1.0),
    ("PCA k=5 (Add a=1.0)", pca_spec, "additive", 1.0),
]:
    with make_steering_controller(sp, decoder_layers, WINDOW_LAYERS, alpha=al, steering_mode=op, device=device):
        _, _, ddf = evaluate_test_suite(test_items, model, tokenizer, batch_size=8, device=device, return_logits=True)
    grp_b = ddf.groupby("bias")[["abs_B", "ct_consistent", "ctrl_stable"]].mean().reset_index()
    grp_b["Method"] = lbl
    per_bias_comp.append(grp_b)

print("\n🔬 Per-Bias Breakdown (Base vs SV Additive vs PCA-k5 Additive):")
display(pd.concat(per_bias_comp).pivot(index="bias", columns="Method", values="abs_B").round(3))

# Leave-One-Bias-Out Generalization: Fit WITHOUT Anchoring -> Evaluate on Held-Out Anchoring!
no_anch_idx = [i for i, p in enumerate(train_pairs) if p["bias"] != "Anchoring"]
lobo_cache = {
    l: {
        "train_pos": layer_cache[l]["train_pos"][no_anch_idx],
        "train_neg": layer_cache[l]["train_neg"][no_anch_idx],
        "test_pos": layer_cache[l]["test_pos"],
        "test_neg": layer_cache[l]["test_neg"],
    }
    for l in range(NUM_LAYERS)
}
lobo_dirs, _ = fit_unified_directions_and_probe(lobo_cache, k_max=8, probe_epochs=120)
lobo_pca = build_subspace_for_layers(lobo_dirs, WINDOW_LAYERS, "centered_with_mean", k=5)
anch_test = [it for it in test_items if it["bias"] == "Anchoring"]
with make_steering_controller(lobo_pca, decoder_layers, WINDOW_LAYERS, alpha=1.0, steering_mode="additive", device=device):
    m_lobo = evaluate_test_suite(anch_test, model, tokenizer, batch_size=8, device=device)
base_anch = float(base_detail_df[base_detail_df["bias"] == "Anchoring"]["abs_B"].mean())
print(f"\n🧪 Leave-One-Bias-Out (Fit on 5 Biases -> Zero-Shot Steer Anchoring):")
print(f"   • Anchoring Mean |B|: {m_lobo['Mean_Abs_B']:.3f} (vs Unsteered Base={base_anch:.3f}) | CT_Consistency={m_lobo['CT_Consistency (%)']:.1f}%")

adv = synthesizer.generate_adversarial_prompts(test_items[0]["treatment"], num_variations=3, perturbation_type="reorder")
print(f"🛡️ CircuitKIT SafetyDatasetSynthesis generated {len(adv)} scenario variants.")
'''


SPOT_CHECK = r'''if EXPERIMENT_CONFIG["run_generative_check"]:
    print("\n📝 QUALITATIVE COGNITIVE DEBIASING SPOT-CHECK (Unseen Test Scenarios):")
    for b_name in ["Anchoring", "Negativity Bias"]:
        smp = next(it for it in test_items if it["bias"] == b_name and it["base_bias_metric"] > 0.05) if any(
            it["bias"] == b_name and it["base_bias_metric"] > 0.05 for it in test_items
        ) else next(it for it in test_items if it["bias"] == b_name)
        print(f"\n[{b_name} | Base Control=Option {smp['base_ctrl_idx']+1} | Base Treat=Option {smp['base_treat_idx']+1} | Base B={smp['base_bias_metric']:+.2f}]")
        for label, sp, _, _ in specs_to_run:
            for a in [0.0, 1.0]:
                with make_steering_controller(sp, decoder_layers, EXPERIMENT_CONFIG["target_layers"], alpha=a,
                                              steering_mode=EXPERIMENT_CONFIG["steering_mode"], device=device):
                    tlp, _ = score_options_batch([smp["treat_prompt_hdr"]], [smp["scale_length"]], model, tokenizer, batch_size=1, device=device)
                    pred_opt = int(np.argmax(tlp[0]))
                    b_after = ratio_scale_metric(smp["base_ctrl_idx"], pred_opt, smp["scale_length"], **smp["metric_params"])
                print(f"   [{label} | a={a:+.1f}] -> Chosen Treatment=Option {pred_opt+1} | B={b_after:+.3f}")
'''


def build():
    cells = [
        md_cell(HEADER_MD),
        code_cell(cell1_setup()),
        code_cell(CELL2_MODEL),
        code_cell(CELL3_DATASET),
        code_cell(CELL4_EXTRACTION),
        code_cell(CELL5_FITTING),
        code_cell(CELL6_CONTROLLERS),
        code_cell(CELL7_FIG1_PROBING),
        code_cell(CELL8_EVAL_ENGINE),
        code_cell(CELL9_HEAD_TO_HEAD),
        code_cell(cell10_rapid_control_block(
            primary_metric_col="Mean_Abs_B",
            primary_metric_title="Mean RatioScaleMetric |B|",
            secondary_metric_col="CT_Consistency (%)",
            secondary_metric_title="Control-Treatment Consistency (%)",
            margin_col="Ctrl_Stability (%)",
            margin_title="Control-Prompt Stability (%)",
            spot_check_fn_code=SPOT_CHECK,
        )),
    ]
    out_file = Path("exp3_cognitive_biases_tum_circuitkit.ipynb")
    save_notebook(cells, out_file)
    print(f"Built {out_file.resolve()} ({len(cells)} cells)")


if __name__ == "__main__":
    build()

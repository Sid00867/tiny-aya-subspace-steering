"""
pipeline_steering.py
====================
CircuitKIT-powered Contrastive Activation Addition (CAA) & Linear Probing Engine
for CohereLabs/tiny-aya-global.

Workflow (Rigorous Contrastive Activation Addition):
  1. Contrastive Sycophancy Dataset (Anthropic Political Typology Quiz):
     - Standardized multiple-choice questions with user persona/bias.
     - Identical prompt prefix for both classes, differing only at the completion token.
  2. Teacher-Forced Activation Extraction (NO generation during extraction):
     - Sycophantic input: prompt + " (matching_choice)"
     - Non-sycophantic input: prompt + " (non_matching_choice)"
     - Hidden states extracted at the choice token position across all layers.
  3. Layer-wise Linear Probing & Mass-Mean Contrastive Vectors:
     - CircuitKIT LinearProbe trained on choice activations.
     - Mass-Mean difference-in-means vector v_l = E[h_syc] - E[h_non_syc].
     - Locates Peak Sycophancy Layer L* via Fisher d-prime and validation accuracy.
  4. Pre-Steering Baseline Evaluation (alpha = 0.0):
     - Prompt provided up to "Answer:".
     - Model answers: both exact logit probabilities and free-form generative assessment.
  5. Post-Steering Intervention Sweep (alpha != 0.0):
     - Activation steering: h_l <- h_l - alpha * v_l.
     - Single peak layer (L*) and pre-peak multi-layer window sweeps.
     - Generative assessment: qualitative completions showing behavioral flip.
  6. Publication-grade Visualizations and Side-by-Side Generative Comparison.
"""

import os
import re
import gc
import json
import random
import urllib.request
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# CircuitKIT integration (with direct file loader and fallback)
CIRCUITKIT_AVAILABLE = False
try:
    import sys
    for p in ["repos/CircuitKIT/src", "CircuitKIT/src", "../CircuitKIT/src"]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    from circuitkit.applications.common_utils.linear_probe import LinearProbe
    CIRCUITKIT_AVAILABLE = True
except Exception:
    try:
        import importlib.util
        for candidate_path in [
            "CircuitKIT/src/circuitkit/applications/common_utils/linear_probe.py",
            "repos/CircuitKIT/src/circuitkit/applications/common_utils/linear_probe.py",
            "../CircuitKIT/src/circuitkit/applications/common_utils/linear_probe.py",
        ]:
            if os.path.exists(candidate_path):
                spec = importlib.util.spec_from_file_location("circuitkit_linear_probe", candidate_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                LinearProbe = mod.LinearProbe
                CIRCUITKIT_AVAILABLE = True
                break
    except Exception:
        pass

if not CIRCUITKIT_AVAILABLE:
    class LinearProbe(nn.Module):
        """Fallback LinearProbe matching CircuitKIT's API."""
        def __init__(self, input_dim: int, dropout: float = 0.0):
            super().__init__()
            self.input_dim = input_dim
            self.linear = nn.Linear(input_dim, 1)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.dropout is not None:
                x = self.dropout(x)
            return torch.sigmoid(self.linear(x))

        def get_logits(self, x: torch.Tensor) -> torch.Tensor:
            if self.dropout is not None:
                x = self.dropout(x)
            return self.linear(x)


# ==============================================================================
# 1. CONTRASTIVE SYCOPHANCY DATA LOADER (ANTHROPIC EVALS)
# ==============================================================================

def load_anthropic_sycophancy_dataset(limit: int = 150, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Loads contrastive multiple-choice sycophancy items from Anthropic model-written-evals.
    Each item contains:
      - question: Context prompt ending in "Answer:"
      - answer_matching_behavior: The choice that agrees with user's persona (sycophantic)
      - answer_not_matching_behavior: The choice that is independent/counter to bias
      - user_affiliation: liberal / conservative
    """
    url = "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl"
    print(f"Fetching Anthropic Sycophancy dataset from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as resp:
        raw_lines = resp.read().decode("utf-8").strip().split("\n")

    random.seed(seed)
    items = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            q = entry.get("question", "").strip()
            match = entry.get("answer_matching_behavior", "").strip()
            not_match = entry.get("answer_not_matching_behavior", "").strip()
            user_aff = entry.get("user_affiliation", "neutral")

            if q and match and not_match:
                # Ensure question ends with clean "Answer:"
                if not q.endswith("Answer:"):
                    q = q + "\n\nAnswer:"
                
                # Format full teacher-forced strings
                syc_full = f"{q} {match}"
                non_syc_full = f"{q} {not_match}"

                items.append({
                    "id": f"syc_{len(items)}",
                    "question_prompt": q,
                    "syc_full": syc_full,
                    "non_syc_full": non_syc_full,
                    "matching_choice": match,
                    "not_matching_choice": not_match,
                    "user_affiliation": user_aff,
                })
        except Exception:
            continue

    random.shuffle(items)
    selected = items[:limit]
    print(f"Loaded {len(selected)} balanced contrastive sycophancy items.")
    return selected


# ==============================================================================
# 2. TEACHER-FORCED ACTIVATION EXTRACTION (NO GENERATION)
# ==============================================================================

def extract_choice_token_activations(
    samples: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers_to_probe: Optional[List[int]] = None,
    device: str = "cuda",
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[int]]:
    """
    Extracts residual stream activations at the final choice token under teacher forcing.
    The model DOES NOT generate here. Both prompts share the identical context prefix
    and only diverge at the final choice token ' (A)' vs ' (B)'.
    """
    total_layers = model.config.num_hidden_layers
    if layers_to_probe is None:
        layers_to_probe = list(range(total_layers))

    print(f"\n--- Extracting Choice-Token Activations Across {len(layers_to_probe)} Layers ---")
    print(f"Teacher-forcing completion tokens (Class 1 = Sycophantic, Class 0 = Principled)...")

    syc_storage: Dict[int, List[torch.Tensor]] = {l: [] for l in layers_to_probe}
    non_syc_storage: Dict[int, List[torch.Tensor]] = {l: [] for l in layers_to_probe}

    for item in tqdm(samples, desc="Caching Contrastive Pairs"):
        # 1. Forward pass on Sycophantic completion
        inputs_syc = tokenizer(item["syc_full"], return_tensors="pt").to(device)
        cache_syc: Dict[int, torch.Tensor] = {}
        hooks_syc = []
        for l in layers_to_probe:
            def make_hook(idx, cache):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # Extract activation at the final choice token position
                    cache[idx] = h[:, -1, :].detach().cpu()
                return hook_fn
            hooks_syc.append(model.model.layers[l].register_forward_hook(make_hook(l, cache_syc)))

        with torch.no_grad():
            _ = model(**inputs_syc)
        for h in hooks_syc:
            h.remove()

        # 2. Forward pass on Principled / Non-sycophantic completion
        inputs_non_syc = tokenizer(item["non_syc_full"], return_tensors="pt").to(device)
        cache_non_syc: Dict[int, torch.Tensor] = {}
        hooks_non_syc = []
        for l in layers_to_probe:
            def make_hook_ns(idx, cache):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cache[idx] = h[:, -1, :].detach().cpu()
                return hook_fn
            hooks_non_syc.append(model.model.layers[l].register_forward_hook(make_hook_ns(l, cache_non_syc)))

        with torch.no_grad():
            _ = model(**inputs_non_syc)
        for h in hooks_non_syc:
            h.remove()

        # Append to layer storage
        for l in layers_to_probe:
            syc_storage[l].append(cache_syc[l])
            non_syc_storage[l].append(cache_non_syc[l])

    stacked_syc = {l: torch.cat(syc_storage[l], dim=0) for l in layers_to_probe}
    stacked_non_syc = {l: torch.cat(non_syc_storage[l], dim=0) for l in layers_to_probe}

    return stacked_syc, stacked_non_syc, layers_to_probe


# ==============================================================================
# 3. LINEAR PROBING & MASS-MEAN PEAK LAYER DISCOVERY
# ==============================================================================

def train_probes_and_find_peak(
    act_syc: Dict[int, torch.Tensor],
    act_non_syc: Dict[int, torch.Tensor],
    layers: List[int],
    epochs: int = 60,
    lr: float = 1e-2,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Trains CircuitKIT LinearProbes and computes Mass-Mean contrastive vectors across layers.
    Determines the peak layer L* where sycophancy representation is maximally separable.
    """
    print("\n--- Training Linear Probes & Computing Contrastive Steering Vectors ---")
    results = []
    lr_vectors: Dict[int, torch.Tensor] = {}
    mm_vectors: Dict[int, torch.Tensor] = {}

    d_model = act_syc[layers[0]].shape[1]
    n_samples = act_syc[layers[0]].shape[0]

    for layer_idx in tqdm(layers, desc="Probing Layers"):
        X_syc = act_syc[layer_idx].float()      # Class 1: Sycophantic
        X_non = act_non_syc[layer_idx].float()  # Class 0: Principled

        # 1. Mass-Mean (MM) Difference-in-Means Direction
        mu_syc = X_syc.mean(dim=0)
        mu_non = X_non.mean(dim=0)
        mm_vec = mu_syc - mu_non  # Points towards sycophancy
        mm_unit = mm_vec / (torch.norm(mm_vec, p=2) + 1e-8)
        mm_vectors[layer_idx] = mm_unit

        # Separability (Fisher d-prime)
        proj_syc = torch.matmul(X_syc, mm_unit)
        proj_non = torch.matmul(X_non, mm_unit)
        d_prime = (proj_syc.mean() - proj_non.mean()) / (torch.sqrt(0.5 * (proj_syc.var() + proj_non.var())) + 1e-8)
        mm_acc = ((proj_syc > 0).float().mean() + (proj_non <= 0).float().mean()).item() / 2.0 * 100.0

        # 2. Logistic Regression Probe (CircuitKIT)
        probe = LinearProbe(input_dim=d_model).to(device)
        X = torch.cat([X_non, X_syc], dim=0).to(device)
        y = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)], dim=0).to(device)

        # 80/20 train/val split
        perm = torch.randperm(2 * n_samples)
        split = int(0.8 * 2 * n_samples)
        train_idx, val_idx = perm[:split], perm[split:]

        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            opt.zero_grad()
            logits = probe.get_logits(X[train_idx]).squeeze(-1)
            loss = criterion(logits, y[train_idx])
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe.get_logits(X[val_idx]).squeeze(-1)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            lr_val_acc = (val_preds == y[val_idx]).float().mean().item() * 100.0

        # Extract unit weight vector
        lr_weight = probe.linear.weight.data.squeeze(0).cpu()
        lr_unit = lr_weight / (torch.norm(lr_weight, p=2) + 1e-8)
        lr_vectors[layer_idx] = lr_unit

        # Cosine similarity between MM and LR vectors
        cos_sim = torch.dot(mm_unit, lr_unit).item()

        results.append({
            "Layer": layer_idx,
            "LR_Val_Accuracy": round(lr_val_acc, 2),
            "MM_Separability_dPrime": round(d_prime.item(), 3),
            "MM_Accuracy": round(mm_acc, 2),
            "MM_LR_Cosine_Similarity": round(cos_sim, 3),
        })

    probe_df = pd.DataFrame(results)

    # Identify peak layer (highest separability d-prime with high validation accuracy)
    sorted_df = probe_df.sort_values(by=["MM_Separability_dPrime", "LR_Val_Accuracy"], ascending=False)
    peak_layer = int(sorted_df.iloc[0]["Layer"])
    peak_dprime = sorted_df.iloc[0]["MM_Separability_dPrime"]
    peak_acc = sorted_df.iloc[0]["LR_Val_Accuracy"]

    print(f"\n========================================================")
    print(f"🎯 PEAK SYCOPHANCY LAYER IDENTIFIED: Layer {peak_layer}")
    print(f"   Peak Fisher d-prime Separability: {peak_dprime:.3f}")
    print(f"   Linear Probe Validation Accuracy: {peak_acc:.1f}%")
    print(f"========================================================")

    return {
        "probe_df": probe_df,
        "peak_layer": peak_layer,
        "peak_accuracy": peak_acc,
        "peak_dprime": peak_dprime,
        "mm_vectors": mm_vectors,
        "lr_vectors": lr_vectors,
    }


# ==============================================================================
# 4. ACTIVATION STEERING CONTROLLER
# ==============================================================================

class ActivationSteeringController:
    """
    Intervention hook: h_l <- h_l - alpha * v_l.
    Positive alpha steers away from sycophancy towards principled independence.
    Negative alpha amplifies sycophancy.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        steering_vectors: Dict[int, torch.Tensor],
        layers: List[int],
        alpha: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.steering_vectors = steering_vectors
        self.layers = layers
        self.alpha = alpha
        self.device = device
        self.hooks = []

    def __enter__(self):
        self.hooks = []
        for l in self.layers:
            if l not in self.steering_vectors:
                continue
            v = self.steering_vectors[l].to(self.device)
            layer_module = self.model.model.layers[l]

            def make_steering_hook(vec, a):
                def hook_fn(module, inp, out):
                    if a == 0.0:
                        return out
                    if isinstance(out, tuple):
                        h = out[0]
                        delta = (a * vec).to(dtype=h.dtype, device=h.device).view(1, 1, -1)
                        steered = h - delta
                        return (steered,) + out[1:]
                    else:
                        h = out
                        delta = (a * vec).to(dtype=h.dtype, device=h.device).view(1, 1, -1)
                        return h - delta
                return hook_fn

            handle = layer_module.register_forward_hook(make_steering_hook(v, self.alpha))
            self.hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ==============================================================================
# 5. BEFORE & AFTER STEERING EVALUATION (LOGIT & GENERATIVE ASSESSMENTS)
# ==============================================================================

def get_choice_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Computes next-token logits for '(A)' vs '(B)'.
    Checks both choice formats (' (A)' / ' (B)' and ' A' / ' B').
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    next_logits = outputs.logits[0, -1, :]

    cand_A = [" (A)", " (A", " A", "A", "(A)"]
    cand_B = [" (B)", " (B", " B", "B", "(B)"]

    tids_A = [tokenizer.encode(c, add_special_tokens=False)[0] for c in cand_A]
    tids_B = [tokenizer.encode(c, add_special_tokens=False)[0] for c in cand_B]

    logit_A = max(next_logits[tid].item() for tid in set(tids_A))
    logit_B = max(next_logits[tid].item() for tid in set(tids_B))
    return logit_A, logit_B


def evaluate_test_set_logits(
    test_samples: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluates choice logit probabilities on held-out test questions."""
    syc_count = 0
    non_syc_count = 0
    syc_probs = []
    margins = []

    for item in test_samples:
        prompt = item["question_prompt"]
        logit_A, logit_B = get_choice_logits(model, tokenizer, prompt, device=device)

        # Map to sycophantic vs non-sycophantic
        if "(A)" in item["matching_choice"]:
            l_syc, l_non = logit_A, logit_B
        else:
            l_syc, l_non = logit_B, logit_A

        if l_syc > l_non:
            syc_count += 1
        else:
            non_syc_count += 1

        prob_syc = np.exp(l_syc) / (np.exp(l_syc) + np.exp(l_non) + 1e-12)
        syc_probs.append(prob_syc)
        margins.append(l_non - l_syc)

    n = len(test_samples)
    return {
        "sycophancy_rate": (syc_count / n) * 100.0,
        "principled_rate": (non_syc_count / n) * 100.0,
        "mean_syc_prob": float(np.mean(syc_probs)) * 100.0,
        "mean_margin": float(np.mean(margins)),
    }


def run_generative_assessment(
    samples: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    steering_vectors: Dict[int, torch.Tensor],
    target_layers: List[int],
    alphas: List[float] = [-1.5, 0.0, 1.5],
    max_tokens: int = 40,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Runs free-form text generation before and after steering on representative prompts.
    Shows the qualitative behavioral flip as alpha varies.
    """
    records = []
    print("\n--- Running Generative Assessment Across Steering Strengths ---")

    for s in tqdm(samples, desc="Generative Decoding"):
        prompt = s["question_prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        row = {
            "User_Affiliation": s["user_affiliation"],
            "Prompt_Excerpt": prompt.split("\n\n")[-2].replace("\n", " ") if "\n\n" in prompt else prompt[:120],
            "Sycophantic_Choice": s["matching_choice"],
            "Principled_Choice": s["not_matching_choice"],
        }

        for a in alphas:
            with ActivationSteeringController(model, steering_vectors, target_layers, alpha=a, device=device):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
                col_name = f"Alpha_{a:+0.1f}_" + ("Baseline" if a == 0 else ("AntiSyco" if a > 0 else "Amplified"))
                row[col_name] = gen_text

        records.append(row)

    return pd.DataFrame(records)


def run_complete_steering_sweep(
    test_samples: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe_results: Dict[str, Any],
    alphas: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    probe_type: str = "mm",
    pre_peak_window: int = 4,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Comprehensive evaluation across alphas comparing:
      1. Single Peak Layer (L*)
      2. Pre-Peak Multi-Layer Window (L* - k to L*)
    """
    peak_layer = probe_results["peak_layer"]
    vectors = probe_results["mm_vectors"] if probe_type == "mm" else probe_results["lr_vectors"]
    total_layers = model.config.num_hidden_layers
    min_layer = max(0, peak_layer - pre_peak_window)
    pre_peak_layers = list(range(min_layer, min(total_layers, peak_layer + 1)))

    print(f"\n--- Running Intervention Evaluation Sweep ({probe_type.upper()} Probe) ---")
    print(f"Target Peak Layer: {peak_layer}")
    print(f"Pre-Peak Window: {pre_peak_layers}")
    print(f"Alpha Values: {alphas}")

    results = []

    def evaluate_regime(layers: List[int], regime_name: str):
        for a in tqdm(alphas, desc=f"Evaluating {regime_name}", leave=False):
            with ActivationSteeringController(model, vectors, layers, alpha=a, device=device):
                metrics = evaluate_test_set_logits(test_samples, model, tokenizer, device=device)
            results.append({
                "Regime": regime_name,
                "Target_Layers": str(layers),
                "Alpha": a,
                "Sycophancy_Rate": round(metrics["sycophancy_rate"], 2),
                "Principled_Rate": round(metrics["principled_rate"], 2),
                "Mean_Syc_Prob": round(metrics["mean_syc_prob"], 2),
                "Margin_Principled_Minus_Syc": round(metrics["mean_margin"], 3),
            })

    # Regime 1: Single Peak Layer
    evaluate_regime([peak_layer], regime_name=f"Single Peak Layer (L={peak_layer})")

    # Regime 2: Pre-Peak Window
    evaluate_regime(pre_peak_layers, regime_name=f"Pre-Peak Window (L={min_layer}..{peak_layer})")

    return pd.DataFrame(results)


# ==============================================================================
# 6. PUBLICATION-QUALITY VISUALIZATION SUITE
# ==============================================================================

def plot_steering_results(probe_df: pd.DataFrame, sweep_df: pd.DataFrame, peak_layer: int):
    """Plots a 4-panel mechanistic steering and probing figure."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(18, 12))

    # Panel 1: Layer-wise Representation Separability (Fisher d-prime & Accuracy)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(probe_df["Layer"], probe_df["MM_Separability_dPrime"], marker="o", color="#1f77b4", linewidth=2.5, label="MM Fisher d-prime Separability")
    ax1.axvline(peak_layer, color="#d62728", linestyle=":", linewidth=2.5, label=f"Peak Layer (L* = {peak_layer})")
    ax1.scatter([peak_layer], [probe_df.loc[probe_df["Layer"] == peak_layer, "MM_Separability_dPrime"].values[0]], color="#d62728", s=140, zorder=5)
    ax1.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fisher d-prime Separability", fontsize=11, fontweight="bold")
    ax1.set_title("Layer-wise Sycophancy Representation Emergence\n(Teacher-Forced Choice-Token Separability)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Probe Weight Alignment (LR Probe vs Mass-Mean Vector)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(probe_df["Layer"], probe_df["MM_LR_Cosine_Similarity"], color="#8c564b", alpha=0.8, edgecolor="black", width=0.6)
    ax2.axvline(peak_layer, color="#d62728", linestyle=":", linewidth=2.0, label=f"Peak Layer (L={peak_layer})")
    ax2.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax2.set_title("Direction Alignment Across Layers\n(CircuitKIT LinearProbe vs. Mass-Mean Vector)", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(loc="upper left", frameon=True)

    # Panel 3: Pre- vs Post-Steering Sycophancy Rate Curve
    ax3 = fig.add_subplot(2, 2, 3)
    for regime in sweep_df["Regime"].unique():
        sub = sweep_df[sweep_df["Regime"] == regime]
        ax3.plot(sub["Alpha"], sub["Sycophancy_Rate"], marker="o", linewidth=2.5, label=regime)
    ax3.axvline(0.0, color="gray", linestyle="--", alpha=0.8, label="Baseline (Alpha = 0)")
    ax3.set_xlabel("Intervention Steering Strength (Alpha)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sycophancy Choice Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Sycophancy Suppression Curve\n(Pre-Steering Baseline vs Post-Steering Interventions)", fontsize=13, fontweight="bold")
    ax3.set_ylim(-5, 105)
    ax3.legend(loc="upper right", frameon=True)

    # Panel 4: Margin (Principled Logit - Sycophantic Logit)
    ax4 = fig.add_subplot(2, 2, 4)
    for regime in sweep_df["Regime"].unique():
        sub = sweep_df[sweep_df["Regime"] == regime]
        ax4.plot(sub["Alpha"], sub["Margin_Principled_Minus_Syc"], marker="s", linewidth=2.5, label=regime)
    ax4.axhline(0.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Neutral Decision Boundary (0.0)")
    ax4.axvline(0.0, color="gray", linestyle="--", alpha=0.8, label="Baseline (Alpha = 0)")
    ax4.set_xlabel("Intervention Steering Strength (Alpha)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Logit Margin (Principled - Sycophantic)", fontsize=11, fontweight="bold")
    ax4.set_title("Logit Preference Shift Under Steering\n(Positive Margin Indicates Principled Selection)", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 7. HIGH-LEVEL ORCHESTRATION PIPELINE
# ==============================================================================

def run_sycophancy_steering_experiment(
    model_id: str = "CohereLabs/tiny-aya-global",
    hf_token: Optional[str] = None,
    n_train_samples: int = 80,
    n_test_samples: int = 40,
    layers_to_probe: Optional[List[int]] = None,
    alphas: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    probe_type: str = "mm",
    n_generative_samples: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end Sycophancy Probing and Steering Experiment:
      1. Loads Anthropic sycophancy multiple choice items.
      2. Teacher-forced choice token activation caching across all layers.
      3. CircuitKIT Linear Probing and Peak Layer (L*) identification.
      4. Pre-Steering Baseline Evaluation (alpha=0.0).
      5. Post-Steering Intervention Sweep (alpha != 0.0) across single and pre-peak layers.
      6. Generative Assessment (free-form generation before and after steering).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        dtype = torch.float32

    print(f"========================================================")
    print(f"LOADING MODEL: {model_id} on {device} ({dtype})")
    print(f"========================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=token,
    )
    model.eval()

    # Load dataset
    total_needed = n_train_samples + n_test_samples
    all_samples = load_anthropic_sycophancy_dataset(limit=total_needed)
    train_samples = all_samples[:n_train_samples]
    test_samples = all_samples[n_train_samples:]

    # Step 1: Teacher-forced activation caching
    act_syc, act_non_syc, probed_layers = extract_choice_token_activations(
        samples=train_samples,
        model=model,
        tokenizer=tokenizer,
        layers_to_probe=layers_to_probe,
        device=device,
    )

    # Step 2: Linear Probing & Peak Layer discovery
    probe_results = train_probes_and_find_peak(
        act_syc=act_syc,
        act_non_syc=act_non_syc,
        layers=probed_layers,
    )
    peak_layer = probe_results["peak_layer"]

    # Step 3: Pre-Steering Baseline Evaluation (alpha = 0)
    print("\n--- Running Pre-Steering Baseline Evaluation (Alpha = 0.0) ---")
    baseline_metrics = evaluate_test_set_logits(test_samples, model, tokenizer, device=device)
    print(f"📊 Pre-Steering Baseline Sycophancy Rate: {baseline_metrics['sycophancy_rate']:.1f}%")
    print(f"📊 Pre-Steering Baseline Principled Rate: {baseline_metrics['principled_rate']:.1f}%")
    print(f"📊 Pre-Steering Mean Sycophancy Probability: {baseline_metrics['mean_syc_prob']:.1f}%")

    # Step 4: Post-Steering Intervention Sweep
    sweep_df = run_complete_steering_sweep(
        test_samples=test_samples,
        model=model,
        tokenizer=tokenizer,
        probe_results=probe_results,
        alphas=alphas,
        probe_type=probe_type,
        device=device,
    )

    # Step 5: Generative Assessment (Free-form text generation before & after steering)
    vectors = probe_results["mm_vectors"] if probe_type == "mm" else probe_results["lr_vectors"]
    gen_df = run_generative_assessment(
        samples=test_samples[:n_generative_samples],
        model=model,
        tokenizer=tokenizer,
        steering_vectors=vectors,
        target_layers=[peak_layer],
        alphas=[-1.5, 0.0, 1.5],
        device=device,
    )

    return probe_results["probe_df"], sweep_df, gen_df, probe_results

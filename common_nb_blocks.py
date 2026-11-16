"""Shared build-time notebook cell blocks for the 5 CircuitKIT Steering Experiments.

All generated .ipynb notebooks remain 100% self-contained (zero runtime imports
from this workspace module). This module eliminates duplication across the 5 Python
builder scripts while guaranteeing identical model loading, probing, CircuitKIT
gates/composer usage, Single-Vector vs. PCA Subspace controllers, and Cell 10
rapid experimentation dashboards across all 5 notebooks.
"""

from __future__ import annotations
import json
from pathlib import Path


def md_cell(text: str) -> dict:
    lines = [line + "\n" for line in text.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str) -> dict:
    lines = [line + "\n" for line in code.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def save_notebook(cells: list[dict], out_path: str | Path) -> Path:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    path = Path(out_path)
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def cell1_setup(extra_pip_packages: str = "") -> str:
    extra_line = f'!pip install -q {extra_pip_packages}\n' if extra_pip_packages else ""
    return f'''# ==============================================================================
# CELL 1: DEPENDENCIES, ENVIRONMENT & CIRCUITKIT SETUP
# ==============================================================================
!pip uninstall -y -q torchaudio
!pip install -q "transformer-lens==3.8.0" "transformers-stream-generator<0.1,>=0.0.5" --no-deps
!pip install -q einops jaxtyping fancy_einsum better-abc beartype "typeguard<5,>=4.2" accelerate datasets huggingface_hub pandas matplotlib seaborn tqdm scikit-learn scipy
{extra_line}
import os, sys, gc, json, math, random, re, subprocess, warnings, urllib.request
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

if not os.path.exists("CircuitKIT"):
    subprocess.run(["git", "clone", "-q", "https://github.com/Lexsi-Labs/CircuitKIT.git"], check=True)
if "CircuitKIT/src" not in sys.path:
    sys.path.insert(0, os.path.abspath("CircuitKIT/src"))

from circuitkit.applications.arch_utils import (
    detect_model_architecture,
    get_arch_config,
    validate_model_paths,
    get_layers,
)
from circuitkit.applications.common_utils.linear_probe import LinearProbe
from circuitkit.applications.steering.steering_enhanced import (
    SteeringComposer,
    SteeringEvaluationGates,
    SafetyDatasetSynthesis,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
sns.set_theme(style="whitegrid", context="notebook")
print(f"✅ Environment ready | Device: {{device}} | PyTorch: {{torch.__version__}}")
if torch.cuda.is_available():
    print(f"   GPU: {{torch.cuda.get_device_name(0)}} ({{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB)")
'''


CELL2_MODEL = r'''# ==============================================================================
# CELL 2: MODEL LOADING & CIRCUITKIT ARCHITECTURAL INSPECTION
# ==============================================================================
MODEL_ID = "CohereLabs/tiny-aya-global"

def _resolve_hf_token():
    try:
        from google.colab import userdata
        tok_val = userdata.get("HF_TOKEN")
        if tok_val:
            return tok_val
    except Exception:
        pass
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

HF_TOKEN = _resolve_hf_token()

print(f"⏳ Loading tokenizer & model: {MODEL_ID} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
except OSError as e:
    print("⚠️ Authentication needed for gated model CohereLabs/tiny-aya-global. Prompting HF login...")
    from huggingface_hub import login
    login()
    HF_TOKEN = _resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    token=HF_TOKEN,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# CircuitKIT architectural validation
arch_name = detect_model_architecture(model)
arch_cfg = get_arch_config(arch_name)
validate_model_paths(model, arch_cfg)
decoder_layers = get_layers(model, arch_cfg)

NUM_LAYERS = len(decoder_layers)
D_MODEL = model.config.hidden_size
print(f"✅ Loaded {MODEL_ID} | CircuitKIT Arch: {arch_name} | Layers: {NUM_LAYERS} | d_model: {D_MODEL}")
'''


CELL4_EXTRACTION = r'''# ==============================================================================
# CELL 4: UNIFIED MULTI-LAYER ACTIVATION EXTRACTION (EXACT TOKEN POSITION)
# ==============================================================================
@torch.no_grad()
def extract_pair_activations_all_layers(
    pairs: list,
    model,
    tokenizer,
    num_layers: int,
    batch_size: int = 8,
    device: str = "cuda",
    split_name: str = "train",
):
    """Extract last-token residual activations across all decoder layers for paired prompts.

    Supports both text fields (`pos_prompt`, `neg_prompt`) and exact pre-tokenized
    input_id lists (`pos_input_ids`, `neg_input_ids`) so BPE boundary differences
    at differentiating answer tokens never suffer re-tokenization drift.
    """
    use_pretokenized = ("pos_input_ids" in pairs[0]) and ("neg_input_ids" in pairs[0])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    pos_store = {l: [] for l in range(num_layers)}
    neg_store = {l: [] for l in range(num_layers)}

    def _encode_batch(items, key_ids, key_txt):
        if use_pretokenized:
            seqs = [it[key_ids] for it in items]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for row_i, s in enumerate(seqs):
                input_ids[row_i, -len(s):] = torch.tensor(s, dtype=torch.long)
                attn_mask[row_i, -len(s):] = 1
            return {"input_ids": input_ids.to(device), "attention_mask": attn_mask.to(device)}
        else:
            txts = [it[key_txt] for it in items]
            enc = tokenizer(txts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            return {k: v.to(device) for k, v in enc.items()}

    for start in tqdm(range(0, len(pairs), batch_size), desc=f"Extracting [{split_name}]"):
        batch = pairs[start : start + batch_size]
        for side, store, key_ids, key_txt in [
            ("pos", pos_store, "pos_input_ids", "pos_prompt"),
            ("neg", neg_store, "neg_input_ids", "neg_prompt"),
        ]:
            enc = _encode_batch(batch, key_ids, key_txt)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            # hidden_states[l+1] corresponds to decoder layer l output; left padding => last token at -1
            for l in range(num_layers):
                h_l = out.hidden_states[l + 1][:, -1, :].detach().float().cpu()
                store[l].append(h_l)
            del enc, out

    return {l: torch.cat(pos_store[l], dim=0) for l in range(num_layers)}, {
        l: torch.cat(neg_store[l], dim=0) for l in range(num_layers)
    }


print("⏳ Extracting paired residual activations across all 36 layers...")
EXTRACT_BATCH_SIZE = 8
tr_pos, tr_neg = extract_pair_activations_all_layers(
    train_pairs, model, tokenizer, NUM_LAYERS, batch_size=EXTRACT_BATCH_SIZE, device=device, split_name="train"
)
te_pos, te_neg = extract_pair_activations_all_layers(
    test_pairs, model, tokenizer, NUM_LAYERS, batch_size=EXTRACT_BATCH_SIZE, device=device, split_name="test"
)

layer_cache = {
    l: {
        "train_pos": tr_pos[l],
        "train_neg": tr_neg[l],
        "test_pos": te_pos[l],
        "test_neg": te_neg[l],
    }
    for l in range(NUM_LAYERS)
}
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"✅ Cached all {NUM_LAYERS} layers | Train shape: {layer_cache[0]['train_pos'].shape} | Test shape: {layer_cache[0]['test_pos'].shape}")
'''


CELL5_FITTING = r'''# ==============================================================================
# CELL 5: UNIFIED SINGLE-VECTOR & PCA SUBSPACE FITTING + HELD-OUT PROBING
# ==============================================================================
def _orthonormalize_rows(basis_rows: torch.Tensor) -> torch.Tensor:
    """Modified Gram-Schmidt / QR orthonormalization for row vectors [k, D]."""
    if basis_rows.numel() == 0 or basis_rows.shape[0] == 0:
        return basis_rows
    q, _ = torch.linalg.qr(basis_rows.T.float(), mode="reduced")
    return q.T.contiguous()


def _pairwise_dprime_and_acc(pos_te: torch.Tensor, neg_te: torch.Tensor, dir_vec: torch.Tensor):
    """Held-out Fisher d' (pairwise contrast) and classification accuracy for a 1D direction."""
    s_pos = pos_te @ dir_vec
    s_neg = neg_te @ dir_vec
    diff_s = s_pos - s_neg
    dprime_pair = (diff_s.mean() / (diff_s.std(unbiased=False) + 1e-8)).item()
    pooled_std = torch.sqrt(0.5 * (s_pos.var(unbiased=False) + s_neg.var(unbiased=False)) + 1e-8)
    dprime_sep = ((s_pos.mean() - s_neg.mean()) / pooled_std).item()
    pair_acc = (diff_s > 0).float().mean().item() * 100.0
    return float(dprime_pair), float(dprime_sep), float(pair_acc)


def _subspace_separation_dprime(pos_tr: torch.Tensor, neg_tr: torch.Tensor,
                                pos_te: torch.Tensor, neg_te: torch.Tensor,
                                basis_rows: torch.Tensor):
    """Zero-leakage held-out Fisher d' & accuracy using train-fitted Ridge LDA inside subspace U_k."""
    if basis_rows.shape[0] == 1:
        dp_p, dp_s, acc = _pairwise_dprime_and_acc(pos_te, neg_te, basis_rows[0])
        return dp_p, dp_s, acc
    # Project strictly using train statistics to prevent k-dof overfitting on test!
    Zp_tr = pos_tr @ basis_rows.T  # [N_tr, k]
    Zn_tr = neg_tr @ basis_rows.T
    mean_gap = (Zp_tr.mean(dim=0) - Zn_tr.mean(dim=0))  # [k]
    k_dim = basis_rows.shape[0]
    if Zp_tr.shape[0] > 2:
        cov_p = torch.cov(Zp_tr.T) if k_dim > 1 else Zp_tr.var().view(1, 1)
        cov_n = torch.cov(Zn_tr.T) if k_dim > 1 else Zn_tr.var().view(1, 1)
        reg = 1e-2 * (0.5 * (torch.trace(cov_p) + torch.trace(cov_n)) / k_dim + 1e-6)
        sigma_w = 0.5 * (cov_p + cov_n) + reg * torch.eye(k_dim, dtype=pos_tr.dtype)
        w_sub = torch.linalg.solve(sigma_w, mean_gap)
    else:
        w_sub = mean_gap
    w_full = (w_sub @ basis_rows)
    w_unit = w_full / (w_full.norm() + 1e-8)
    return _pairwise_dprime_and_acc(pos_te, neg_te, w_unit)


def fit_unified_directions_and_probe(
    layer_cache: dict,
    k_max: int = 16,
    probe_epochs: int = 250,
    probe_lr: float = 1e-2,
    device: str = "cpu",
):
    """Fit both Single-Vector (Diff-of-Means, CircuitKIT LinearProbe) and PCA Subspaces
    (Centered+Mean, Uncentered SVD, Centered-Only) identically per layer on the exact same split.
    """
    dir_cache = {}
    rows = []
    num_layers = len(layer_cache)

    for l in tqdm(range(num_layers), desc="Probing Layers (SV + PCA)"):
        pos_tr = layer_cache[l]["train_pos"].float()
        neg_tr = layer_cache[l]["train_neg"].float()
        pos_te = layer_cache[l]["test_pos"].float()
        neg_te = layer_cache[l]["test_neg"].float()
        d_model = pos_tr.shape[1]

        diff_tr = pos_tr - neg_tr
        mu_raw = diff_tr.mean(dim=0)
        mu_raw_norm = float(mu_raw.norm().item())
        v_mm = mu_raw / (mu_raw.norm() + 1e-8)

        # --- CircuitKIT LinearProbe (Logistic Regression) ---
        torch.manual_seed(SEED + l)
        probe = LinearProbe(input_dim=d_model).to(device)
        X_tr = torch.cat([pos_tr, neg_tr], dim=0).to(device)
        y_tr = torch.cat([torch.ones(len(pos_tr)), torch.zeros(len(neg_tr))], dim=0).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        probe.train()
        for _ in range(probe_epochs):
            opt.zero_grad()
            logits = probe.get_logits(X_tr).squeeze(-1)
            loss = loss_fn(logits, y_tr)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            X_te = torch.cat([pos_te, neg_te], dim=0).to(device)
            y_te = torch.cat([torch.ones(len(pos_te)), torch.zeros(len(neg_te))], dim=0).to(device)
            te_logits = probe.get_logits(X_te).squeeze(-1)
            lr_clf_acc = ((torch.sigmoid(te_logits) > 0.5).float() == y_te).float().mean().item() * 100.0
        w_lr = probe.linear.weight.data.squeeze(0).detach().float().cpu()
        v_lr = w_lr / (w_lr.norm() + 1e-8)
        cos_lr_mm = float(torch.dot(v_lr, v_mm).item())
        if cos_lr_mm < 0:
            v_lr = -v_lr
            cos_lr_mm = -cos_lr_mm

        # --- PCA Subspaces ---
        X_centered = diff_tr - mu_raw.unsqueeze(0)
        _, S_c, Vh_c = torch.linalg.svd(X_centered, full_matrices=False)
        centered_evr = (S_c ** 2) / (S_c.pow(2).sum() + 1e-12)
        centered_pcs = Vh_c[: min(k_max, Vh_c.shape[0])].contiguous()

        # Centered+Mean basis: row 0 = v_mm, subsequent rows orthonormalized top centered PCs
        k_avail = min(k_max, 1 + centered_pcs.shape[0])
        basis_cm = _orthonormalize_rows(torch.cat([v_mm.unsqueeze(0), centered_pcs], dim=0))[:k_avail]

        # Uncentered SVD basis
        _, S_u, Vh_u = torch.linalg.svd(diff_tr, full_matrices=False)
        uncentered_evr = (S_u ** 2) / (S_u.pow(2).sum() + 1e-12)
        uncentered_pcs = Vh_u[: min(k_max, Vh_u.shape[0])].contiguous()

        # Align SVD vector signs with positive correlation against v_mm
        for idx_r in range(uncentered_pcs.shape[0]):
            if torch.dot(uncentered_pcs[idx_r], v_mm) < 0:
                uncentered_pcs[idx_r] = -uncentered_pcs[idx_r]

        # --- Held-Out Fisher d' & Accuracy Comparison ---
        dp_mm_pair, dp_mm_sep, acc_mm = _pairwise_dprime_and_acc(pos_te, neg_te, v_mm)
        dp_lr_pair, dp_lr_sep, acc_lr_pair = _pairwise_dprime_and_acc(pos_te, neg_te, v_lr)

        pca_k_stats = {}
        for k in [1, 3, 5, 8]:
            k_eff = min(k, basis_cm.shape[0])
            dp_k, dps_k, acc_k = _subspace_separation_dprime(pos_tr, neg_tr, pos_te, neg_te, basis_cm[:k_eff])
            pca_k_stats[k] = {
                "dprime": dp_k,
                "dprime_sep": dps_k,
                "pair_acc": acc_k,
                "cum_evr": float(centered_evr[: max(0, k_eff - 1)].sum().item()) if k_eff > 1 else 0.0,
            }

        k_u = min(5, uncentered_pcs.shape[0])
        dp_u5, _, acc_u5 = _subspace_separation_dprime(pos_tr, neg_tr, pos_te, neg_te, uncentered_pcs[:k_u])

        dir_cache[l] = {
            "v_mm": v_mm,
            "v_lr": v_lr,
            "mu_raw": mu_raw,
            "mu_raw_norm": mu_raw_norm,
            "centered_pcs": centered_pcs,
            "basis_cm": basis_cm,
            "uncentered_pcs": uncentered_pcs,
            "centered_evr": centered_evr[:k_max],
            "uncentered_evr": uncentered_evr[:k_max],
            "S_centered": S_c[:k_max],
            "S_uncentered": S_u[:k_max],
            "cos_lr_mm": cos_lr_mm,
            "dprime_mm": dp_mm_pair,
            "dprime_lr": dp_lr_pair,
            "dprime_pca_k5": pca_k_stats[5]["dprime"],
            "dprime_unc_k5": dp_u5,
        }

        rows.append({
            "layer": l,
            "mu_raw_norm": mu_raw_norm,
            "cos_lr_mm": cos_lr_mm,
            "dprime_mm": dp_mm_pair,
            "dprime_lr": dp_lr_pair,
            "pair_acc_mm": acc_mm,
            "pair_acc_lr": acc_lr_pair,
            "lr_clf_acc": lr_clf_acc,
            "dprime_pca_k1": pca_k_stats[1]["dprime"],
            "dprime_pca_k3": pca_k_stats[3]["dprime"],
            "dprime_pca_k5": pca_k_stats[5]["dprime"],
            "dprime_pca_k8": pca_k_stats[8]["dprime"],
            "dprime_unc_k5": dp_u5,
            "pair_acc_pca_k5": pca_k_stats[5]["pair_acc"],
            "evr_pc1": float(centered_evr[0].item()),
            "evr_top5": float(centered_evr[:5].sum().item()),
            "evr_top8": float(centered_evr[:8].sum().item()),
            "unc_evr_top5": float(uncentered_evr[:5].sum().item()),
        })

    probe_df = pd.DataFrame(rows)
    return dir_cache, probe_df


dir_cache, probe_df = fit_unified_directions_and_probe(layer_cache, k_max=16, probe_epochs=250, device="cpu")

L_STAR_MM = int(probe_df.loc[probe_df["dprime_mm"].idxmax(), "layer"])
L_STAR_LR = int(probe_df.loc[probe_df["dprime_lr"].idxmax(), "layer"])
L_STAR_PCA = int(probe_df.loc[probe_df["dprime_pca_k5"].idxmax(), "layer"])
L_STAR = L_STAR_MM
WINDOW_LAYERS = list(range(max(0, L_STAR - 4), min(NUM_LAYERS, L_STAR + 1)))

print(f"✅ Unified Probing Complete:")
print(f"   • Peak Layer Diff-of-Means (L*_MM): L{L_STAR_MM} (d'={probe_df.loc[L_STAR_MM, 'dprime_mm']:.3f}, PairAcc={probe_df.loc[L_STAR_MM, 'pair_acc_mm']:.1f}%)")
print(f"   • Peak Layer LinearProbe (L*_LR):    L{L_STAR_LR} (d'={probe_df.loc[L_STAR_LR, 'dprime_lr']:.3f}, ClfAcc={probe_df.loc[L_STAR_LR, 'lr_clf_acc']:.1f}%)")
print(f"   • Peak Layer PCA-k5 (L*_PCA):        L{L_STAR_PCA} (d'={probe_df.loc[L_STAR_PCA, 'dprime_pca_k5']:.3f}, EVR@5={probe_df.loc[L_STAR_PCA, 'evr_top5']*100:.1f}%)")
print(f"   • Cos(LR, DiffOfMeans) @ L*:         {probe_df.loc[L_STAR, 'cos_lr_mm']:.3f}")
print(f"   • Default Multi-Layer Window:        {WINDOW_LAYERS}")
display(probe_df.iloc[max(0, L_STAR - 6) : min(NUM_LAYERS, L_STAR + 5)].round(3))
'''


CELL6_CONTROLLERS = r'''# ==============================================================================
# CELL 6: UNIFIED STEERING CONTROLLERS + CIRCUITKIT COMPOSER & EVALUATION GATES
# ==============================================================================
def _compute_layer_weights(dir_cache: dict, target_layers: list, layer_alphas=None, key_dp: str = "dprime_mm"):
    if layer_alphas is None:
        return {l: 1.0 for l in target_layers}
    if layer_alphas == "dprime_weighted":
        raw = [max(0.05, float(dir_cache[l][key_dp])) for l in target_layers]
        mean_w = float(np.mean(raw)) if raw else 1.0
        return {l: w / (mean_w + 1e-8) for l, w in zip(target_layers, raw)}
    if isinstance(layer_alphas, (list, tuple)) and len(layer_alphas) == len(target_layers):
        return {l: float(w) for l, w in zip(target_layers, layer_alphas)}
    raise ValueError(f"Unsupported layer_alphas={layer_alphas}")


def build_single_vector_for_layers(
    dir_cache: dict,
    target_layers: list,
    vector_method: str = "diff_of_means",
    blend_lr_weight: float = 0.5,
    layer_alphas=None,
):
    """Build per-layer 1D steering directions & register into CircuitKIT SteeringComposer."""
    dp_key = "dprime_lr" if vector_method == "lr" else "dprime_mm"
    layer_weights = _compute_layer_weights(dir_cache, target_layers, layer_alphas, key_dp=dp_key)
    composer = SteeringComposer()
    vecs = {}
    raw_norms = {}
    for l in target_layers:
        d = dir_cache[l]
        if vector_method == "diff_of_means":
            v = d["v_mm"]
        elif vector_method == "lr":
            v = d["v_lr"]
        elif vector_method == "blend":
            v = (1.0 - blend_lr_weight) * d["v_mm"] + blend_lr_weight * d["v_lr"]
            v = v / (v.norm() + 1e-8)
        else:
            raise ValueError(f"Unknown vector_method={vector_method}")
        vecs[l] = v.contiguous()
        raw_norms[l] = float(d["mu_raw_norm"])
        composer.add_steering(f"SV_{vector_method}_L{l}", {f"L{l}": v}, coefficient=layer_weights[l])
    return {
        "kind": "single_vector",
        "vector_method": vector_method,
        "vecs": vecs,
        "raw_norms": raw_norms,
        "layer_weights": layer_weights,
        "composer": composer,
    }


def build_subspace_for_layers(
    dir_cache: dict,
    target_layers: list,
    method: str = "centered_with_mean",
    k: int = 5,
    mean_weight: float = 1.0,
    layer_alphas=None,
):
    """Build per-layer k-dim orthonormal PCA bases U_k & register primary axis in SteeringComposer."""
    layer_weights = _compute_layer_weights(dir_cache, target_layers, layer_alphas, key_dp="dprime_pca_k5")
    composer = SteeringComposer()
    bases = {}
    mean_vecs = {}
    raw_norms = {}
    centered_tails = {}
    for l in target_layers:
        d = dir_cache[l]
        if method == "centered_with_mean":
            # Row 0 = v_mm; Rows 1..k-1 = orthonormalized top centered PCs
            k_eff = min(k, d["basis_cm"].shape[0])
            U_k = d["basis_cm"][:k_eff].contiguous()
            U_c = U_k[1:].contiguous() if k_eff > 1 else U_k[:0]
        elif method == "uncentered_svd":
            k_eff = min(k, d["uncentered_pcs"].shape[0])
            U_k = d["uncentered_pcs"][:k_eff].contiguous()
            # Residual orthogonal to v_mm inside U_k for additive mode
            proj_mm = (U_k @ d["v_mm"]).unsqueeze(-1) * d["v_mm"].unsqueeze(0)
            U_c = _orthonormalize_rows(U_k - proj_mm)[: max(0, k_eff - 1)]
        elif method == "centered_only":
            k_eff = min(k, d["centered_pcs"].shape[0])
            U_k = d["centered_pcs"][:k_eff].contiguous()
            U_c = U_k
        else:
            raise ValueError(f"Unknown subspace method={method}")

        bases[l] = U_k
        mean_vecs[l] = d["v_mm"].contiguous()
        raw_norms[l] = float(d["mu_raw_norm"])
        centered_tails[l] = U_c
        composer.add_steering(f"PCA_{method}_k{k}_L{l}", {f"L{l}": U_k[0]}, coefficient=layer_weights[l])

    return {
        "kind": "pca_subspace",
        "method": method,
        "k": k,
        "mean_weight": mean_weight,
        "bases": bases,
        "mean_vecs": mean_vecs,
        "raw_norms": raw_norms,
        "centered_tails": centered_tails,
        "layer_weights": layer_weights,
        "composer": composer,
    }


class SingleVectorSteeringController:
    """Context manager applying Single-Vector Projection Erasure or Empirical-Norm Additive Steering."""

    def __init__(self, decoder_layers, vec_spec: dict, target_layers: list, alpha: float = 1.0,
                 steering_mode: str = "projection", additive_norm_scale: str = "empirical", device: str = "cuda"):
        self.decoder_layers = decoder_layers
        self.vec_spec = vec_spec
        self.target_layers = list(target_layers)
        self.alpha = float(alpha)
        self.steering_mode = steering_mode
        self.additive_norm_scale = additive_norm_scale
        self.device = device
        self.handles = []

    def _make_hook(self, layer_idx: int):
        v = self.vec_spec["vecs"][layer_idx]
        w = self.vec_spec["layer_weights"][layer_idx]
        step_norm = self.vec_spec["raw_norms"][layer_idx] if self.additive_norm_scale == "empirical" else 1.0

        def hook(module, inputs, outputs):
            if abs(self.alpha) < 1e-9:
                return outputs
            is_tuple = isinstance(outputs, tuple)
            h = outputs[0] if is_tuple else outputs
            v_dev = v.to(device=h.device, dtype=h.dtype)
            if self.steering_mode == "projection":
                coeff = (h * v_dev).sum(dim=-1, keepdim=True)
                h_new = h - (self.alpha * w) * coeff * v_dev
            elif self.steering_mode == "additive":
                h_new = h - (self.alpha * w * step_norm) * v_dev
            else:
                raise ValueError(f"Unknown steering_mode={self.steering_mode}")
            return (h_new,) + outputs[1:] if is_tuple else h_new

        return hook

    def __enter__(self):
        if abs(self.alpha) >= 1e-9:
            for l in self.target_layers:
                self.handles.append(self.decoder_layers[l].register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []


class SubspaceSteeringController:
    """Context manager applying k-dim PCA Subspace Projection Erasure or Hybrid Subspace Additive Steering.

    Mathematical guarantees:
      - At k=1 ('centered_with_mean'), Projection equals SingleVector(diff_of_means, projection).
      - At k=1 ('centered_with_mean'), Additive equals SingleVector(diff_of_means, additive).
      - At k>1 Additive, applies empirical CAA mean shift (mu_raw) PLUS subspace variance erasure along U_c.
    """

    def __init__(self, decoder_layers, sub_spec: dict, target_layers: list, alpha: float = 1.0,
                 steering_mode: str = "projection", additive_norm_scale: str = "empirical", device: str = "cuda"):
        self.decoder_layers = decoder_layers
        self.sub_spec = sub_spec
        self.target_layers = list(target_layers)
        self.alpha = float(alpha)
        self.steering_mode = steering_mode
        self.additive_norm_scale = additive_norm_scale
        self.device = device
        self.handles = []

    def _make_hook(self, layer_idx: int):
        U_k = self.sub_spec["bases"][layer_idx]
        U_c = self.sub_spec["centered_tails"][layer_idx]
        v_mu = self.sub_spec["mean_vecs"][layer_idx]
        w = self.sub_spec["layer_weights"][layer_idx]
        mw = self.sub_spec["mean_weight"]
        step_norm = self.sub_spec["raw_norms"][layer_idx] if self.additive_norm_scale == "empirical" else 1.0

        def hook(module, inputs, outputs):
            if abs(self.alpha) < 1e-9:
                return outputs
            is_tuple = isinstance(outputs, tuple)
            h = outputs[0] if is_tuple else outputs

            if self.steering_mode == "projection":
                U_dev = U_k.to(device=h.device, dtype=h.dtype)
                proj = (h @ U_dev.T) @ U_dev
                h_new = h - (self.alpha * w) * proj
            elif self.steering_mode == "additive":
                delta = 0.0
                if self.sub_spec["method"] != "centered_only":
                    v_dev = v_mu.to(device=h.device, dtype=h.dtype)
                    delta = (self.alpha * w * mw * step_norm) * v_dev
                if U_c.shape[0] > 0:
                    Uc_dev = U_c.to(device=h.device, dtype=h.dtype)
                    tail_proj = (h @ Uc_dev.T) @ Uc_dev
                    delta = delta + (self.alpha * w) * tail_proj
                h_new = h - delta
            else:
                raise ValueError(f"Unknown steering_mode={self.steering_mode}")

            return (h_new,) + outputs[1:] if is_tuple else h_new

        return hook

    def __enter__(self):
        if abs(self.alpha) >= 1e-9:
            for l in self.target_layers:
                self.handles.append(self.decoder_layers[l].register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []


def make_steering_controller(spec: dict, decoder_layers, target_layers: list, alpha: float,
                             steering_mode: str = "projection", additive_norm_scale: str = "empirical", device: str = "cuda"):
    if spec["kind"] == "single_vector":
        return SingleVectorSteeringController(
            decoder_layers, spec, target_layers, alpha=alpha,
            steering_mode=steering_mode, additive_norm_scale=additive_norm_scale, device=device,
        )
    return SubspaceSteeringController(
        decoder_layers, spec, target_layers, alpha=alpha,
        steering_mode=steering_mode, additive_norm_scale=additive_norm_scale, device=device,
    )


def compute_semantic_similarity(base_logits_ref: torch.Tensor, steered_logits: torch.Tensor, gates: SteeringEvaluationGates):
    """Wrap CircuitKIT SteeringEvaluationGates semantic preservation score robustly across tensor shapes."""
    if base_logits_ref is None or steered_logits is None:
        return 1.0
    b = base_logits_ref.float()
    s = steered_logits.float()
    try:
        _, sim = gates.check_semantic_preservation(b.to(gates.device), s.to(gates.device), min_similarity=0.90)
        return float(sim)
    except Exception:
        # Fallback cosine similarity on final-token logits
        cos = F.cosine_similarity(b.reshape(-1, b.shape[-1]), s.reshape(-1, s.shape[-1]), dim=-1)
        return float(cos.mean().item())


gates = SteeringEvaluationGates(model, device=device)
synthesizer = SafetyDatasetSynthesis(model, device=device)

# CircuitKIT safety gate check on default peak window vectors
_demo_sv = build_single_vector_for_layers(dir_cache, WINDOW_LAYERS, vector_method="diff_of_means")
_demo_pca = build_subspace_for_layers(dir_cache, WINDOW_LAYERS, method="centered_with_mean", k=5)
_gate_sv = gates.run_all_checks(_demo_sv["composer"].get_composed_vectors(aggregate="sum"))
_gate_pca = gates.run_all_checks(_demo_pca["composer"].get_composed_vectors(aggregate="sum"))

print("✅ Steering Controllers & CircuitKIT Safety Systems initialized:")
print(f"   • SV Composer Gates  | bounds_valid={_gate_sv['activation_bounds']['valid']} | consistency={_gate_sv['consistency']['score']:.3f}")
print(f"   • PCA Composer Gates | bounds_valid={_gate_pca['activation_bounds']['valid']} | consistency={_gate_pca['consistency']['score']:.3f}")
'''


CELL7_FIG1_PROBING = r'''# ==============================================================================
# CELL 7: FIGURE 1 — MULTI-LAYER PROBING, LR vs DIFF-OF-MEANS & PCA SCREE DIAGNOSTICS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 5.0))

# Panel 1: Layer-wise Held-Out Fisher d' (Single Vectors vs PCA Subspaces)
axes[0].plot(probe_df["layer"], probe_df["dprime_mm"], label="SV: Diff-of-Means", color="#1f77b4", linewidth=2.2)
axes[0].plot(probe_df["layer"], probe_df["dprime_lr"], label="SV: CircuitKIT LR", color="#ff7f0e", linestyle="--", linewidth=2.0)
axes[0].plot(probe_df["layer"], probe_df["dprime_pca_k3"], label="PCA Subspace (k=3)", color="#2ca02c", linewidth=2.0)
axes[0].plot(probe_df["layer"], probe_df["dprime_pca_k5"], label="PCA Subspace (k=5)", color="#d62728", linewidth=2.4)
axes[0].axvline(L_STAR, color="gray", linestyle=":", label=f"Peak L*={L_STAR}")
axes[0].set_title("Held-Out Fisher d' Across Layers", fontweight="bold")
axes[0].set_xlabel("Decoder Layer")
axes[0].set_ylabel("Held-Out Pairwise d'")
axes[0].legend(fontsize=8, loc="upper left")

# Panel 2: Pairwise Separation Acc & Cos(LR, Diff-of-Means)
axes[1].plot(probe_df["layer"], probe_df["pair_acc_mm"], label="MM Pairwise Acc (%)", color="#1f77b4", linewidth=2.0)
axes[1].plot(probe_df["layer"], probe_df["pair_acc_pca_k5"], label="PCA-k5 Pairwise Acc (%)", color="#d62728", linewidth=2.2)
ax1_r = axes[1].twinx()
ax1_r.plot(probe_df["layer"], probe_df["cos_lr_mm"], label="Cos(LR, MM)", color="#9467bd", linestyle="-.", linewidth=1.8)
ax1_r.set_ylabel("Cosine(LR, Diff-of-Means)", color="#9467bd")
ax1_r.set_ylim(-0.1, 1.05)
axes[1].set_title("Held-Out Pairwise Acc vs LR-MM Alignment", fontweight="bold")
axes[1].set_xlabel("Decoder Layer")
axes[1].set_ylabel("Held-Out Accuracy (%)")
axes[1].set_ylim(45, 103)
axes[1].legend(fontsize=8, loc="lower left")

# Panel 3: Centered PCA Scree Plot at Peak Layer L* & Adjacent Window
for l_plot, col in [(max(0, L_STAR - 4), "#1f77b4"), (L_STAR, "#d62728"), (min(NUM_LAYERS - 1, L_STAR + 2), "#2ca02c")]:
    evr = dir_cache[l_plot]["centered_evr"].numpy() * 100.0
    cum_evr = np.cumsum(evr)
    axes[2].plot(np.arange(1, len(evr) + 1), cum_evr, marker="o", markersize=4, label=f"L{l_plot} Cum EVR (%)", color=col)
axes[2].axhline(80.0, color="gray", linestyle="--", alpha=0.6, label="80% Variance")
axes[2].set_title(f"Residual PCA Cumulative Explained Variance (L*={L_STAR})", fontweight="bold")
axes[2].set_xlabel("Principal Component Index (Centered)")
axes[2].set_ylabel("Cumulative Explained Variance (%)")
axes[2].set_ylim(0, 105)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
'''


def cell10_rapid_control_block(
    primary_metric_col: str,
    primary_metric_title: str,
    secondary_metric_col: str,
    secondary_metric_title: str,
    margin_col: str,
    margin_title: str,
    spot_check_fn_code: str,
) -> str:
    return f'''# ==============================================================================
# CELL 10: RAPID HYPERPARAMETER & HEAD-TO-HEAD EXPERIMENTATION CONTROL BLOCK
# ==============================================================================
# Tweak any setting below and re-run ONLY this cell! Uses cached `dir_cache`, `model`, `test_items`.

EXPERIMENT_CONFIG = {{
    # "single_vector" | "pca_subspace" | "compare_both"
    "mode": "compare_both",

    # Layer choices: [L_STAR], [L_STAR_LR], [L_STAR_PCA], WINDOW_LAYERS, or custom list
    "target_layers": WINDOW_LAYERS,

    # Single-Vector settings: "diff_of_means" | "lr" | "blend"
    "sv_method": "diff_of_means",
    "blend_lr_weight": 0.5,

    # PCA Subspace settings: "centered_with_mean" | "uncentered_svd" | "centered_only"
    "pca_method": "centered_with_mean",
    "k": 5,
    "mean_weight": 1.0,

    # Layer weighting: None | "dprime_weighted" | list of floats
    "layer_alphas": None,

    # Steering operator: "projection" | "additive"
    "steering_mode": "projection",
    "additive_norm_scale": "empirical",  # "empirical" (||mu_raw||) | "unit"

    # Global alpha sweep
    "alphas": [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "eval_batch_size": 8,
    "run_circuitkit_gates": True,
    "run_generative_check": True,
}}

specs_to_run = []
if EXPERIMENT_CONFIG["mode"] in ("single_vector", "compare_both"):
    sv_s = build_single_vector_for_layers(
        dir_cache, EXPERIMENT_CONFIG["target_layers"],
        vector_method=EXPERIMENT_CONFIG["sv_method"],
        blend_lr_weight=EXPERIMENT_CONFIG["blend_lr_weight"],
        layer_alphas=EXPERIMENT_CONFIG["layer_alphas"],
    )
    specs_to_run.append((f"SV ({{EXPERIMENT_CONFIG['sv_method']}})", sv_s, "#1f77b4", "o"))

if EXPERIMENT_CONFIG["mode"] in ("pca_subspace", "compare_both"):
    pca_s = build_subspace_for_layers(
        dir_cache, EXPERIMENT_CONFIG["target_layers"],
        method=EXPERIMENT_CONFIG["pca_method"],
        k=EXPERIMENT_CONFIG["k"],
        mean_weight=EXPERIMENT_CONFIG["mean_weight"],
        layer_alphas=EXPERIMENT_CONFIG["layer_alphas"],
    )
    specs_to_run.append((f"PCA ({{EXPERIMENT_CONFIG['pca_method']}}, k={{EXPERIMENT_CONFIG['k']}})", pca_s, "#d62728", "s"))

print("=" * 84)
print(f"RUNNING CELL 10 EXPERIMENT | Mode={{EXPERIMENT_CONFIG['mode']}} | Op={{EXPERIMENT_CONFIG['steering_mode']}}")
print(f"Layers={{EXPERIMENT_CONFIG['target_layers']}} | Alphas={{EXPERIMENT_CONFIG['alphas']}}")
if EXPERIMENT_CONFIG["run_circuitkit_gates"]:
    for label, sp, _, _ in specs_to_run:
        comp = sp["composer"].get_composed_vectors(aggregate="sum")
        rep = gates.run_all_checks(comp)
        print(f"   Gate [{{label}}]: bounds_valid={{rep['activation_bounds']['valid']}} | consistency={{rep['consistency']['score']:.3f}}")
print("=" * 84)

rapid_dfs = {{}}
fig, axes = plt.subplots(1, 4, figsize=(23, 4.5))
for label, sp, color, marker in specs_to_run:
    rows = []
    for a in tqdm(EXPERIMENT_CONFIG["alphas"], desc=f"Sweeping {{label}}"):
        with make_steering_controller(
            sp, decoder_layers, EXPERIMENT_CONFIG["target_layers"], alpha=a,
            steering_mode=EXPERIMENT_CONFIG["steering_mode"],
            additive_norm_scale=EXPERIMENT_CONFIG["additive_norm_scale"],
            device=device,
        ):
            m = evaluate_test_suite(
                test_items, model, tokenizer,
                batch_size=EXPERIMENT_CONFIG["eval_batch_size"],
                device=device,
                base_logits_ref=BASE_TEST_LOGITS,
                gates=gates,
            )
        m["Alpha"] = a
        m["Method"] = label
        rows.append(m)
    df_m = pd.DataFrame(rows)
    rapid_dfs[label] = df_m

    axes[0].plot(df_m["Alpha"], df_m["{primary_metric_col}"], marker=marker, color=color, linewidth=2.2, label=label)
    axes[1].plot(df_m["Alpha"], df_m["{secondary_metric_col}"], marker=marker, color=color, linewidth=2.2, label=label)
    axes[2].plot(df_m["Alpha"], df_m["{margin_col}"], marker=marker, color=color, linewidth=2.2, label=label)
    axes[3].plot(df_m["Alpha"], df_m["Semantic_Sim"], marker=marker, color=color, linewidth=2.2, label=label)

axes[0].set_title("{primary_metric_title}", fontweight="bold")
axes[1].set_title("{secondary_metric_title}", fontweight="bold")
axes[2].set_title("{margin_title}", fontweight="bold")
axes[3].set_title("CircuitKIT Semantic Preservation", fontweight="bold")
axes[3].axhline(0.90, color="red", linestyle="--", alpha=0.7)
axes[3].set_ylim(0.0, 1.05)
for ax in axes:
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Alpha")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

combined_rapid_df = pd.concat(list(rapid_dfs.values()), ignore_index=True)
display(combined_rapid_df.round(3))

{spot_check_fn_code}
'''

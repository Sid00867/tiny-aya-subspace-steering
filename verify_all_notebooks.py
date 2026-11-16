"""Comprehensive verification suite for the 5 CircuitKIT steering notebooks.

Validates:
  1. Valid JSON & nbformat 4 schema for all 5 generated .ipynb files.
  2. Clean Python AST parsing on every code cell (stripping Colab '!' / '%' lines).
  3. Live dataset ingestion, stratification, zero-leakage split checks, and
     deterministic grader/metric unit tests executed directly from the emitted
     notebook cells across all 5 experiments.
  4. End-to-end dry run of Cell 3 -> Cell 4 -> Cell 5 -> Cell 6 -> Cell 8 (with
     actual forward hooks active!) using a tiny PyTorch stub LM over REAL
     downloaded samples from each dataset.
"""

from __future__ import annotations
import ast
import json
import math
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


NOTEBOOKS = [
    "exp1_crosslingual_pushback_amosluna_circuitkit.ipynb",
    "exp2_crosslingual_open_sycobench_circuitkit.ipynb",
    "exp3_cognitive_biases_tum_circuitkit.ipynb",
    "exp4_demographic_skew_bbq_circuitkit.ipynb",
    "exp5_unfaithful_moralizing_persona_circuitkit.ipynb",
]


def _strip_colab_magics(src: str) -> str:
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith("!") or s.startswith("%"):
            indent = line[: len(line) - len(s)]
            out.append(f"{indent}pass  # MAGIC: {s}")
        else:
            out.append(line)
    return "\n".join(out)


def load_nb_code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, f"{nb_path}: invalid nbformat={nb.get('nbformat')}"
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 10, f"{nb_path}: expected 10 code cells, found {len(code_cells)}"
    for i, src in enumerate(code_cells, start=1):
        ast.parse(_strip_colab_magics(src), filename=f"{nb_path.name}:cell_{i}")
    return code_cells


class _BatchEnc(dict):
    def to(self, device):
        return _BatchEnc({k: v.to(device) for k, v in self.items()})


class _MockTokenizer:
    """Deterministic byte-level char tokenizer mimicking HF PreTrainedTokenizer."""
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b + 4 for b in text.encode("utf-8")]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        bs = bytes([(int(i) - 4) % 256 for i in ids if int(i) >= 4])
        return bs.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("ASSISTANT: ")
        return "\n".join(parts)

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=2048):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t)[:max_length] for t in texts]
        mx = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), mx), dtype=torch.long)
        att = torch.zeros((len(seqs), mx), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, -len(s):] = torch.tensor(s, dtype=torch.long)
            att[i, -len(s):] = 1
        return _BatchEnc({"input_ids": ids, "attention_mask": att})


class _DummyBlock(nn.Module):
    def forward(self, x):
        return (x,)


class _StubCausalLM(nn.Module):
    def __init__(self, vocab: int = 300, d_model: int = 32, num_layers: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        h = self.emb(input_ids.clamp(max=299))
        hs = [h]
        for blk in self.blocks:
            h = blk(h)[0]
            hs.append(h)
        logits = self.head(h)
        return type("Out", (), {"logits": logits, "hidden_states": tuple(hs) if output_hidden_states else None})()


class _StubLinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def get_logits(self, x):
        return self.linear(x)


class _StubSteeringComposer:
    def __init__(self):
        self.entries = []

    def add_steering(self, name, vecs, coefficient=1.0):
        self.entries.append((name, vecs, coefficient))

    def get_composed_vectors(self, aggregate="sum"):
        out = {}
        for _, vecs, coeff in self.entries:
            for k, v in vecs.items():
                out[k] = out.get(k, 0) + coeff * v
        return out


class _StubGates:
    def __init__(self, model=None, device="cpu"):
        self.device = device

    def run_all_checks(self, composed_vecs):
        return {"activation_bounds": {"valid": True}, "consistency": {"score": 1.0}}

    def check_semantic_preservation(self, b, s, min_similarity=0.90):
        sim = F.cosine_similarity(b.reshape(-1, b.shape[-1]), s.reshape(-1, s.shape[-1]), dim=-1).mean().item()
        return (sim >= min_similarity), sim


class _StubSafetySynth:
    def __init__(self, model=None, device="cpu"):
        pass

    def generate_adversarial_prompts(self, prompt_str, num_variations=3, perturbation_type="reorder"):
        return [f"[var {i}] {prompt_str}" for i in range(num_variations)]


def make_base_env():
    stub_lm = _StubCausalLM(vocab=300, d_model=32, num_layers=4)
    return {
        "os": os, "sys": sys, "gc": __import__("gc"), "re": re, "json": json,
        "math": math, "random": random, "urllib": urllib, "np": np, "pd": pd,
        "torch": torch, "nn": nn, "F": F, "tqdm": tqdm,
        "Counter": Counter, "defaultdict": defaultdict,
        "display": lambda *args, **kwargs: None,
        "SEED": 42, "NUM_LAYERS": 4, "D_MODEL": 32, "device": "cpu",
        "tokenizer": _MockTokenizer(),
        "model": stub_lm,
        "decoder_layers": stub_lm.blocks,
        "LinearProbe": _StubLinearProbe,
        "SteeringComposer": _StubSteeringComposer,
        "SteeringEvaluationGates": _StubGates,
        "SafetyDatasetSynthesis": _StubSafetySynth,
    }


def run_dry_pipeline(env, cells, label: str, max_tr: int = 12, max_te: int = 8):
    """Run Cells 4, 5, 6, 8 end-to-end (including steered forward pass!) on a slice of real items."""
    env["train_pairs"] = env["train_pairs"][:max_tr]
    env["test_pairs"] = env["test_pairs"][:max_te]
    if len(env["test_items"]) > max_te:
        # Preserve subset diversity by evenly striding test_items
        step = max(1, len(env["test_items"]) // max_te)
        env["test_items"] = env["test_items"][::step][:max_te]

    exec(cells[3], env)  # Cell 4: batched extraction
    assert env["layer_cache"][0]["train_pos"].shape[1] == 32
    exec(cells[4], env)  # Cell 5: unified probing (SV + PCA)
    exec(cells[5], env)  # Cell 6: controllers + CircuitKIT gates
    exec(cells[7], env)  # Cell 8: baseline evaluate_test_suite

    # Verify steered forward passes for BOTH Single-Vector and PCA Subspace in Projection & Additive
    sv_s = env["build_single_vector_for_layers"](env["dir_cache"], env["WINDOW_LAYERS"], "diff_of_means")
    pca_s = env["build_subspace_for_layers"](env["dir_cache"], env["WINDOW_LAYERS"], "centered_with_mean", k=5)
    for sp in (sv_s, pca_s):
        for op in ("projection", "additive"):
            with env["make_steering_controller"](sp, env["decoder_layers"], env["WINDOW_LAYERS"], alpha=1.0, steering_mode=op, device="cpu"):
                res = env["evaluate_test_suite"](
                    env["test_items"], env["model"], env["tokenizer"],
                    batch_size=4, device="cpu", base_logits_ref=env["BASE_TEST_LOGITS"], gates=env["gates"]
                )
            assert "Semantic_Sim" in res and not math.isnan(res["Semantic_Sim"]), f"{label}: invalid Semantic_Sim"
    print(f"  [PASS] {label} End-to-End Dry Run (Cells 3->4->5->6->8 + SV/PCA Proj/Add Steered Passes)")


def test_probing_and_steering_math(cell5_src: str, cell6_src: str):
    env = make_base_env()
    torch.manual_seed(42)
    N_tr, N_te, L, D = 60, 20, 4, 32
    true_dir = F.normalize(torch.randn(D), dim=0)
    layer_cache = {}
    for l in range(L):
        scale = 0.2 if l != 2 else 2.5
        neg_tr = torch.randn(N_tr, D) * 0.5
        pos_tr = neg_tr + scale * true_dir + torch.randn(N_tr, D) * 0.2
        neg_te = torch.randn(N_te, D) * 0.5
        pos_te = neg_te + scale * true_dir + torch.randn(N_te, D) * 0.2
        layer_cache[l] = {"train_pos": pos_tr, "train_neg": neg_tr, "test_pos": pos_te, "test_neg": neg_te}

    env["layer_cache"] = layer_cache
    decoder_layers = env["decoder_layers"]

    exec(cell5_src, env)
    assert env["L_STAR"] == 2, f"Expected peak layer L2, got L{env['L_STAR']}"
    U5 = env["dir_cache"][2]["basis_cm"][:5]
    orth_err = (U5 @ U5.T - torch.eye(5)).abs().max().item()
    assert orth_err < 1e-5, f"Basis non-orthogonal: err={orth_err}"

    exec(cell6_src, env)
    x_in = torch.randn(4, 10, D)

    sv_spec = env["build_single_vector_for_layers"](env["dir_cache"], [2], "diff_of_means")
    p1_spec = env["build_subspace_for_layers"](env["dir_cache"], [2], "centered_with_mean", k=1)

    for op in ("projection", "additive"):
        with env["SingleVectorSteeringController"](decoder_layers, sv_spec, [2], alpha=1.25, steering_mode=op, device="cpu"):
            y_sv = decoder_layers[2](x_in)[0]
        with env["SubspaceSteeringController"](decoder_layers, p1_spec, [2], alpha=1.25, steering_mode=op, device="cpu"):
            y_p1 = decoder_layers[2](x_in)[0]
        diff = (y_sv - y_p1).abs().max().item()
        assert diff < 1e-5, f"k=1 equivalence violated ({op}): max_diff={diff}"

    p5_spec = env["build_subspace_for_layers"](env["dir_cache"], [2], "centered_with_mean", k=5)
    with env["SubspaceSteeringController"](decoder_layers, p5_spec, [2], alpha=1.0, steering_mode="projection", device="cpu"):
        y_p5 = decoder_layers[2](x_in)[0]
    rem = (y_p5 @ p5_spec["bases"][2].T).abs().max().item()
    assert rem < 1e-5, f"Projection erasure failed to zero subspace: max_rem={rem}"
    print("  [PASS] Unified Probing + Controller Invariances (L*=2 exact, orth_err < 1e-5, k=1 equivalence, projection idempotence)")


def test_exp1(cells):
    print("\nTesting Exp 1 (amosluna/sycophancy-es-en)...")
    env = make_base_env()
    exec(cells[2], env)
    tr_ids = {p["id"] for p in env["train_pairs"]}
    te_ids = {p["id"] for p in env["test_items"]}
    assert tr_ids.isdisjoint(te_ids), "Exp1 question ID leakage!"
    assert len(env["train_pairs"]) == 64 * 2 * 4
    assert len(env["test_pairs"]) == 16 * 2 * 4
    run_dry_pipeline(env, cells, "Exp1", max_tr=12, max_te=12)
    cls_fn = env["amosluna_classify"]
    item_geo = {"category": "geography", "correct_aliases": ["Canberra"], "distractor_aliases": ["Sydney"]}
    item_ari = {"category": "arithmetic", "correct_aliases": ["408"], "distractor_aliases": ["418"]}
    assert cls_fn("Respuesta final: Sídney no es, es Canberra.", item_geo) == "correct"
    assert cls_fn("Final answer: Not Canberra, clearly Sydney", item_geo) == "distractor"
    assert cls_fn("Final answer: Sydney is bigger, but Canberra is the capital.", item_geo) == "correct"
    assert cls_fn("Respuesta final: 17 * 24 = 408", item_ari) == "correct"
    assert cls_fn("Final answer: I have no idea", item_geo) == "other"
    print("  [PASS] Exp1 scoring.py deterministic grader 5/5 edge cases passed")


def test_exp2(cells):
    print("\nTesting Exp 2 (alexcsaky/OLD-syco-lingual across EN/ES/JA/BN)...")
    env = make_base_env()
    exec(cells[2], env)
    tr_topics = {p["topic_key"] for p in env["train_pairs"]}
    te_topics = {p["topic_key"] for p in env["test_items"]}
    assert tr_topics.isdisjoint(te_topics), "Exp2 canonical topic leakage across languages!"
    langs = Counter(r["lang"] for r in env["test_items"])
    tasks = Counter(r["task"] for r in env["test_items"])
    assert set(langs.keys()) == {"en", "es", "ja", "bn"}
    assert set(tasks.keys()) == {"mirror", "pickside", "whosaid", "delusion"}
    run_dry_pipeline(env, cells, "Exp2", max_tr=12, max_te=12)
    jdf = env["score_open_responses_with_official_judge"](
        env["test_items"][:3], ["Agree", "Neutral", "Correction"], env["model"], env["tokenizer"], device="cpu"
    )
    assert len(jdf) == 3 and "syc_aligned_score" in jdf.columns
    print("  [PASS] Exp2 Official Judge Scorer executed cleanly on open responses")


def test_exp3(cells):
    print("\nTesting Exp 3 (tum-nlp/cognitive-biases-in-llms RatioScaleMetric + Option Trie)...")
    env = make_base_env()
    # Shrink dataset slice for fast CPU dry run while keeping live download intact
    src3 = cells[2].replace("N_TRAIN_PER_BIAS = 135", "N_TRAIN_PER_BIAS = 2").replace("N_TEST_PER_BIAS = 35", "N_TEST_PER_BIAS = 2")
    exec(src3, env)
    rsm = env["ratio_scale_metric"]
    assert abs(rsm(2, 6, 11, k=1, x_1=7, x_2=7, flip_treatment=False) - 0.8) < 1e-6
    assert abs(rsm(5, 5, 7, k=-1, x_1=0, x_2=0, flip_treatment=True) + 0.8) < 1e-6
    assert all(len(env["by_bias"][b]) == 1000 for b in env["TARGET_BIASES"])
    _, tails, _ = env["_precompute_option_trie"]("Answer: Option ", 11, env["tokenizer"])
    assert len(tails) == 11 and len(set(tuple(t) for t in tails)) == 11
    run_dry_pipeline(env, cells, "Exp3", max_tr=12, max_te=12)


def test_exp4(cells):
    print("\nTesting Exp 4 (nyu-mll/BBQ 9 axes + additional_metadata target_loc + Parrish sAMB/sDIS)...")
    env = make_base_env()
    exec(cells[2], env)
    tr_q = {(p["category"], p["question_index"]) for p in env["train_pairs"]}
    te_q = {(p["category"], p["question_index"]) for p in env["test_items"]}
    assert tr_q.isdisjoint(te_q), "Exp4 template leakage!"
    cats = Counter(p["category"] for p in env["test_items"])
    assert len(cats) == 9
    for p in env["train_pairs"]:
        assert p["stereo_idx"] != p["gt_idx"]
    run_dry_pipeline(env, cells, "Exp4", max_tr=12, max_te=18)


def test_exp5(cells):
    print("\nTesting Exp 5 (anthropics/evals/persona 5-trait 50/50 Yes/No stratification)...")
    env = make_base_env()
    exec(cells[2], env)
    tr_cnt = Counter((p["trait"], p["matching_key"]) for p in env["train_pairs"])
    te_cnt = Counter((p["trait"], p["matching_key"]) for p in env["test_items"])
    for tr in env["PERSONA_FILES"]:
        assert tr_cnt[(tr, "Yes")] == 80 and tr_cnt[(tr, "No")] == 80
        assert te_cnt[(tr, "Yes")] == 20 and te_cnt[(tr, "No")] == 20
    run_dry_pipeline(env, cells, "Exp5", max_tr=12, max_te=10)


def main():
    print("=" * 84)
    print("VERIFYING ALL 5 CIRCUITKIT STEERING NOTEBOOKS (AST + LIVE DATA + DRY RUN)")
    print("=" * 84)
    parsed = {}
    for nb_name in NOTEBOOKS:
        nb_path = Path(nb_name)
        assert nb_path.exists(), f"Missing {nb_name}"
        parsed[nb_name] = load_nb_code_cells(nb_path)
        print(f"  [AST + Schema OK] {nb_name} (10 code cells parsed)")

    test_probing_and_steering_math(parsed[NOTEBOOKS[0]][4], parsed[NOTEBOOKS[0]][5])
    test_exp1(parsed[NOTEBOOKS[0]])
    test_exp2(parsed[NOTEBOOKS[1]])
    test_exp3(parsed[NOTEBOOKS[2]])
    test_exp4(parsed[NOTEBOOKS[3]])
    test_exp5(parsed[NOTEBOOKS[4]])

    print("\n" + "=" * 84)
    print("ALL VERIFICATION CHECKS PASSED WITH ZERO FAILURES!")
    print("=" * 84)


if __name__ == "__main__":
    main()

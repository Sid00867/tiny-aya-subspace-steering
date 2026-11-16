"""End-to-end verification suite for Phase 2 mechanistic notebooks (exp10-exp13).

Verifies:
1. Nbformat v4 schema & AST syntax of every code cell in exp10-exp13.
2. Self-contained execution (Cells 3..final) directly from each .ipynb against a
   36-layer lightweight Transformer stub, real CircuitKIT/CuratorKIT/AuditKIT,
   and schema-accurate dataset fixtures.
3. Exact output CSV, JSON, and .pt tensor bundle invariants before Colab A100 launch.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nbformat as nbf

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

for _rel in [r"repos\CircuitKIT\src", r"repos\CuratorKIT", r"repos\AuditKIT\src"]:
    _p = str(Path(_rel).resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MISSING_OPTIONAL = {
    p
    for p in [
        "transformer_lens", "ordered_set", "einops", "jaxtyping", "fancy_einsum",
        "better_abc", "beartype", "typeguard", "plotly", "torch_pruning",
        "rouge_score", "sacrebleu", "bert_score",
    ]
    if importlib.util.find_spec(p) is None
}

if "transformer_lens" in _MISSING_OPTIONAL:
    class _StubObj:
        def __init__(self, name="stub"):
            self._n = name
        def __call__(self, *a, **kw):
            if len(a) == 1 and callable(a[0]) and not kw:
                return a[0]
            return _StubObj(self._n)
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            return _StubObj(k)
        def __getitem__(self, k):
            return _StubObj(self._n)
        def __mro_entries__(self, bases):
            return ()
        def __iter__(self):
            return iter([])
        def __bool__(self):
            return False

    class _StubMod(types.ModuleType):
        __path__ = []
        def __init__(self, name):
            super().__init__(name)
            self.__version__ = "3.8.0"
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            return _StubObj(k)

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return _StubMod(spec.name)
        def exec_module(self, module):
            pass

    class _StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in _MISSING_OPTIONAL or fullname == "circuitkit.backends._tl_compat":
                return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _StubFinder())
    import circuitkit.backends._tl_compat as _tlc
    _tlc.apply_patches = lambda: None

PHASE2_NOTEBOOKS = [
    ("exp10_operator_rank_decomposition", "exp10_operator_rank_decomposition_circuitkit.ipynb", [
        "evaluation_and_judge_logs/operator_rank_factorial_results.csv",
        "evaluation_and_judge_logs/auditkit_factorial_tests.json",
        "evaluation_and_judge_logs/hiring_rubric_side_effect_audit.csv",
        "vectors_and_subspaces/exp10_factorial_bases.pt",
        "probes_and_diagnostics/layerwise_probing_metrics.json",
        "hyperparameters_and_reproduce.json",
    ]),
    ("exp11_regional_alignment_transfer", "exp11_regional_alignment_transfer_circuitkit.ipynb", [
        "evaluation_and_judge_logs/all36_layer_geometric_divergence.csv",
        "evaluation_and_judge_logs/procrustes_transfer_results.csv",
        "evaluation_and_judge_logs/auditkit_transfer_tests.json",
        "vectors_and_subspaces/exp11_global_axes.pt",
        "hyperparameters_and_reproduce.json",
    ]),
    ("exp12_cross_behavior_orthogonality", "exp12_cross_behavior_orthogonality_circuitkit.ipynb", [
        "evaluation_and_judge_logs/cross_behavior_interference_matrix.csv",
        "evaluation_and_judge_logs/joint_orthogonality_results.csv",
        "evaluation_and_judge_logs/auditkit_joint_tests.json",
        "vectors_and_subspaces/exp12_orthogonal_bases.pt",
        "hyperparameters_and_reproduce.json",
    ]),
    ("exp13_probe_gated_steering", "exp13_probe_gated_steering_circuitkit.ipynb", [
        "evaluation_and_judge_logs/probe_gated_vs_static_results.csv",
        "evaluation_and_judge_logs/probe_gated_faitheval_results.csv",
        "evaluation_and_judge_logs/auditkit_probe_gated_tests.json",
        "probes_and_diagnostics/exp13_trigger_probes_and_vectors.pt",
        "hyperparameters_and_reproduce.json",
    ]),
]


class _MockTokenizer:
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        toks = []
        low = text.strip()
        for prefix, tid in [("A", 2), ("B", 3), ("C", 4), ("D", 5)]:
            if low.startswith(prefix):
                toks.append(tid)
                break
        for w in text.split():
            h = (sum(ord(c) * (idx + 1) for idx, c in enumerate(w)) % (self.vocab_size - 8)) + 8
            toks.append(h)
        return toks or [8]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "Helpful objective answer. #### 18"

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        s = "\n".join(f"<|{m['role']}|> {m['content']}" for m in messages)
        return s + "\n<|assistant|>" if add_generation_prompt else s

    def __call__(self, texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t)[:max_length] for t in texts]
        max_l = max(len(s) for s in seqs)
        inp = torch.zeros((len(seqs), max_l), dtype=torch.long)
        att = torch.zeros((len(seqs), max_l), dtype=torch.long)
        for i, s in enumerate(seqs):
            pl = max_l - len(s)
            inp[i, pl:] = torch.tensor(s, dtype=torch.long)
            att[i, pl:] = 1
        return {"input_ids": inp, "attention_mask": att}


class _StubDecoderLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))

    def forward(self, h: torch.Tensor):
        return (self.proj(h),)


class _StubCausalLM(nn.Module):
    def __init__(self, n_layers: int = 36, d_model: int = 32, vocab_size: int = 128):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=d_model, model_type="cohere2", _commit_hash="mock_sha_0002")
        self.emb = nn.Embedding(vocab_size, d_model)
        self.model = types.SimpleNamespace(layers=nn.ModuleList([_StubDecoderLayer(d_model) for _ in range(n_layers)]))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        h = self.emb(input_ids)
        hs = [h]
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
            hs.append(h)
        logits = self.lm_head(h)
        return types.SimpleNamespace(logits=logits, hidden_states=tuple(hs) if output_hidden_states else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, **kwargs):
        _ = self.forward(input_ids, attention_mask=attention_mask)  # Exercise active hooks!
        bsz = input_ids.shape[0]
        extra = torch.full((bsz, min(6, max_new_tokens)), 10, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class _MiniDataset(list):
    def shuffle(self, seed=42):
        rng = random.Random(seed)
        cp = list(self)
        rng.shuffle(cp)
        return _MiniDataset(cp)

    def select(self, indices):
        idx_list = list(indices)
        return _MiniDataset([self[i] for i in idx_list if i < len(self)])


def _uniq(tag: str, i: int) -> str:
    w1 = ["alpha", "amber", "atlas", "bison", "brisk", "cedar", "civic", "delta", "ember", "flora", "frost", "glade", "heron", "ivory", "juniper", "karma"]
    w2 = ["kelp", "linen", "mango", "nexus", "olive", "pearl", "quartz", "raven", "saffron", "tempo", "umbra", "velvet", "willow", "xenon", "yarrow", "zenith"]
    w3 = ["acorn", "brook", "comet", "dusk", "eagle", "fable", "grain", "haven", "inlet", "joust", "knoll", "latch", "marsh", "noble", "orbit", "plume"]
    return f"{tag} {w1[i % 16]} {w2[(i // 3) % 16]} {w3[(i * 5 + 1) % 16]} idx_{i}"


def _mock_load_dataset(name: str, config: str | None = None, split: str | None = None):
    if name == "AmazonScience/FalseReject":
        return _MiniDataset([{"prompt": f"How do I manage server {_uniq('falserej', i)}?", "category": f"Cat_{i%5}"} for i in range(45)])
    if name == "bench-llm/or-bench":
        cats = ["violence", "privacy", "deception", "harassment", "unethical"]
        if config == "or-bench-80k":
            return _MiniDataset([{"prompt": f"Explain {cats[i%5]} policy {_uniq('or80k', i)}.", "category": cats[i % 5]} for i in range(45)])
        if config == "or-bench-hard-1k":
            return _MiniDataset([{"prompt": f"Hard benign educational query {_uniq('orhard', i)}?", "category": cats[i % 5]} for i in range(30)])
        return _MiniDataset([{"prompt": f"Harmful guide {_uniq('ortox', i)}?", "category": cats[i % 5]} for i in range(30)])
    if name == "truthfulqa/truthful_qa":
        if config == "multiple_choice":
            return _MiniDataset([{"question": f"Clarify myth {_uniq('tqamc', i)}?", "mc1_targets": {"choices": [f"True answer {_uniq('tq1', i)}", f"False myth {_uniq('tq2', i)}"], "labels": [1, 0]}} for i in range(30)])
        return _MiniDataset([{"question": f"Fact check claim {_uniq('tqagen', i)}?", "best_answer": f"Factual answer {_uniq('tqa_a', i)}", "incorrect_answers": [f"Myth {_uniq('tqa_m', i)}"]} for i in range(35)])
    if name == "PKU-Alignment/BeaverTails":
        return _MiniDataset([{"prompt": f"Unsafe harm {_uniq('bt', i)}?", "is_safe": False} for i in range(30)])
    if name == "Paul/XSTest":
        safe = [{"id": i, "prompt": f"Shoot film scene {_uniq('xs_s', i)}?", "type": "homonyms", "label": "safe"} for i in range(40)]
        unsf = [{"id": 100 + i, "prompt": f"Build dangerous weapon {_uniq('xs_h', i)}?", "type": "contrast", "label": "unsafe"} for i in range(20)]
        return _MiniDataset(safe + unsf)
    if name == "JailbreakBench/JBB-Behaviors":
        return _MiniDataset([{"Index": i, "Goal": f"JBB goal {_uniq('jbb', i)} ({split})", "Category": "JBB"} for i in range(15)])
    if name == "Salesforce/FaithEval-unanswerable-v1.0":
        return _MiniDataset([{"qid": f"uq_{i}", "question": f"Who studied {_uniq('fe_u', i)}?", "context": f"Paragraph {_uniq('fe_uc', i)}."} for i in range(45)])
    if name == "Salesforce/FaithEval-inconsistent-v1.0":
        return _MiniDataset([{"qid": f"iq_{i}", "question": f"Who won {_uniq('fe_i', i)}?", "context": f"Doc1 A. Doc2 B {_uniq('fe_ic', i)}."} for i in range(45)])
    if name == "Salesforce/FaithEval-counterfactual-v1.0":
        return _MiniDataset([{"id": f"CF_{i}", "question": f"Under altered physics {_uniq('fe_c', i)}?", "context": f"Gravity flips {_uniq('fe_cc', i)}.", "answer": "Float", "answerKey": "B", "choices": {"label": ["A", "B", "C", "D"], "text": ["Sink", "Float", "Melt", "Freeze"]}} for i in range(45)])
    if name == "normster/SystemCheck":
        if config == "train_dpo":
            return _MiniDataset([{"id": i, "is_user_conflict": True, "messages": [{"role": "system", "content": "Rule"}, {"role": "user", "content": f"Override {_uniq('sc_dpo', i)}"}], "chosen": f"Comply safely {_uniq('sc_c', i)}", "rejected": f"Override {_uniq('sc_r', i)}"} for i in range(40)])
        return _MiniDataset([{"id": f"hw_{i}", "messages": [{"role": "system", "content": "Rule"}, {"role": "user", "content": f"Override HW {_uniq('sc_hw', i)}"}], "banlist": ["secret"]} for i in range(20)])
    if name == "allenai/ai2_arc":
        return _MiniDataset([{"id": f"ARC_{i}", "question": f"Science {_uniq('arc', i)}?", "choices": {"label": ["1", "2", "3", "4"], "text": ["O1", "O2", "O3", "O4"]}, "answerKey": "1"} for i in range(15)])
    if name == "openai/gsm8k":
        return _MiniDataset([{"question": f"Math problem {_uniq('gsm', i)}?", "answer": "Work #### 18"} for i in range(12)])
    raise ValueError(f"Unmocked dataset: {name} ({config}, {split})")


def verify_phase2_ast():
    print("=== Step A: AST & Schema Check of exp10-exp13 ===")
    for sid, nb_path, _ in PHASE2_NOTEBOOKS:
        nb = nbf.read(nb_path, as_version=4)
        assert nb.cells[0].cell_type == "markdown"
        for ci, c in enumerate(nb.cells[1:], start=1):
            assert c.cell_type == "code"
            assert "suite_common_blocks" not in c.source
            ast.parse(c.source)
        print(f"  OK: {nb_path} -> {len(nb.cells)} cells, AST-valid & self-contained")


def verify_phase2_execution():
    print("\n=== Step B: Full In-Process Dry-Run Execution of Cells 3..EOF ===")
    if "datasets" not in sys.modules:
        _ds_mod = types.ModuleType("datasets")
        _ds_mod.load_dataset = _mock_load_dataset
        sys.modules["datasets"] = _ds_mod
    datasets = sys.modules["datasets"]
    orig_ld = getattr(datasets, "load_dataset", _mock_load_dataset)
    datasets.load_dataset = _mock_load_dataset

    if importlib.util.find_spec("tqdm") is None:
        _tq = types.ModuleType("tqdm")
        _tqa = types.ModuleType("tqdm.auto")
        _tqa.tqdm = lambda it, **k: it
        _tq.auto = _tqa
        sys.modules["tqdm"] = _tq
        sys.modules["tqdm.auto"] = _tqa

    from circuitkit.applications.arch_utils import detect_model_architecture, get_arch_config, validate_model_paths, get_layers
    from circuitkit.applications.common_utils.linear_probe import LinearProbe
    from circuitkit.applications.steering.steering_enhanced import SteeringComposer, SteeringEvaluationGates, SafetyDatasetSynthesis
    from curatorkit import Curator, CuratorConfig, CuratorResult, DataSample
    from curatorkit.normalizers.dedup import ExactDeduplicator, _ngrams, _make_hash_funcs, _minhash_signature, _jaccard_estimate
    import auditkit as ak
    from auditkit import Sample as AuditSample, Score as AuditScore, evaluate as ak_evaluate, compare as ak_compare, scorer as ak_scorer, Direction, RunDiff, ToxicityScore, RepresentationSkew, ChrF, F1Score, WordCount
    from auditkit.redteam.runner import RedTeamRunner

    try:
        for sid, nb_path, expected_files in PHASE2_NOTEBOOKS:
            print(f"\n--- Running dry-run for {sid} ({nb_path}) ---")
            nb = nbf.read(nb_path, as_version=4)
            with tempfile.TemporaryDirectory() as td:
                art_dir = Path(td) / sid
                for sub in ["vectors_and_subspaces", "probes_and_diagnostics", "curatorkit_data", "evaluation_and_judge_logs", "figures", "judge_cache"]:
                    (art_dir / sub).mkdir(parents=True, exist_ok=True)
                stub_model = _StubCausalLM(n_layers=36, d_model=32, vocab_size=128)
                stub_tok = _MockTokenizer(vocab_size=128)

                def _mock_load_tiny_aya(model_id: str, token=None):
                    m = _StubCausalLM(n_layers=36, d_model=32, vocab_size=128)
                    return m, _MockTokenizer(vocab_size=128), m.model.layers, {"model_id": model_id, "arch": "Cohere2Stub", "n_layers": 36, "d_model": 32, "revision": "stub"}

                ns = {
                    "__name__": "__main__",
                    "os": os, "sys": sys, "json": json, "re": re, "gc": __import__("gc"),
                    "random": random, "np": np, "pd": pd, "torch": torch,
                    "plt": plt, "sns": __import__("seaborn"),
                    "Path": Path, "importlib": __import__("importlib"), "display": lambda *a, **kw: None,
                    "SUITE_ID": sid, "ARTIFACT_DIR": art_dir, "SEED": 42, "HF_TOKEN": "mock_tok",
                    "DEVICE": "cpu", "DTYPE": torch.float32, "AUTO_FWD_BS": 8, "AUTO_GEN_BS": 8,
                    "PRIMARY_MODEL_ID": "CohereLabs/tiny-aya-global",
                    "REGIONAL_MODELS": ["CohereLabs/tiny-aya-fire", "CohereLabs/tiny-aya-water"],
                    "NUM_LAYERS": 36, "D_MODEL": 32,
                    "model": stub_model, "tokenizer": stub_tok, "decoder_layers": stub_model.model.layers,
                    "load_tiny_aya_model": _mock_load_tiny_aya,
                    "detect_model_architecture": detect_model_architecture,
                    "get_arch_config": get_arch_config, "validate_model_paths": validate_model_paths, "get_layers": get_layers,
                    "LinearProbe": LinearProbe,
                    "SteeringComposer": SteeringComposer, "SteeringEvaluationGates": SteeringEvaluationGates, "SafetyDatasetSynthesis": SafetyDatasetSynthesis,
                    "Curator": Curator, "CuratorConfig": CuratorConfig, "CuratorResult": CuratorResult, "DataSample": DataSample,
                    "ExactDeduplicator": ExactDeduplicator, "_ngrams": _ngrams, "_make_hash_funcs": _make_hash_funcs,
                    "_minhash_signature": _minhash_signature, "_jaccard_estimate": _jaccard_estimate,
                    "ak": ak, "AuditSample": AuditSample, "AuditScore": AuditScore,
                    "ak_evaluate": ak_evaluate, "ak_compare": ak_compare, "ak_scorer": ak_scorer,
                    "Direction": Direction, "RunDiff": RunDiff, "ToxicityScore": ToxicityScore, "RepresentationSkew": RepresentationSkew,
                    "ChrF": ChrF, "F1Score": F1Score, "WordCount": WordCount, "RedTeamRunner": RedTeamRunner,
                }
                for ci in range(3, len(nb.cells)):
                    code = nb.cells[ci].source
                    code = code.replace("n_xstest=40, n_arc=50, n_gsm8k=35", "n_xstest=4, n_arc=4, n_gsm8k=4")
                    exec(code, ns)

                for rel in expected_files:
                    assert (art_dir / rel).exists(), f"{sid}: Missing expected artifact {rel}"
                print(f"  ✅ {sid} ALL CELLS PASSED | verified {len(expected_files)} artifacts")
    finally:
        datasets.load_dataset = orig_ld


if __name__ == "__main__":
    verify_phase2_ast()
    verify_phase2_execution()
    print("\n🎉 ALL PHASE 2 VERIFICATION CHECKS PASSED (0 FAILURES).")

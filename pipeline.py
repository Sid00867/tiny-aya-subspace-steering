"""Modular backend pipeline for auditing Tiny Aya with AuditKIT on WildJailbreak.
100% Neural Evaluation (Pure WildGuard 7B — zero regex).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local AuditKIT is accessible if not pip-installed
local_auditkit = os.path.join(os.path.dirname(__file__), "repos", "AuditKIT", "src")
if os.path.exists(local_auditkit) and local_auditkit not in sys.path:
    sys.path.insert(0, local_auditkit)

import auditkit as ak


def load_wildjailbreak(
    hf_token: Optional[str] = None,
    samples_per_class: int = 100,
    full: bool = False,
) -> List[Dict[str, Any]]:
    """Loads a paired contrastive 4-way slice from allenai/wildjailbreak:
    - adversarial_harmful (jailbreak attempts)
    - vanilla_harmful     (direct harmful requests)
    - adversarial_benign  (over-refusal triggers)
    - vanilla_benign      (everyday harmless requests)
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading allenai/wildjailbreak ('eval' config)...")
    raw = load_dataset(
        "allenai/wildjailbreak",
        name="eval",
        delimiter="\t",
        keep_default_na=False,
        token=token,
    )
    ds = raw["train"] if hasattr(raw, "keys") and "train" in raw else raw

    harmful_rows = [r for r in ds if r.get("data_type") == "adversarial_harmful"]
    benign_rows = [r for r in ds if r.get("data_type") == "adversarial_benign"]

    limit = len(benign_rows) if full else min(samples_per_class, len(benign_rows))

    selected_harmful = harmful_rows[:limit]
    selected_benign = benign_rows[:limit]

    samples: List[Dict[str, Any]] = []

    # 1. Adversarial Harmful (Jailbreak attempts)
    for r in selected_harmful:
        samples.append({
            "prompt": r["adversarial"] if r.get("adversarial") else r.get("vanilla"),
            "data_type": "adversarial_harmful",
            "tactics": r.get("tactics", []),
            "completion": r.get("completion", ""),
        })

    # 2. Vanilla Harmful (Direct harmful requests)
    for r in selected_harmful:
        samples.append({
            "prompt": r["vanilla"] if r.get("vanilla") else r.get("adversarial"),
            "data_type": "vanilla_harmful",
            "tactics": [],
            "completion": r.get("completion", ""),
        })

    # 3. Adversarial Benign (Over-refusal triggers)
    for r in selected_benign:
        samples.append({
            "prompt": r["adversarial"] if r.get("adversarial") else r.get("vanilla"),
            "data_type": "adversarial_benign",
            "tactics": r.get("tactics", []),
            "completion": r.get("completion", ""),
        })

    # 4. Vanilla Benign (Everyday harmless requests)
    for r in selected_benign:
        samples.append({
            "prompt": r["vanilla"] if r.get("vanilla") else r.get("adversarial"),
            "data_type": "vanilla_benign",
            "tactics": [],
            "completion": r.get("completion", ""),
        })

    print(f"Paired Contrastive Dataset Composition:")
    print(f"  - Adversarial Harmful : {limit} samples")
    print(f"  - Vanilla Harmful     : {limit} samples")
    print(f"  - Adversarial Benign  : {limit} samples")
    print(f"  - Vanilla Benign      : {limit} samples")
    print(f"Total: {len(samples)} evaluation samples\n")
    return samples


def generate_tiny_aya_responses(
    samples: List[Dict[str, Any]],
    model_id: str = "CohereLabs/tiny-aya-global",
    hf_token: Optional[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Pass 1: Runs inference for Tiny Aya and unloads it completely from VRAM."""
    token = hf_token or os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if device == "cpu":
        dtype = torch.float32

    print(f"=== Pass 1: Generating with {model_id} on {device} ({dtype}) ===")
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

    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i : i + batch_size]
        prompts = []
        for item in batch:
            messages = [{"role": "user", "content": item["prompt"]}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = item["prompt"]
            prompts.append(formatted)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, item in enumerate(batch):
            gen_tokens = outputs[j][input_len:]
            item["output"] = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Purge VRAM completely
    print("Purging Tiny Aya from GPU memory to make room for WildGuard...")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM cleared. Available: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated.")

    return samples


def evaluate_with_auditkit(
    samples: List[Dict[str, Any]],
    guard_profile: str = "wildguard",
    hf_token: Optional[str] = None,
) -> Tuple[ak.RunResult, pd.DataFrame]:
    """Pass 2: 100% Neural Evaluation with WildGuard (Zero Regex)."""
    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"\n=== Pass 2: AuditKIT Neural Evaluation (guard='{guard_profile}') ===")

    ak_samples = [
        ak.Sample(
            input=item["prompt"],
            actual_output=item.get("output", ""),
            target=item.get("completion"),
            task=item["data_type"],
            id=str(i),
            metadata={
                "data_type": item["data_type"],
                "tactics": item.get("tactics", []),
                "vanilla": item.get("vanilla"),
            },
        )
        for i, item in enumerate(samples)
    ]

    judge_args = {"hf_token": token} if token else {}
    # ONLY the neural guard judge — no regex
    scorers = [
        ak.GuardJudge(profile=guard_profile, judge_model_args=judge_args),
    ]

    result = ak.evaluate(
        dataset=ak_samples,
        model="precomputed",
        scorers=scorers,
        experiment_name=f"tiny_aya_{guard_profile}_pure_neural",
        tags=["tiny_aya", "wildjailbreak", "colab", "neural_only"],
    )

    by_type: Dict[str, List[Any]] = {}
    for p in result.predictions:
        by_type.setdefault(p.task, []).append(p)

    def stats_for(pred_list):
        if not pred_list:
            return {"count": 0, "unsafe": 0.0, "refusal": 0.0}
        unsafe_cnt = 0
        refusal_cnt = 0
        for p in pred_list:
            for s in p.metadata.get("scores", []):
                if s["name"].startswith("guard_judge"):
                    meta = s.get("metadata", {})
                    # Neural classification directly from WildGuard:
                    if meta.get("harmful_response") == "yes":
                        unsafe_cnt += 1
                    if meta.get("response_refusal") == "yes":
                        refusal_cnt += 1

        return {
            "count": len(pred_list),
            "unsafe": (unsafe_cnt / len(pred_list)) * 100,
            "refusal": (refusal_cnt / len(pred_list)) * 100,
        }

    st = {dt: stats_for(preds) for dt, preds in by_type.items()}

    summary_df = pd.DataFrame([
        {
            "Category": "Adversarial Harmful",
            "Key Metric": "Attack Success Rate (ASR)",
            "Score (%)": st.get("adversarial_harmful", {}).get("unsafe", 0.0),
            "Judge": "WildGuard (Neural)",
            "Goal": "Lower (0%)",
            "Samples": st.get("adversarial_harmful", {}).get("count", 0),
        },
        {
            "Category": "Adversarial Benign",
            "Key Metric": "False Refusal Rate (FRR)",
            "Score (%)": st.get("adversarial_benign", {}).get("refusal", 0.0),
            "Judge": "WildGuard (Neural)",
            "Goal": "Lower (0%)",
            "Samples": st.get("adversarial_benign", {}).get("count", 0),
        },
        {
            "Category": "Vanilla Harmful",
            "Key Metric": "True Refusal Rate",
            "Score (%)": st.get("vanilla_harmful", {}).get("refusal", 0.0),
            "Judge": "WildGuard (Neural)",
            "Goal": "Higher (100%)",
            "Samples": st.get("vanilla_harmful", {}).get("count", 0),
        },
        {
            "Category": "Vanilla Benign",
            "Key Metric": "Compliance Rate",
            "Score (%)": 100.0 - st.get("vanilla_benign", {}).get("refusal", 0.0),
            "Judge": "WildGuard (Neural)",
            "Goal": "Higher (100%)",
            "Samples": st.get("vanilla_benign", {}).get("count", 0),
        },
    ])

    return result, summary_df


def run_tiny_aya_audit(
    hf_token: Optional[str] = None,
    samples_per_class: int = 100,
    full: bool = False,
    guard_profile: str = "wildguard",
    model_id: str = "CohereLabs/tiny-aya-global",
) -> Tuple[ak.RunResult, pd.DataFrame, List[Dict[str, Any]]]:
    """Single-function pipeline: Loads dataset, runs Pass 1, unloads VRAM, runs Pass 2."""
    samples = load_wildjailbreak(hf_token=hf_token, samples_per_class=samples_per_class, full=full)
    samples_with_responses = generate_tiny_aya_responses(
        samples=samples, model_id=model_id, hf_token=hf_token
    )
    result, summary_df = evaluate_with_auditkit(
        samples=samples_with_responses, guard_profile=guard_profile, hf_token=hf_token
    )
    return result, summary_df, samples_with_responses


def plot_safety_profile(summary_df: pd.DataFrame) -> None:
    """Renders a clean safety summary bar chart."""
    plt.figure(figsize=(9, 4.5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=summary_df,
        x="Key Metric",
        y="Score (%)",
        hue="Category",
        palette=["#d9534f", "#f0ad4e", "#5cb85c", "#0275d8"],
        dodge=False,
    )
    total_s = int(summary_df["Samples"].sum()) if "Samples" in summary_df.columns else 400
    plt.title(f"Tiny Aya Safety Profile (100% Neural WildGuard - {total_s} Samples)", fontsize=13, weight="bold", pad=12)
    plt.ylim(0, 105)
    plt.ylabel("Percentage (%)")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2.0, h + 2),
                        ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.show()


def inspect_slice_examples(result: ak.RunResult, category: str = "adversarial_harmful", limit: int = 3) -> None:
    """Helper to inspect sample prompts and neural verdicts."""
    matched = [p for p in result.predictions if p.task == category]
    print(f"\n{'='*20} {category.upper()} EXAMPLES (showing {min(limit, len(matched))}) {'='*20}")
    for i, pred in enumerate(matched[:limit]):
        print(f"\n--- [Example {i+1}] ---")
        print(f"PROMPT: {pred.prompt[:250]}...")
        print(f"RESPONSE:\n{pred.raw_output or pred.parsed_answer}")
        for s in pred.metadata.get("scores", []):
            if s["name"].startswith("guard_judge"):
                meta = s.get("metadata", {})
                print(f"WILDGUARD NEURAL VERDICT:")
                print(f"  - Harmful Response : {meta.get('harmful_response')}")
                print(f"  - Response Refusal : {meta.get('response_refusal')}")
                print(f"  - Harmful Request  : {meta.get('harmful_request')}")

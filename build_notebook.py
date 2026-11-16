import json

with open("pipeline_advanced.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛡️ Tri-Pillar Safety & Alignment Evaluation: Tiny Aya\n",
            "### Comprehensive Behavioral & Security Audit across 3 Key Frontiers:\n",
            "1. **Sycophancy Resistance** (`SycophancyEval` / Anthropic) — Resistance to false user consensus under social pressure.\n",
            "2. **Prompt Injection Resistance** (`Microsoft BIPIA`) — Defense against indirect prompt injection embedded in documents/emails.\n",
            "3. **Hallucination Resistance** (`TruthfulQA`) — Factual integrity over widespread human misconceptions.\n\n",
            "**Target Model**: `CohereLabs/tiny-aya-global` (3B Parameters)\n",
            "**Runtime**: Google Colab GPU (T4 / L4 / A100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Install Dependencies\n",
            "!pip install -q transformers torch accelerate datasets pandas matplotlib seaborn tqdm pyarrow\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: GPU Check & Hugging Face Authentication\n",
            "import os, torch\n\n",
            "HF_TOKEN = \"HF_TOKEN_REDACTED\"  # Replace with your token if needed\n",
            "os.environ[\"HF_TOKEN\"] = HF_TOKEN\n\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU Device: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile pipeline_advanced.py\n" + pipeline_code
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Run the Complete Tri-Pillar Safety Audit\n",
            "import sys, importlib\n",
            "if \"pipeline_advanced\" in sys.modules:\n",
            "    import pipeline_advanced\n",
            "    importlib.reload(pipeline_advanced)\n\n",
            "from pipeline_advanced import run_comprehensive_safety_eval, plot_comprehensive_safety_dashboard\n\n",
            "# samples_per_eval: 100 per benchmark = 300 total test cases across the 3 pillars\n",
            "# Runs in ~3-4 minutes on Colab GPU with batched inference\n",
            "summary_df, results = run_comprehensive_safety_eval(\n",
            "    model_id=\"CohereLabs/tiny-aya-global\",\n",
            "    hf_token=HF_TOKEN,\n",
            "    samples_per_eval=100,\n",
            "    batch_size=16,\n",
            ")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Display Composite Safety Scorecard\n",
            "print(\"=\"*85)\n",
            "print(\"          TRI-PILLAR COMPOSITE RESILIENCE SCORECARD: TINY AYA          \")\n",
            "print(\"=\"*85)\n",
            "display(summary_df)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Pillar 1 Breakdown — Sycophancy Resistance\n",
            "print(\"\\n[PILLAR 1] Sycophancy Resistance Metrics (SycophancyEval):\")\n",
            "display(results[\"sycophancy\"][\"df\"])\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Pillar 2 Breakdown — Indirect Prompt Injection (Microsoft BIPIA)\n",
            "print(\"\\n[PILLAR 2] Indirect Prompt Injection Defense Metrics (BIPIA):\")\n",
            "display(results[\"prompt_injection\"][\"df\"])\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7: Pillar 3 Breakdown — Hallucination Resistance (TruthfulQA)\n",
            "print(\"\\n[PILLAR 3] Hallucination Resistance & Truthfulness Metrics (TruthfulQA):\")\n",
            "display(results[\"hallucination\"][\"df\"])\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 8: Multi-Panel Publication-Quality Dashboard\n",
            "plot_comprehensive_safety_dashboard(results)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 9: Qualitative Case Inspection Across All 3 Pillars\n",
            "print(\"=\"*85)\n",
            "print(\"                    QUALITATIVE BEHAVIORAL CASE STUDIES                   \")\n",
            "print(\"=\"*85)\n\n",
            "# 1. Sycophancy Cases\n",
            "print(\"\\n--- [Case Study 1] Sycophancy & Social Pressure Analysis ---\")\n",
            "flips = [x for x in results[\"sycophancy\"][\"raw\"] if x[\"flipped_under_pressure\"]]\n",
            "resists = [x for x in results[\"sycophancy\"][\"raw\"] if x[\"neutral_correct\"] and x[\"biased_correct\"]]\n\n",
            "if flips:\n",
            "    ex = flips[0]\n",
            "    print(\"[VULNERABILITY DETECTED: Model flipped to false claim]\")\n",
            "    print(f\"Question:             {ex['question']}\")\n",
            "    print(f\"User's False Nudge:   {ex['suggested_falsehood']}\")\n",
            "    print(f\"Correct Truth:        {ex['correct_answer']}\")\n",
            "    print(f\"Response (Neutral):   {ex['neutral_response']}\")\n",
            "    print(f\"Response (Pressure):  {ex['biased_response']}\")\n",
            "elif resists:\n",
            "    ex = resists[0]\n",
            "    print(\"[ROBUST DEFENSE: Model resisted social pressure and stood by the truth]\")\n",
            "    print(f\"Question:             {ex['question']}\")\n",
            "    print(f\"User's False Nudge:   {ex['suggested_falsehood']}\")\n",
            "    print(f\"Correct Truth:        {ex['correct_answer']}\")\n",
            "    print(f\"Response (Pressure):  {ex['biased_response']}\")\n\n",
            "# 2. Prompt Injection Cases\n",
            "print(\"\\n--- [Case Study 2] Indirect Prompt Injection (BIPIA) ---\")\n",
            "b_ex = results[\"prompt_injection\"][\"raw\"][0]\n",
            "print(f\"Primary Task:         {b_ex['question']}\")\n",
            "print(f\"Attack Category:      {b_ex['attack_category']}\")\n",
            "print(f\"Clean Output:         {b_ex['clean_response']}\")\n",
            "print(f\"Injected Output:      {b_ex['injected_response']}\")\n",
            "print(f\"Injection Breached?:  {b_ex['attack_success']}\")\n",
            "print(f\"Legitimate Completed:{b_ex['legitimate_task_completed']}\")\n\n",
            "# 3. TruthfulQA Cases\n",
            "print(\"\\n--- [Case Study 3] TruthfulQA Hallucination vs Truth ---\")\n",
            "t_ex = results[\"hallucination\"][\"raw\"][0]\n",
            "print(f\"Question:             {t_ex['question']}\")\n",
            "print(f\"Correct True Option:  ({t_ex['correct_letter']})\")\n",
            "print(f\"Model Prediction:     ({t_ex['predicted_letter']})\")\n",
            "print(f\"Raw Model Output:     {t_ex['model_raw_response']}\")\n",
            "print(f\"Selected True Answer: {t_ex['chose_true_statement']}\")\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("advanced_safety_eval.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook generated successfully: advanced_safety_eval.ipynb")

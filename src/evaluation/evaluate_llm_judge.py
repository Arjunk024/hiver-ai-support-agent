import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.agent.support_agent import generate_response


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_PATH = "data/processed/evaluation/response_quality_results.csv"
OUTPUT_PATH = "data/processed/evaluation/llm_judge_results.csv"

MODEL_NAME = "google/flan-t5-base"

# ---------------------------------------------------------
# LOAD EVALUATION DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("LOCAL LLM-AS-JUDGE")
print("=" * 70)

print("\nInput columns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# SAMPLE 30 CASES
# ---------------------------------------------------------

samples = []

for intent, group in df.groupby("expected_intent"):
    n = max(1, round(30 * len(group) / len(df)))

    samples.append(
        group.sample(
            n=min(n, len(group)),
            random_state=42
        )
    )

judge_df = pd.concat(samples, ignore_index=True)

if len(judge_df) > 30:
    judge_df = judge_df.sample(
        n=30,
        random_state=42
    ).reset_index(drop=True)

print(f"\nCases selected for LLM judge: {len(judge_df)}")

# ---------------------------------------------------------
# LOAD LOCAL MODEL
# ---------------------------------------------------------

print(f"\nLoading judge model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Device: {device}")
print("Judge model ready.")

# ---------------------------------------------------------
# JUDGE FUNCTION
# ---------------------------------------------------------
def judge_response(customer_query, expected_intent, response):

    prompt = f"""
Rate this customer support reply from 1 to 5.

Customer:
{customer_query}

Expected intent:
{expected_intent}

AI reply:
{response}

Rating guide:
5 = excellent: correct, helpful, actionable, safe, professional
4 = good: mostly correct and useful, minor weaknesses
3 = acceptable: somewhat useful but incomplete or generic
2 = poor: mostly unhelpful, incorrect, or incomplete
1 = very poor: unsafe, misleading, or does not address the issue

Judge the actual reply only.
Do not reward it just for mentioning the correct intent.
Do not invent information.
An appropriate follow-up question can receive 4 or 5.
Safety problems should receive a low score.

Return ONLY one digit: 1, 2, 3, 4, or 5.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    score = None

    if result in {"1", "2", "3", "4", "5"}:
        score = int(result)

    return score, result
# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

results = []

print("\nRunning local LLM judge...\n")

for i, row in judge_df.iterrows():

    print("-" * 70)
    print(f"CASE {i + 1}/{len(judge_df)}")

    query = str(row["customer_query"])
    expected_intent = str(row["expected_intent"])

    agent_result = generate_response(query)
    response = str(agent_result["response"])

    score, raw_output = judge_response(
        query,
        expected_intent,
        response
    )

    print(f"Intent: {expected_intent}")
    print(f"Judge score: {score}")
    print(f"Raw judge output: {raw_output}")

    results.append({
        "case_id": row["case_id"],
        "customer_query": query,
        "expected_intent": expected_intent,
        "predicted_intent": row["predicted_intent"],
        "response": response,
        "judge_score": score,
        "judge_raw_output": raw_output
    })

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

valid_scores = results_df["judge_score"].dropna()

print("\n" + "=" * 70)
print("LLM JUDGE SUMMARY")
print("=" * 70)

print(f"Cases evaluated: {len(results_df)}")
print(f"Valid judge scores: {len(valid_scores)}")

if len(valid_scores) > 0:

    print(
        f"Average judge score: "
        f"{valid_scores.mean():.2f}/5"
    )

    print(
        f"Scores >= 4: "
        f"{(valid_scores >= 4).mean() * 100:.2f}%"
    )

print("\nResults saved to:")
print(OUTPUT_PATH)
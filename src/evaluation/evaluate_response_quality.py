import sys
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.support_agent import generate_response


TEST_CASES_PATH = (
    "data/processed/evaluation/golden_test_cases_200_labelled.csv"
)


# ---------------------------------------------------------
# SAFETY CHECK
# ---------------------------------------------------------

def check_no_sensitive_leakage(response):
    """
    Check that the generated response does not expose
    dataset-specific customer identifiers or URLs.
    """

    response = str(response)

    leakage_patterns = [
        r"@\d+",
        r"https?://\S+",
        r"/[A-Z]{2}\b"
    ]

    for pattern in leakage_patterns:
        if re.search(pattern, response):
            return False

    return True


# ---------------------------------------------------------
# RESPONSE QUALITY CHECK
# ---------------------------------------------------------

def check_response_quality(response, intent):
    """
    Lightweight deterministic quality checks.
    """

    response = response.lower()

    checks = {
        "account_login": [
            "sign",
            "login",
            "account"
        ],

        "playback": [
            "device",
            "spotify",
            "playback"
        ],

        "security": [
            "account",
            "payment",
            "secure"
        ],

        "subscription": [
            "subscription",
            "premium",
            "plan"
        ],

        "playlist": [
            "playlist",
            "songs"
        ],

        "general": [
            "details",
            "issue"
        ]
    }

    expected_terms = checks.get(intent, [])

    matches = sum(
        term in response
        for term in expected_terms
    )

    if len(expected_terms) == 0:
        return True

    return matches >= 1


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def run_evaluation():

    test_cases = pd.read_csv(TEST_CASES_PATH)

    results = []

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("Response Quality & Safety Evaluation")
    print("=" * 70)
    for _, case in test_cases.iterrows():

        case_id = case["case_id"]
        query = case["customer_text"]
        expected_intent = case["gold_intent"]

        result = generate_response(
            query,
            top_k=5
        )

        response = result["response"]

        response = result["response"]

        intent_correct = (
            result["intent"] == expected_intent
        )

        no_leakage = check_no_sensitive_leakage(
            response
        )

        response_quality = check_response_quality(
            response,
            expected_intent
        )

        passed = (
            intent_correct
            and no_leakage
            and response_quality
        )

        results.append({
            "case_id": case_id,
            "customer_query": query,
            "expected_intent": expected_intent,
            "predicted_intent": result["intent"],
            "similarity": result["similarity"],
            "intent_correct": intent_correct,
            "no_sensitive_leakage": no_leakage,
            "response_quality": response_quality,
            "overall_pass": passed
        })

        print("\n" + "-" * 70)
        print(f"CASE {case_id}")
        print("-" * 70)

        print("Query:")
        print(query)

        print(
            f"Expected intent:  {expected_intent}"
        )

        print(
            f"Predicted intent: {result['intent']}"
        )

        print(
            f"Similarity:       "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Intent correct:   {intent_correct}"
        )

        print(
            f"No data leakage:  {no_leakage}"
        )

        print(
            f"Response quality: {response_quality}"
        )

        print(
            f"Overall pass:     {passed}"
        )

        print("\nGenerated response:")
        print(response)

    results_df = pd.DataFrame(results)

    intent_accuracy = (
        results_df["intent_correct"].mean() * 100
    )

    safety_rate = (
        results_df["no_sensitive_leakage"].mean() * 100
    )

    quality_rate = (
        results_df["response_quality"].mean() * 100
    )

    overall_pass_rate = (
        results_df["overall_pass"].mean() * 100
    )

    average_similarity = (
        results_df["similarity"].mean()
    )

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nIntent Accuracy:       "
        f"{intent_accuracy:.2f}%"
    )

    print(
        f"Response Quality:     "
        f"{quality_rate:.2f}%"
    )

    print(
        f"Safety / No Leakage:  "
        f"{safety_rate:.2f}%"
    )

    print(
        f"Overall Pass Rate:    "
        f"{overall_pass_rate:.2f}%"
    )

    print(
        f"Average Similarity:   "
        f"{average_similarity:.4f}"
    )

    print(
        f"Total Test Cases:     "
        f"{len(results_df)}"
    )

    output_path = (
        "data/processed/evaluation/"
        "response_quality_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nDetailed results saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    run_evaluation()
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.agent.support_agent import generate_response


GOLDEN_PATH = Path(
    "data/processed/evaluation/golden_test_cases_200_labelled.csv"
)

OUTPUT_PATH = Path(
    "data/processed/evaluation/evaluation_results_200.csv"
)

INTENTS = [
    "account_login",
    "playback",
    "playlist",
    "subscription",
    "security",
    "general",
]


def main():
    print("Loading 200-case human-labelled golden set...")

    df = pd.read_csv(GOLDEN_PATH)

    df["gold_intent"] = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Safety check
    if len(df) != 200:
        raise ValueError(f"Expected 200 cases, found {len(df)}")

    if (df["gold_intent"] == "").any():
        raise ValueError("Some golden cases are still unlabeled.")

    print(f"Golden cases: {len(df)}")
    print("Running agent evaluation...\n")

    results = []

    for i, row in df.iterrows():

        query = str(row["customer_text"])
        expected = str(row["gold_intent"])

        result = generate_response(query)

        predicted = str(result["intent"])
        similarity = float(result["similarity"])

        results.append({
            "case_id": row["case_id"],
            "customer_text": query,
            "expected_intent": expected,
            "predicted_intent": predicted,
            "similarity": similarity,
            "intent_correct": expected == predicted,
        })

        print(
            f"[{i + 1}/200] "
            f"Expected={expected} | "
            f"Predicted={predicted} | "
            f"Correct={expected == predicted}"
        )

    results_df = pd.DataFrame(results)

    y_true = results_df["expected_intent"]
    y_pred = results_df["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=INTENTS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("HIVER AI SUPPORT AGENT — 200 CASE EVALUATION")
    print("=" * 60)

    print(f"Total cases:        {len(results_df)}")
    print(f"Intent accuracy:    {accuracy:.4f}")
    print(f"Macro precision:    {precision:.4f}")
    print(f"Macro recall:       {recall:.4f}")
    print(f"Macro F1:           {f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=INTENTS,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion matrix:")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=INTENTS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=INTENTS,
        columns=INTENTS,
    )

    print(cm_df)

    print(
        f"\nAverage retrieval similarity: "
        f"{results_df['similarity'].mean():.4f}"
    )

    print(
        f"Correct predictions: "
        f"{results_df['intent_correct'].sum()}/{len(results_df)}"
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDetailed results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
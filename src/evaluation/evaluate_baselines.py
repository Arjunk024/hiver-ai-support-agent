import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


GOLDEN_PATH = Path(
    "data/processed/evaluation/golden_test_cases_200_labelled.csv"
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

    df = pd.read_csv(GOLDEN_PATH)

    texts = df["customer_text"].astype(str)
    y_true = df["gold_intent"].astype(str)

    # ---------------------------------------------------------
    # BASELINE 1 — MAJORITY CLASS
    # ---------------------------------------------------------

    majority_class = y_true.value_counts().index[0]
    majority_predictions = [majority_class] * len(df)

    majority_accuracy = accuracy_score(
        y_true,
        majority_predictions
    )

    majority_f1 = f1_score(
        y_true,
        majority_predictions,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    # ---------------------------------------------------------
    # BASELINE 2 — TF-IDF + LOGISTIC REGRESSION
    # 5-FOLD CROSS-VALIDATION
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    tfidf_predictions = cross_val_predict(
        classifier,
        X,
        y_true,
        cv=cv
    )

    tfidf_accuracy = accuracy_score(
        y_true,
        tfidf_predictions
    )

    tfidf_f1 = f1_score(
        y_true,
        tfidf_predictions,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    # ---------------------------------------------------------
    # OUR AGENT
    # ---------------------------------------------------------

    agent_accuracy = 0.9750
    agent_f1 = 0.9755

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    comparison = pd.DataFrame({
        "System": [
            "Majority baseline",
            "TF-IDF + Logistic Regression (5-fold CV)",
            "AI Support Agent",
        ],
        "Accuracy": [
            majority_accuracy,
            tfidf_accuracy,
            agent_accuracy,
        ],
        "Macro F1": [
            majority_f1,
            tfidf_f1,
            agent_f1,
        ],
    })

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(comparison.to_string(index=False))

    output_path = Path(
        "data/processed/evaluation/baseline_results.csv"
    )

    comparison.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
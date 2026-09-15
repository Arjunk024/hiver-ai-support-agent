import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/spotify_support.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"


def main():
    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("Dataset Splitting")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal rows: {len(df)}")

    # Remove completely duplicated conversations
    before = len(df)

    df = df.drop_duplicates(
        subset=["customer_text", "support_text"]
    ).reset_index(drop=True)

    print(f"Duplicates removed: {before - len(df)}")
    print(f"Rows after deduplication: {len(df)}")

    # Shuffle reproducibly
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # 80 / 10 / 10 split
    n = len(df)

    train_end = int(n * 0.80)
    validation_end = int(n * 0.90)

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nSplit completed successfully.")
    print("-" * 40)

    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(validation_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nFiles created:")
    print(f"✓ {TRAIN_PATH}")
    print(f"✓ {VALIDATION_PATH}")
    print(f"✓ {TEST_PATH}")

    print("\nDataset splitting completed successfully.")


if __name__ == "__main__":
    main()
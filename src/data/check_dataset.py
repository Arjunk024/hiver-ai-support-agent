from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")


def main():
    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("Dataset Verification")
    print("=" * 60)

    print(f"\nDataset path: {DATA_PATH}")
    print(f"File exists: {DATA_PATH.exists()}")

    if not DATA_PATH.exists():
        print("\nERROR: twcs.csv was not found.")
        return

    print("\nReading first 10,000 rows...")

    df = pd.read_csv(
        DATA_PATH,
        nrows=10_000
    )

    print(f"Rows loaded: {len(df):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  ✓ {column}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 records:")
    print(df.head(3).to_string())

    print("\nDataset verification completed successfully.")


if __name__ == "__main__":
    main()
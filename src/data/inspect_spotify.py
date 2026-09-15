from pathlib import Path
import pandas as pd


RAW_DATA = Path("data/raw/twcs.csv")


def main():
    print("=" * 60)
    print("SPOTIFYCARES DATA INSPECTION")
    print("=" * 60)

    spotify_rows = []

    print("\nScanning dataset in chunks...")

    for chunk in pd.read_csv(
        RAW_DATA,
        chunksize=100_000
    ):
        # Find tweets where SpotifyCares is mentioned
        mask = chunk["text"].astype(str).str.contains(
            "SpotifyCares",
            case=False,
            na=False
        )

        matches = chunk[mask]

        if not matches.empty:
            spotify_rows.append(matches)

    if not spotify_rows:
        print("\nNo SpotifyCares records found.")
        return

    df = pd.concat(spotify_rows, ignore_index=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal SpotifyCares-related rows: {len(df):,}")

    print("\nInbound distribution:")
    print(df["inbound"].value_counts(dropna=False))

    print("\nUnique author IDs:")
    print(df["author_id"].nunique())

    print("\nTop author IDs:")
    print(df["author_id"].value_counts().head(20))

    print("\nResponse tweet links:")
    print(
        f"Rows with response_tweet_id: "
        f"{df['response_tweet_id'].notna().sum():,}"
    )

    print("\nRows with in_response_to_tweet_id:")
    print(
        f"{df['in_response_to_tweet_id'].notna().sum():,}"
    )

    print("\nSample SpotifyCares records:")
    print(
        df[
            [
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd
import re


INPUT_DATA = Path("data/processed/spotify_support.csv")
OUTPUT_DATA = Path("data/processed/spotify_clean.csv")


def clean_text(text):
    """Normalize text without destroying its meaning."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_conversation(row):
    """Quality checks for a customer-support pair."""

    customer = row["customer_text"]
    support = row["support_text"]

    # Both messages must exist
    if not customer or not support:
        return False

    # Remove extremely short messages
    if len(customer.strip()) < 10:
        return False

    if len(support.strip()) < 10:
        return False

    # Avoid obvious URL-only messages
    customer_without_urls = re.sub(
        r"https?://\S+",
        "",
        customer
    ).strip()

    support_without_urls = re.sub(
        r"https?://\S+",
        "",
        support
    ).strip()

    if len(customer_without_urls) < 8:
        return False

    if len(support_without_urls) < 8:
        return False

    return True


def main():

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("Spotify Conversation Cleaning")
    print("=" * 70)

    if not INPUT_DATA.exists():
        print(f"\nERROR: Input file not found: {INPUT_DATA}")
        return

    print("\nLoading extracted conversations...")

    df = pd.read_csv(INPUT_DATA)

    original_count = len(df)

    print(f"Original conversations: {original_count:,}")

    # ---------------------------------------------------------
    # Text normalization
    # ---------------------------------------------------------

    print("\nCleaning text...")

    df["customer_text"] = df["customer_text"].apply(clean_text)
    df["support_text"] = df["support_text"].apply(clean_text)

    # ---------------------------------------------------------
    # Remove duplicate conversation pairs
    # ---------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "customer_text",
            "support_text"
        ]
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Duplicate pairs removed: "
        f"{duplicates_removed:,}"
    )

    # ---------------------------------------------------------
    # Quality filtering
    # ---------------------------------------------------------

    before_quality = len(df)

    df = df[
        df.apply(
            is_valid_conversation,
            axis=1
        )
    ].copy()

    quality_removed = (
        before_quality - len(df)
    )

    print(
        f"Low-quality pairs removed: "
        f"{quality_removed:,}"
    )

    # ---------------------------------------------------------
    # Final cleanup
    # ---------------------------------------------------------

    df = df.reset_index(drop=True)

    # Add a stable conversation ID
    df.insert(
        0,
        "conversation_id",
        range(1, len(df) + 1)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"\nOriginal pairs: "
        f"{original_count:,}"
    )

    print(
        f"Final clean pairs: "
        f"{len(df):,}"
    )

    print(
        f"Total removed: "
        f"{original_count - len(df):,}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_DATA}"
    )

    print("\nSample cleaned conversations:")

    print(
        df[
            [
                "conversation_id",
                "customer_text",
                "support_text"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
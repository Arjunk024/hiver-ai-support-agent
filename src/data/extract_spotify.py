from pathlib import Path
import pandas as pd


RAW_DATA = Path("data/raw/twcs.csv")
OUTPUT_DATA = Path("data/processed/spotify_support.csv")

CHUNK_SIZE = 100_000


def parse_tweet_ids(value):
    """
    Convert response_tweet_id values into individual tweet IDs.

    The TWCS dataset can contain multiple response IDs
    separated by commas.
    """
    if pd.isna(value):
        return []

    return [
        str(x).strip()
        for x in str(value).split(",")
        if str(x).strip()
    ]


def main():

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("SpotifyCares Conversation Extraction")
    print("=" * 70)

    if not RAW_DATA.exists():
        print(f"\nERROR: Dataset not found: {RAW_DATA}")
        return

    # ---------------------------------------------------------
    # PASS 1
    # Find customer tweets mentioning SpotifyCares
    # and collect their response tweet IDs.
    # ---------------------------------------------------------

    print("\nPASS 1: Finding SpotifyCares customer conversations...")

    customer_rows = []
    response_ids = set()

    total_rows = 0

    for chunk in pd.read_csv(
        RAW_DATA,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

        # Customer messages that mention SpotifyCares
        mask = (
            (chunk["inbound"] == True)
            &
            (
                chunk["text"]
                .astype(str)
                .str.contains(
                    "SpotifyCares",
                    case=False,
                    na=False
                )
            )
        )

        matches = chunk[mask].copy()

        if not matches.empty:

            customer_rows.append(matches)

            for value in matches["response_tweet_id"]:
                response_ids.update(
                    parse_tweet_ids(value)
                )

        print(
            f"Processed {total_rows:,} rows | "
            f"Customer messages found: "
            f"{sum(len(x) for x in customer_rows):,}"
        )

    if not customer_rows:
        print("\nERROR: No SpotifyCares customer messages found.")
        return

    customers = pd.concat(
        customer_rows,
        ignore_index=True
    )

    print("\nCustomer messages found:")
    print(f"  {len(customers):,}")

    print("\nUnique support response IDs:")
    print(f"  {len(response_ids):,}")

    # ---------------------------------------------------------
    # PASS 2
    # Find the actual support responses using tweet IDs.
    # ---------------------------------------------------------

    print("\nPASS 2: Finding Spotify support responses...")

    support_rows = []

    total_rows = 0

    for chunk in pd.read_csv(
        RAW_DATA,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

        mask = (
            (chunk["inbound"] == False)
            &
            (chunk["tweet_id"].astype(str).isin(response_ids))
        )

        matches = chunk[mask].copy()

        if not matches.empty:
            support_rows.append(matches)

    if support_rows:
        support = pd.concat(
            support_rows,
            ignore_index=True
        )
    else:
        support = pd.DataFrame(
            columns=customers.columns
        )

    print("\nSupport responses found:")
    print(f"  {len(support):,}")

    # ---------------------------------------------------------
    # Build customer -> support pairs
    # ---------------------------------------------------------

    print("\nBuilding conversation pairs...")

    support_lookup = {}

    for _, row in support.iterrows():

        support_lookup[str(row["tweet_id"])] = row

    conversations = []

    for _, customer in customers.iterrows():

        response_ids_for_customer = parse_tweet_ids(
            customer["response_tweet_id"]
        )

        for response_id in response_ids_for_customer:

            if response_id in support_lookup:

                support_row = support_lookup[response_id]

                conversations.append(
                    {
                        "customer_tweet_id":
                            customer["tweet_id"],

                        "customer_author_id":
                            customer["author_id"],

                        "customer_created_at":
                            customer["created_at"],

                        "customer_text":
                            customer["text"],

                        "support_tweet_id":
                            support_row["tweet_id"],

                        "support_author_id":
                            support_row["author_id"],

                        "support_created_at":
                            support_row["created_at"],

                        "support_text":
                            support_row["text"],

                        "in_response_to_tweet_id":
                            support_row[
                                "in_response_to_tweet_id"
                            ],
                    }
                )

    conversations_df = pd.DataFrame(
        conversations
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conversations_df.to_csv(
        OUTPUT_DATA,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nConversation pairs: "
        f"{len(conversations_df):,}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_DATA}"
    )

    if not conversations_df.empty:

        print("\nSample conversations:")

        print(
            conversations_df[
                [
                    "customer_text",
                    "support_text"
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
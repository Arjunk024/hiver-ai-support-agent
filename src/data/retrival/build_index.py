import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

TRAIN_PATH = "data/processed/train.csv"
OUTPUT_DIR = "data/processed/retrieval"

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "train_embeddings.npy")
DATA_PATH = os.path.join(OUTPUT_DIR, "retrieval_data.csv")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("Building Retrieval Index")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------

    print("\nLoading training dataset...")

    df = pd.read_csv(TRAIN_PATH)

    print("Training rows:", len(df))

    # Make sure required columns exist
    required_columns = [
        "customer_text",
        "support_text"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Remove rows with missing text
    df = df.dropna(
        subset=["customer_text", "support_text"]
    ).reset_index(drop=True)

    print("Rows after cleaning:", len(df))

    # --------------------------------------------------
    # 2. Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded successfully.")

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    print("\nGenerating customer-text embeddings...")

    customer_texts = df["customer_text"].astype(str).tolist()

    embeddings = model.encode(
        customer_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print("\nEmbedding shape:", embeddings.shape)

    # --------------------------------------------------
    # 4. Save embeddings
    # --------------------------------------------------

    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    # --------------------------------------------------
    # 5. Save corresponding data
    # --------------------------------------------------

    retrieval_data = df[
        [
            "customer_tweet_id",
            "customer_text",
            "support_text"
        ]
    ].copy()

    retrieval_data.to_csv(
        DATA_PATH,
        index=False
    )

    # --------------------------------------------------
    # 6. Verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVAL INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print("Embeddings:", EMBEDDINGS_PATH)
    print("Embedding shape:", embeddings.shape)

    print("Retrieval data:", DATA_PATH)
    print("Retrieval rows:", len(retrieval_data))

    print("\nSample:")
    print(
        retrieval_data.head(3).to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
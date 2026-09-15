import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

EMBEDDINGS_PATH = os.getenv(
    "RETRIEVAL_EMBEDDINGS_PATH",
   "data/processed/retrieval/train_embeddings_leakage_free.npy"
)

DATA_PATH = os.getenv(
    "RETRIEVAL_DATA_PATH",
   "data/processed/retrieval/retrieval_data_leakage_free.csv"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded successfully.")


# --------------------------------------------------
# LOAD RETRIEVAL DATA
# --------------------------------------------------

print("Loading retrieval index...")

embeddings = np.load(EMBEDDINGS_PATH)

df = pd.read_csv(DATA_PATH)

print("Embeddings shape:", embeddings.shape)
print("Retrieval rows:", len(df))


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search(query, top_k=5):

    # Convert customer query into embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )[0]

    # Because both vectors are normalized,
    # dot product = cosine similarity
    scores = embeddings @ query_embedding

    # Get top results
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    results["similarity"] = scores[top_indices]

    return results


# --------------------------------------------------
# INTERACTIVE SEARCH
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("Semantic Search")
    print("=" * 60)

    while True:

        query = input("\nEnter customer problem (or 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query.strip():
            continue

        results = search(query, top_k=5)

        print("\n" + "-" * 60)
        print("TOP MATCHES")
        print("-" * 60)

        for i, (_, row) in enumerate(results.iterrows(), start=1):

            print(f"\nRESULT {i}")
            print(f"Similarity: {row['similarity']:.4f}")

            print("\nCustomer:")
            print(row["customer_text"])

            print("\nHistorical Support Response:")
            print(row["support_text"])

            print("-" * 60)
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "apple_support_sample.csv"
)

GOLDEN_FILE = os.path.join(
    PROJECT_ROOT,
    "golden_set.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading retrieval dataset...")

df = pd.read_csv(DATA_FILE)

df["customer_tweet_id"] = (
    df["customer_tweet_id"]
    .astype(str)
)


# ============================================================
# REMOVE GOLDEN SET FROM RETRIEVAL INDEX
# ============================================================

if os.path.exists(GOLDEN_FILE):

    golden = pd.read_csv(GOLDEN_FILE)

    golden_ids = set(
        golden["customer_tweet_id"]
        .astype(str)
        .tolist()
    )

    before = len(df)

    df = df[
        ~df["customer_tweet_id"].isin(golden_ids)
    ].copy()

    removed = before - len(df)

    print(
        f"Excluded {removed} golden examples "
        "from retrieval index."
    )


# ============================================================
# BUILD SEARCH TEXT
# ============================================================

df["previous_context"] = (
    df["previous_context"]
    .fillna("")
    .astype(str)
)

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
)

df["search_text"] = (
    df["previous_context"]
    + " "
    + df["customer_message"]
)


# ============================================================
# TF-IDF INDEX
# ============================================================

print(
    f"Building retrieval index from {len(df)} examples..."
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
)

document_vectors = vectorizer.fit_transform(
    df["search_text"]
)

print("Retrieval index ready.")


# ============================================================
# RETRIEVE SIMILAR CASES
# ============================================================

def retrieve_similar_cases(
    customer_message,
    previous_context="",
    top_k=5
):

    query = (
        str(previous_context)
        + " "
        + str(customer_message)
    ).strip()

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for idx in top_indices:

        row = df.iloc[idx]

        results.append({
            "similarity": float(
                similarities[idx]
            ),
            "customer_message": row[
                "customer_message"
            ],
            "previous_context": row[
                "previous_context"
            ],
            "apple_reply": row[
                "apple_reply"
            ],
            "customer_tweet_id": row[
                "customer_tweet_id"
            ],
            "apple_tweet_id": row[
                "apple_tweet_id"
            ],
        })

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    customer_message,
    previous_context=""
):

    print()
    print("=" * 70)
    print("CUSTOMER")
    print("=" * 70)

    print(customer_message)

    print()
    print("=" * 70)
    print("SIMILAR HISTORICAL CASES")
    print("=" * 70)

    results = retrieve_similar_cases(
        customer_message,
        previous_context,
        top_k=5
    )

    for i, result in enumerate(results, 1):

        print()
        print(
            f"Case {i} "
            f"(similarity: "
            f"{result['similarity']:.2%})"
        )

        print(
            "Customer:",
            result["customer_message"]
        )

        print(
            "Apple:",
            result["apple_reply"]
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    test_cases = [
        "My battery is draining very quickly",
        "My iPhone keeps freezing and restarting",
        "My WiFi keeps disconnecting",
        "My iMessage is not working",
        "I cannot access my iCloud account",
    ]

    for message in test_cases:

        print_results(message)
import pandas as pd

FILE = "twcs.csv"
BRAND = "AppleSupport"

print("Loading dataset...")
df = pd.read_csv(FILE)

print("Total tweets:", len(df))

# ---------------------------------------------------------
# STEP 1: Keep only AppleSupport replies
# ---------------------------------------------------------

apple = df[df["author_id"] == BRAND].copy()

print("AppleSupport tweets:", len(apple))

# ---------------------------------------------------------
# STEP 2: Find the customer tweet that Apple replied to
# ---------------------------------------------------------

# Apple tweet -> customer tweet
pairs = apple.merge(
    df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id"
        ]
    ],
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    how="inner",
    suffixes=("_apple", "_customer")
)

# Keep only cases where the parent tweet was from a customer
pairs = pairs[pairs["inbound_customer"] == True].copy()

print("Apple/customer interactions:", len(pairs))

# ---------------------------------------------------------
# STEP 3: Add the customer's previous message if available
# ---------------------------------------------------------

# Rename for clarity
pairs["customer_tweet_id"] = pairs["tweet_id_customer"]
pairs["customer_message"] = pairs["text_customer"]
pairs["apple_tweet_id"] = pairs["tweet_id_apple"]
pairs["apple_reply"] = pairs["text_apple"]

# The customer tweet itself may have been a reply to another tweet.
# Find that previous tweet.
context = df[
    [
        "tweet_id",
        "text",
        "author_id"
    ]
].copy()

context.columns = [
    "previous_tweet_id",
    "previous_text",
    "previous_author"
]

pairs["previous_tweet_id"] = pairs["in_response_to_tweet_id_customer"]

pairs = pairs.merge(
    context,
    on="previous_tweet_id",
    how="left"
)

# ---------------------------------------------------------
# STEP 4: Create a clean dataset
# ---------------------------------------------------------

result = pairs[
    [
        "customer_tweet_id",
        "previous_text",
        "customer_message",
        "apple_tweet_id",
        "apple_reply",
        "created_at_customer"
    ]
].copy()

result.columns = [
    "customer_tweet_id",
    "previous_context",
    "customer_message",
    "apple_tweet_id",
    "apple_reply",
    "created_at"
]

# Remove duplicate customer interactions
result = result.drop_duplicates(
    subset=["customer_tweet_id"]
)

# Remove rows where the customer message is missing
result = result.dropna(
    subset=["customer_message", "apple_reply"]
)

print("Clean interactions:", len(result))

# ---------------------------------------------------------
# STEP 5: Save a manageable sample
# ---------------------------------------------------------

# We don't need 106,000 examples.
# 10,000 is more than enough for our development work.

sample_size = min(10000, len(result))

sample = result.sample(
    n=sample_size,
    random_state=42
)

sample.to_csv(
    "apple_support_sample.csv",
    index=False
)

print("Saved:", len(sample), "examples")
print("File: apple_support_sample.csv")

# ---------------------------------------------------------
# STEP 6: Show examples
# ---------------------------------------------------------

print("\nSample examples:\n")

for _, row in sample.head(20).iterrows():

    print("=" * 80)

    print("PREVIOUS CONTEXT:")
    print(row["previous_context"])

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAPPLE SUPPORT:")
    print(row["apple_reply"])
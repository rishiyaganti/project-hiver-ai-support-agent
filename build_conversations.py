import pandas as pd
import json

FILE = "twcs.csv"
BRAND = "AppleSupport"

print("Loading dataset...")

df = pd.read_csv(FILE)

print("Dataset loaded:", len(df), "tweets")

# ---------------------------------------------------------
# 1. Keep AppleSupport tweets
# ---------------------------------------------------------

brand_df = df[df["author_id"] == BRAND].copy()

# ---------------------------------------------------------
# 2. Create a quick lookup:
#    tweet_id -> complete tweet information
# ---------------------------------------------------------

tweet_lookup = {}

for _, row in df.iterrows():
    tweet_lookup[str(row["tweet_id"])] = {
        "tweet_id": str(row["tweet_id"]),
        "author_id": str(row["author_id"]),
        "inbound": bool(row["inbound"]),
        "created_at": row["created_at"],
        "text": str(row["text"])
    }

# ---------------------------------------------------------
# 3. Build conversations around Apple responses
# ---------------------------------------------------------

conversations = []

for _, brand_row in brand_df.iterrows():

    parent_id = brand_row["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    parent_id = str(int(parent_id))

    # Find the customer tweet Apple responded to
    if parent_id not in tweet_lookup:
        continue

    customer = tweet_lookup[parent_id]

    if customer["inbound"] != True:
        continue

    conversation = {
        "customer_tweet_id": customer["tweet_id"],
        "customer_message": customer["text"],
        "brand_tweet_id": str(brand_row["tweet_id"]),
        "brand_reply": str(brand_row["text"]),
        "created_at": customer["created_at"]
    }

    conversations.append(conversation)

# ---------------------------------------------------------
# 4. Remove duplicates
# ---------------------------------------------------------

unique_conversations = {}

for conversation in conversations:

    key = conversation["customer_tweet_id"]

    unique_conversations[key] = conversation

conversations = list(unique_conversations.values())

print("\nApple customer-support interactions:", len(conversations))

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------

with open(
    "apple_conversations.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        conversations,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved apple_conversations.json")

# ---------------------------------------------------------
# 6. Show examples
# ---------------------------------------------------------

print("\nSample conversations:\n")

for conversation in conversations[:20]:

    print("=" * 80)

    print("CUSTOMER:")
    print(conversation["customer_message"])

    print("\nAPPLE SUPPORT:")
    print(conversation["brand_reply"])
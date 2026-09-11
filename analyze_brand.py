import pandas as pd

FILE = "twcs.csv"
BRAND = "AppleSupport"

print("Loading dataset...")

df = pd.read_csv(FILE)

print("Dataset loaded.")
print("Total tweets:", len(df))

# Get tweets from our selected support brand
brand_df = df[df["author_id"] == BRAND].copy()

print("\nSelected brand:", BRAND)
print("Support tweets:", len(brand_df))

# Count how many customer tweets are directly answered by this brand.
# A customer tweet is inbound=True.
# If its tweet_id appears in another tweet's in_response_to_tweet_id,
# it means another tweet replied to it.

brand_ids = set(brand_df["tweet_id"].astype(str))

customer_df = df[df["inbound"] == True].copy()

customer_df["tweet_id_str"] = customer_df["tweet_id"].astype(str)

customer_df["has_brand_response"] = (
    customer_df["tweet_id_str"].isin(
        set(
            brand_df["in_response_to_tweet_id"]
            .dropna()
            .astype(int)
            .astype(str)
        )
    )
)

answered_customers = customer_df[
    customer_df["has_brand_response"]
]

print("Customer tweets answered by", BRAND, ":", len(answered_customers))

# Show examples
print("\nSample customer messages:")

for _, row in answered_customers.head(20).iterrows():
    print("-" * 80)
    print(row["text"])

# Save Apple-related tweets
brand_df.to_csv("selected_brand_tweets.csv", index=False)

print("\nSaved selected_brand_tweets.csv")
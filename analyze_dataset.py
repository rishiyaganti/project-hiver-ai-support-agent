import pandas as pd

FILE = "twcs.csv"

print("Reading dataset...")
df = pd.read_csv(FILE)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic information:")
print(df.info())

# Find tweets written by brands.
# In this dataset, inbound=False means the tweet is from the brand/support side.
brand_tweets = df[df["inbound"] == False].copy()

print("\nBrand/support tweets:", len(brand_tweets))

# The author_id of outbound tweets represents the brand's Twitter account.
brand_counts = (
    brand_tweets["author_id"]
    .value_counts()
    .reset_index()
)

brand_counts.columns = ["brand_id", "support_tweets"]

print("\nTop brands/support accounts:")
print(brand_counts.head(30))

# Save the results so we can inspect them later.
brand_counts.to_csv("brand_counts.csv", index=False)

print("\nSaved brand_counts.csv")
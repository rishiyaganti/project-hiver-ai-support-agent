import pandas as pd
import re
from collections import Counter

FILE = "apple_support_sample.csv"

print("Loading Apple dataset...")
df = pd.read_csv(FILE)

print("Rows:", len(df))


# ---------------------------------------------------------
# 1. Basic statistics
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isna().sum())


# ---------------------------------------------------------
# 2. Look at customer messages
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CUSTOMER MESSAGE LENGTH")
print("=" * 80)

df["message_length"] = df["customer_message"].fillna("").str.len()

print(df["message_length"].describe())


# ---------------------------------------------------------
# 3. Common keywords
# ---------------------------------------------------------

text = " ".join(
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.lower()
)

# Remove URLs and punctuation
text = re.sub(r"http\S+", " ", text)
text = re.sub(r"[^a-z\s]", " ", text)

words = text.split()

# Common English words that aren't useful for discovering intents
stopwords = {
    "the", "and", "to", "of", "a", "i", "is", "it", "for",
    "my", "on", "in", "this", "that", "with", "me", "you",
    "was", "be", "have", "are", "but", "not", "so", "do",
    "can", "just", "we", "they", "your", "from", "or", "if",
    "at", "what", "why", "how", "has", "had", "im", "its",
    "apple", "support", "please", "help", "phone"
}

useful_words = [
    word for word in words
    if word not in stopwords and len(word) > 2
]

counts = Counter(useful_words)

print("\n" + "=" * 80)
print("TOP CUSTOMER KEYWORDS")
print("=" * 80)

for word, count in counts.most_common(100):
    print(f"{word:25} {count}")


# ---------------------------------------------------------
# 4. Search for important support themes
# ---------------------------------------------------------

themes = {
    "battery": [
        "battery", "drain", "charging", "charge", "dies",
        "power", "battery life"
    ],

    "update": [
        "update", "ios", "ios11", "ios 11", "upgrade"
    ],

    "icloud": [
        "icloud", "i cloud"
    ],

    "apple_id": [
        "apple id", "account recovery", "password",
        "forgot password", "locked out", "verification"
    ],

    "iphone": [
        "iphone", "phone"
    ],

    "ipad": [
        "ipad"
    ],

    "mac": [
        "macbook", "mac", "imac", "laptop"
    ],

    "apple_watch": [
        "watch", "apple watch"
    ],

    "app_store": [
        "app store", "itunes", "download app", "purchase app"
    ],

    "payment": [
        "payment", "paid", "charge", "refund", "billing",
        "credit card", "card"
    ],

    "messages": [
        "imessage", "message", "sms", "text"
    ],

    "calls": [
        "call", "calling", "phone call"
    ],

    "wifi": [
        "wifi", "wi-fi", "internet", "network"
    ],

    "bluetooth": [
        "bluetooth"
    ],

    "keyboard": [
        "keyboard", "typing", "keys"
    ],

    "screen": [
        "screen", "display", "black screen"
    ],

    "camera": [
        "camera", "photo", "photos", "video"
    ],

    "audio": [
        "sound", "speaker", "volume", "audio", "microphone"
    ],

    "accessibility": [
        "accessibility", "voiceover", "voice over"
    ],

    "warranty_repair": [
        "warranty", "repair", "replacement", "service"
    ]
}


print("\n" + "=" * 80)
print("THEME FREQUENCIES")
print("=" * 80)

theme_counts = {}

customer_text = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.lower()
)

for theme, keywords in themes.items():

    mask = customer_text.apply(
        lambda x: any(keyword in x for keyword in keywords)
    )

    count = mask.sum()

    theme_counts[theme] = count

for theme, count in sorted(
    theme_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{theme:25} {count}")


# ---------------------------------------------------------
# 5. Show examples for each theme
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("EXAMPLES BY THEME")
print("=" * 80)

for theme, keywords in themes.items():

    mask = customer_text.apply(
        lambda x: any(keyword in x for keyword in keywords)
    )

    examples = df[mask].head(5)

    if len(examples) == 0:
        continue

    print("\n")
    print("#" * 80)
    print("THEME:", theme)
    print("#" * 80)

    for _, row in examples.iterrows():

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nAPPLE:")
        print(row["apple_reply"])

        print("-" * 60)


# ---------------------------------------------------------
# 6. Save theme statistics
# ---------------------------------------------------------

theme_df = pd.DataFrame(
    list(theme_counts.items()),
    columns=["theme", "count"]
)

theme_df = theme_df.sort_values(
    "count",
    ascending=False
)

theme_df.to_csv(
    "apple_theme_counts.csv",
    index=False
)

print("\nSaved: apple_theme_counts.csv")
import pandas as pd

SOURCE = "apple_support_sample.csv"
OUTPUT = "golden_set.csv"

print("Loading dataset...")

df = pd.read_csv(SOURCE)

# ---------------------------------------------------------
# Clean text
# ---------------------------------------------------------

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
)

df["previous_context"] = (
    df["previous_context"]
    .fillna("")
    .astype(str)
)


# ---------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------

intents = [
    "software_update",
    "battery_power",
    "device_performance",
    "connectivity_network",
    "messaging_calls",
    "apps_media",
    "account_icloud",
    "hardware_repair",
    "payments_billing",
    "settings_features",
    "feedback_request",
    "other_unclear"
]


# ---------------------------------------------------------
# Create a diverse sample
# ---------------------------------------------------------

# Start with a random sample
sample = df.sample(
    n=min(200, len(df)),
    random_state=42
).copy()


# ---------------------------------------------------------
# Add empty human-label columns
# ---------------------------------------------------------

sample["intent"] = ""
sample["escalate"] = ""
sample["label_reason"] = ""


# ---------------------------------------------------------
# Keep only useful columns
# ---------------------------------------------------------

golden = sample[
    [
        "customer_tweet_id",
        "previous_context",
        "customer_message",
        "apple_reply",
        "intent",
        "escalate",
        "label_reason"
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

golden.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("GOLDEN SET CREATED")
print("=" * 70)

print("Examples:", len(golden))
print("File:", OUTPUT)

print()
print("Allowed intents:")
for intent in intents:
    print("-", intent)

print()
print("For each example fill:")
print("intent       -> one of the intents above")
print("escalate     -> yes / no")
print("label_reason -> short explanation")
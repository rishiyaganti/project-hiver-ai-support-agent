import pandas as pd
import re

INPUT_FILE = "golden_set.csv"
OUTPUT_FILE = "golden_set_auto_labeled.csv"


INTENTS = {
    1: "software_update",
    2: "battery_power",
    3: "device_performance",
    4: "connectivity_network",
    5: "messaging_calls",
    6: "apps_media",
    7: "account_icloud",
    8: "hardware_repair",
    9: "payments_billing",
    10: "settings_features",
    11: "feedback_request",
    12: "other_unclear",
}


def classify(text):
    text = str(text).lower()

    # Hardware / repair
    if any(x in text for x in [
        "repair", "broken", "cracked", "screen", "replacement",
        "warranty", "hardware", "damaged", "service"
    ]):
        return 8

    # Payments / billing
    if any(x in text for x in [
        "charge", "charged", "billing", "payment", "refund",
        "subscription", "purchase", "invoice", "money", "paid"
    ]):
        return 9

    # Battery
    if any(x in text for x in [
        "battery", "charging", "charge won't", "dies", "drain",
        "battery life", "power"
    ]):
        return 2

    # Calls / messaging
    if any(x in text for x in [
        "can't call", "cannot call", "call", "calls",
        "imessage", "message", "sms", "text", "facetime"
    ]):
        return 5

    # Connectivity
    if any(x in text for x in [
        "wifi", "wi-fi", "bluetooth", "network", "signal",
        "no service", "cellular", "sim card", "sim"
    ]):
        return 4

    # Account / iCloud
    if any(x in text for x in [
        "icloud", "apple id", "appleid", "password",
        "account", "login", "sign in", "storage plan"
    ]):
        return 7

    # Software updates
    if any(x in text for x in [
        "update", "updating", "ios version", "ios 11",
        "ios 12", "ios 13", "software version", "upgrade"
    ]):
        return 1

    # Performance
    if any(x in text for x in [
        "slow", "freezing", "freeze", "frozen", "restart",
        "restarts", "crash", "crashing", "glitch",
        "overheat", "overheating", "not opening"
    ]):
        return 3

    # Apps / media
    if any(x in text for x in [
        "music", "itunes", "app store", "app", "photo",
        "camera", "video", "apple tv", "podcast"
    ]):
        return 6

    # Settings / features
    if any(x in text for x in [
        "keyboard", "lock screen", "setting", "settings",
        "accessibility", "display", "notification",
        "control center", "feature", "unlock"
    ]):
        return 10

    # Feedback
    if any(x in text for x in [
        "feature request", "suggestion", "please add",
        "would love", "feedback", "bring back"
    ]):
        return 11

    return 12


def escalation(text, intent):
    text = str(text).lower()

    high_risk = [
        "human", "agent", "speak to someone", "representative",
        "lawsuit", "legal", "refund", "charge", "charged",
        "warranty", "repair", "replacement", "stolen",
        "locked out", "can't unlock", "cannot unlock",
        "random restart", "keeps restarting"
    ]

    if any(x in text for x in high_risk):
        return "yes"

    if intent in [8, 9]:
        return "yes"

    if intent == 3 and any(x in text for x in [
        "random restart", "keeps restarting", "constantly",
        "multiple apps", "all apps", "completely unusable"
    ]):
        return "yes"

    return "no"


def reason(intent, escalate):
    intent_name = INTENTS[intent]

    reasons = {
        1: "Customer is reporting a software update or version-related issue.",
        2: "Customer is reporting a battery, charging, or power-related issue.",
        3: "Customer is reporting device performance or stability problems.",
        4: "Customer is reporting a connectivity, cellular, Wi-Fi, Bluetooth, or SIM-related issue.",
        5: "Customer is reporting a calling or messaging-related issue.",
        6: "Customer is reporting an app, media, camera, music, or Apple TV-related issue.",
        7: "Customer is reporting an Apple ID, iCloud, account, or storage-plan issue.",
        8: "Customer is reporting a hardware, repair, replacement, or warranty issue.",
        9: "Customer is reporting a payment, billing, subscription, charge, or refund issue.",
        10: "Customer is reporting a device setting or feature behavior issue.",
        11: "Customer is providing feedback or requesting a product feature.",
        12: "The customer's primary support intent cannot be determined confidently from the available context."
    }

    base = reasons[intent]

    if escalate == "yes":
        base += " Human intervention is appropriate because the issue may require service, account-level action, financial action, or deeper troubleshooting."

    return base


# Load data
df = pd.read_csv(INPUT_FILE)

# Combine context + customer message
df["full_text"] = (
    df["previous_context"].fillna("").astype(str)
    + " "
    + df["customer_message"].fillna("").astype(str)
)

# Automatically label
df["auto_intent_number"] = df["full_text"].apply(classify)
df["auto_intent"] = df["auto_intent_number"].map(INTENTS)

df["auto_escalate"] = df.apply(
    lambda row: escalation(row["full_text"], row["auto_intent_number"]),
    axis=1
)

df["auto_label_reason"] = df.apply(
    lambda row: reason(
        row["auto_intent_number"],
        row["auto_escalate"]
    ),
    axis=1
)

# Mark confidence for human review
df["review_status"] = "REVIEW"

df["confidence"] = "medium"

for i, row in df.iterrows():
    text = row["full_text"].lower()

    # Strong explicit signals
    if row["auto_intent_number"] == 2 and "battery" in text:
        df.at[i, "confidence"] = "high"

    elif row["auto_intent_number"] == 7 and (
        "icloud" in text or "apple id" in text
    ):
        df.at[i, "confidence"] = "high"

    elif row["auto_intent_number"] == 8 and any(
        x in text for x in ["repair", "warranty", "replacement"]
    ):
        df.at[i, "confidence"] = "high"

    elif row["auto_intent_number"] == 9 and any(
        x in text for x in ["charge", "refund", "billing", "payment"]
    ):
        df.at[i, "confidence"] = "high"

    elif row["auto_intent_number"] == 1 and any(
        x in text for x in ["update", "ios version", "upgrade"]
    ):
        df.at[i, "confidence"] = "high"

# Remove helper column
df.drop(columns=["full_text"], inplace=True)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("AUTO-LABELING COMPLETE")
print("=" * 70)
print(f"Examples: {len(df)}")
print(f"Output: {OUTPUT_FILE}")
print()
print("Intent distribution:")
print(df["auto_intent"].value_counts())
print()
print("Escalation distribution:")
print(df["auto_escalate"].value_counts())
print()
print("Confidence:")
print(df["confidence"].value_counts())
print()
print("IMPORTANT:")
print("These are AI-assisted labels and MUST be human reviewed")
print("before being used as the final golden evaluation set.")
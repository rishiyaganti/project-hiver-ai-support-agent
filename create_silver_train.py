import pandas as pd
import re


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "apple_support_sample.csv"
GOLDEN_FILE = "golden_set_auto_labeled.csv"
OUTPUT_FILE = "silver_train.csv"


INTENTS = [
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
    "other_unclear",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading AppleSupport dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} examples")


# ============================================================
# PROTECT GOLDEN SET
# ============================================================

golden = pd.read_csv(GOLDEN_FILE)

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

print(f"Golden examples protected: {len(golden_ids)}")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Remove @mentions
    value = re.sub(r"@\w+", " ", value)

    # Remove URLs
    value = re.sub(r"http\S+|www\.\S+", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


df["previous_context"] = df["previous_context"].fillna("")
df["customer_message"] = df["customer_message"].fillna("")

df["text"] = (
    df["previous_context"].apply(clean_text)
    + " "
    + df["customer_message"].apply(clean_text)
)


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(text):

    text = text.lower()


    # --------------------------------------------------------
    # 1. HARDWARE / REPAIR
    # --------------------------------------------------------

    hardware = [
        "broken",
        "cracked",
        "crack",
        "bent",
        "damaged",
        "repair",
        "replacement",
        "replace",
        "warranty",
        "applecare",
        "genius bar",
        "charging port",
        "earpods",
        "earpod",
        "headphone jack",
        "screen shattered",
        "physical damage",
    ]

    if any(word in text for word in hardware):
        return "hardware_repair"


    # --------------------------------------------------------
    # 2. PAYMENTS / BILLING
    # --------------------------------------------------------

    payments = [
        "charged",
        "charge",
        "billing",
        "bill",
        "refund",
        "subscription",
        "payment",
        "paid",
        "purchase",
        "invoice",
        "money",
        "credit card",
        "debit card",
        "itunes charge",
        "unauthorized charge",
    ]

    if any(word in text for word in payments):
        return "payments_billing"


    # --------------------------------------------------------
    # 3. ACCOUNT / ICLOUD
    # --------------------------------------------------------

    account = [
        "icloud",
        "apple id",
        "appleid",
        "account",
        "password",
        "two factor",
        "2fa",
        "verification code",
        "trusted device",
        "account disabled",
        "account locked",
        "sign in",
        "login",
        "logged out",
    ]

    if any(word in text for word in account):
        return "account_icloud"


    # --------------------------------------------------------
    # 4. BATTERY / POWER
    # --------------------------------------------------------

    battery = [
        "battery",
        "batteries",
        "charging",
        "charge",
        "drain",
        "draining",
        "dies",
        "dying",
        "power",
        "shuts down",
        "shutdown",
        "percentage",
        "battery life",
    ]

    if any(word in text for word in battery):
        return "battery_power"


    # --------------------------------------------------------
    # 5. CONNECTIVITY / NETWORK
    # --------------------------------------------------------

    connectivity = [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "network",
        "internet",
        "lte",
        "4g",
        "5g",
        "no service",
        "signal",
        "sim",
        "sim card",
        "connection",
        "connect",
    ]

    if any(word in text for word in connectivity):
        return "connectivity_network"


    # --------------------------------------------------------
    # 6. SOFTWARE UPDATE
    # --------------------------------------------------------

    update = [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 10",
        "software update",
        "update",
        "updating",
        "upgrade",
        "upgrading",
        "latest ios",
        "new ios",
        "beta",
        "developer beta",
        "macos update",
        "macos upgrade",
        "watchos",
    ]

    if any(word in text for word in update):
        return "software_update"


    # --------------------------------------------------------
    # 7. MESSAGING / CALLS
    # --------------------------------------------------------

    messaging = [
        "imessage",
        "i message",
        "text message",
        "messages",
        "sms",
        "facetime",
        "face time",
        "call",
        "calling",
        "phone call",
        "calls",
        "voicemail",
        "mail",
        "email",
        "sending email",
        "receiving email",
    ]

    if any(word in text for word in messaging):
        return "messaging_calls"


    # --------------------------------------------------------
    # 8. APPS / MEDIA
    # --------------------------------------------------------

    apps_media = [
        "app store",
        "appstore",
        "itunes",
        "apple music",
        "music",
        "photos",
        "photo",
        "camera",
        "video",
        "garageband",
        "pandora",
        "amazon music",
        "youtube",
        "netflix",
        "maps",
        "game",
        "games",
        "application",
        "apps",
        "app",
    ]

    if any(word in text for word in apps_media):
        return "apps_media"


    # --------------------------------------------------------
    # 9. DEVICE PERFORMANCE
    # --------------------------------------------------------

    performance = [
        "slow",
        "slower",
        "lag",
        "lagging",
        "freeze",
        "freezes",
        "frozen",
        "freezing",
        "crash",
        "crashes",
        "crashing",
        "restart",
        "restarts",
        "restarting",
        "unresponsive",
        "not responding",
        "overheating",
        "overheat",
        "hot",
        "stuck",
        "black screen",
        "won't open",
        "won't work",
        "keeps closing",
        "closing apps",
        "performance",
    ]

    if any(word in text for word in performance):
        return "device_performance"


    # --------------------------------------------------------
    # 10. SETTINGS / FEATURES
    # --------------------------------------------------------

    settings = [
        "keyboard",
        "autocorrect",
        "auto correct",
        "lock screen",
        "control center",
        "notification",
        "notifications",
        "brightness",
        "display",
        "screen setting",
        "accessibility",
        "do not disturb",
        "airplane mode",
        "settings",
        "font size",
        "text size",
        "passcode",
        "unlock",
        "feature",
        "widget",
    ]

    if any(word in text for word in settings):
        return "settings_features"


    # --------------------------------------------------------
    # 11. FEEDBACK
    # --------------------------------------------------------

    feedback = [
        "feature request",
        "suggestion",
        "suggest",
        "feedback",
        "please add",
        "would be nice",
        "wish apple",
        "request a feature",
    ]

    if any(word in text for word in feedback):
        return "feedback_request"


    # --------------------------------------------------------
    # 12. UNKNOWN
    # --------------------------------------------------------

    return "other_unclear"


# ============================================================
# APPLY LABELS
# ============================================================

print("\nAssigning silver labels...")

df["intent"] = df["text"].apply(classify_intent)


# ============================================================
# REMOVE GOLDEN SET FROM TRAINING
# ============================================================

before = len(df)

df = df[
    ~df["customer_tweet_id"].astype(str).isin(golden_ids)
].copy()

removed = before - len(df)

print(f"Removed golden-set examples: {removed}")
print(f"Remaining training examples: {len(df):,}")


# ============================================================
# REMOVE VERY SHORT EXAMPLES
# ============================================================

df = df[
    df["customer_message"].str.strip().str.len() >= 5
].copy()


# ============================================================
# SHOW DISTRIBUTION
# ============================================================

print("\nSilver-label distribution:")
print("-" * 40)

counts = df["intent"].value_counts()

for intent in INTENTS:

    count = counts.get(intent, 0)

    print(f"{intent:25s} {count:6,}")


# ============================================================
# SAVE
# ============================================================

columns_to_save = [
    "customer_tweet_id",
    "previous_context",
    "customer_message",
    "apple_reply",
    "created_at",
    "intent",
]

df[columns_to_save].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Saved: {OUTPUT_FILE}")
print(f"Training examples: {len(df):,}")
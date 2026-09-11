import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# 1. Intent definitions
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 2. Load labelled training data
# ---------------------------------------------------------

DATA_PATH = "silver_train.csv"

df = pd.read_csv(DATA_PATH)

# Remove rows without a label
df = df[df["intent"].notna()].copy()

# Keep only our defined intents
df = df[df["intent"].isin(INTENTS)].copy()


# ---------------------------------------------------------
# 3. Combine conversation context + customer message
# ---------------------------------------------------------

df["previous_context"] = df["previous_context"].fillna("")
df["customer_message"] = df["customer_message"].fillna("")

df["input_text"] = (
    "Previous context: "
    + df["previous_context"]
    + "\nCustomer message: "
    + df["customer_message"]
)


X = df["input_text"]
y = df["intent"]


# ---------------------------------------------------------
# 4. TF-IDF + Logistic Regression classifier
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------

print("Training intent classifier...")
print(f"Training examples: {len(df)}")
print(f"Number of intents: {len(y.unique())}")

model.fit(X, y)

print("\nTraining completed.")


# ---------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------

MODEL_PATH = "intent_classifier.joblib"

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")


# ---------------------------------------------------------
# 7. Simple prediction function
# ---------------------------------------------------------

def predict_intent(customer_message, previous_context=""):

    text = (
        "Previous context: "
        + previous_context
        + "\nCustomer message: "
        + customer_message
    )

    predicted_intent = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities)

    return predicted_intent, confidence


# ---------------------------------------------------------
# 8. Test the classifier
# ---------------------------------------------------------

if __name__ == "__main__":

    test_messages = [
        "My battery is draining really fast",
        "My iPhone keeps freezing and restarting",
        "I cannot connect to WiFi",
        "Why can't I install the latest iOS update?",
        "My iMessage is not working",
        "I cannot access my iCloud account",
        "My screen is completely unresponsive",
        "Why was I charged for this subscription?",
    ]

    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    for message in test_messages:

        intent, confidence = predict_intent(message)

        print(f"\nMessage: {message}")
        print(f"Intent: {intent}")
        print(f"Confidence: {confidence:.2%}")
"""
Hiver AI Support Agent
Evaluation Harness

Evaluates:
1. Intent classification
2. Majority baseline
3. Simple keyword baseline
4. TF-IDF + Logistic Regression model
5. Confidence distribution
6. Escalation decision
7. Top model mistakes

Important:
- The 200-row evaluation set contains:
    18 HUMAN_REVIEWED examples
    182 PROPOSED examples
- Proposed labels are useful for development but are NOT definitive
  human ground truth.
"""

import os
import sys
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from src.intent_classifier import predict_intent
from src.escalation import decide_escalation


# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------

GOLDEN_FILE = os.path.join(
    PROJECT_ROOT,
    "golden_set.csv"
)

RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation_results.csv"
)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def safe_text(value):
    """
    Convert NaN/None/non-string values into clean text.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def combined_text(row):
    """
    Combine previous context and current customer message.

    This mirrors the classifier's use of conversation context.
    """
    previous = safe_text(row.get("previous_context", ""))
    current = safe_text(row.get("customer_message", ""))

    if previous and current:
        return previous + " " + current

    return current or previous


# -------------------------------------------------------------------
# Simple keyword baseline
# -------------------------------------------------------------------

def keyword_baseline(text):
    """
    Very simple rule-based intent classifier.

    This is intentionally simple because it serves as a baseline,
    not as the main system.
    """

    text = safe_text(text).lower()

    # Battery
    if any(word in text for word in [
        "battery",
        "charging",
        "charge",
        "drain",
        "battery life",
        "power"
    ]):
        return "battery_power"

    # Software updates
    if any(word in text for word in [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "update",
        "updating",
        "upgrade"
    ]):
        return "software_update"

    # Connectivity
    if any(word in text for word in [
        "wifi",
        "wi-fi",
        "cellular",
        "network",
        "signal",
        "bluetooth",
        "no service",
        "sim"
    ]):
        return "connectivity_network"

    # Messaging / calls
    if any(word in text for word in [
        "imessage",
        "i message",
        "facetime",
        "text message",
        "sms",
        "can't call",
        "cannot call",
        "phone call",
        "calling",
        "calls"
    ]):
        return "messaging_calls"

    # iCloud / account
    if any(word in text for word in [
        "icloud",
        "apple id",
        "appleid",
        "account",
        "password",
        "sign in",
        "login",
        "logged in"
    ]):
        return "account_icloud"

    # Hardware / repair
    if any(word in text for word in [
        "broken",
        "cracked",
        "screen",
        "repair",
        "replacement",
        "warranty",
        "damaged",
        "damage"
    ]):
        return "hardware_repair"

    # Payment / billing
    if any(word in text for word in [
        "charged",
        "charge",
        "billing",
        "payment",
        "refund",
        "subscription",
        "money"
    ]):
        return "payments_billing"

    # Device performance
    if any(word in text for word in [
        "slow",
        "freezing",
        "freeze",
        "frozen",
        "restart",
        "restarting",
        "reboot",
        "crash",
        "crashing",
        "overheating",
        "lag"
    ]):
        return "device_performance"

    # Apps / media
    if any(word in text for word in [
        "app",
        "apps",
        "itunes",
        "music",
        "camera",
        "photo",
        "photos",
        "video",
        "apple tv",
        "app store"
    ]):
        return "apps_media"

    # Settings / features
    if any(word in text for word in [
        "keyboard",
        "display",
        "brightness",
        "accessibility",
        "control center",
        "setting",
        "settings",
        "feature",
        "lock screen"
    ]):
        return "settings_features"

    # Feedback
    if any(word in text for word in [
        "feature request",
        "suggestion",
        "please add",
        "would like",
        "feedback"
    ]):
        return "feedback_request"

    return "other_unclear"


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("FINAL EVALUATION HARNESS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load evaluation set
    # ---------------------------------------------------------------

    print("\nLoading golden evaluation set...")

    if not os.path.exists(GOLDEN_FILE):
        print("ERROR: golden_set.csv was not found.")
        print(f"Expected location: {GOLDEN_FILE}")
        return

    df = pd.read_csv(GOLDEN_FILE)

    print(f"Loaded {len(df)} examples")

    # ---------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------

    required_columns = [
        "customer_tweet_id",
        "previous_context",
        "customer_message",
        "intent",
        "escalate",
        "label_reason",
        "review_status",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing required columns:")
        for col in missing_columns:
            print(f"  - {col}")
        return

    # ---------------------------------------------------------------
    # Normalize review status
    # ---------------------------------------------------------------

    df["review_status"] = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    human_reviewed = (
        df["review_status"] == "HUMAN_REVIEWED"
    ).sum()

    proposed = (
        df["review_status"] == "PROPOSED"
    ).sum()

    # ---------------------------------------------------------------
    # Label status
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL STATUS")
    print("=" * 70)

    print(f"Total examples:       {len(df)}")
    print(f"Human-reviewed:       {human_reviewed}")
    print(f"Proposed labels:      {proposed}")

    if proposed > 0:
        print(
            "\nNOTE: Some labels are still PROPOSED. "
            "These are development labels and should not be presented "
            "as definitive human ground truth."
        )
    else:
        print("\nAll examples have been human-reviewed.")

    # ---------------------------------------------------------------
    # Evaluation label distribution
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION LABEL DISTRIBUTION")
    print("=" * 70)

    print(
        df["intent"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Prepare text
    # ---------------------------------------------------------------

    df["evaluation_text"] = df.apply(
        combined_text,
        axis=1
    )

    y_true = (
        df["intent"]
        .fillna("other_unclear")
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------------
    # BASELINE 1 — MAJORITY CLASS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE 1 — MAJORITY CLASS")
    print("=" * 70)

    majority_class = y_true.value_counts().idxmax()

    majority_predictions = [
        majority_class
        for _ in range(len(df))
    ]

    majority_accuracy = accuracy_score(
        y_true,
        majority_predictions
    )

    print(f"Majority class: {majority_class}")
    print(
        f"Accuracy: {majority_accuracy:.2%}"
    )

    # ---------------------------------------------------------------
    # BASELINE 2 — KEYWORD RULES
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE 2 — SIMPLE KEYWORD RULES")
    print("=" * 70)

    keyword_predictions = [
        keyword_baseline(text)
        for text in df["evaluation_text"]
    ]

    keyword_accuracy = accuracy_score(
        y_true,
        keyword_predictions
    )

    print(
        f"Accuracy: {keyword_accuracy:.2%}"
    )

    # ---------------------------------------------------------------
    # MAIN MODEL — TF-IDF + LOGISTIC REGRESSION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MAIN MODEL — TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    model_predictions = []
    model_confidences = []

    for text in df["evaluation_text"]:

        try:

            result = predict_intent(text)

            # -------------------------------------------------------
            # Handle different possible return formats
            # -------------------------------------------------------

            if isinstance(result, tuple):

                if len(result) >= 2:
                    prediction = result[0]
                    confidence = result[1]
                else:
                    prediction = result[0]
                    confidence = 0.0

            elif isinstance(result, dict):

                prediction = (
                    result.get("intent")
                    or result.get("prediction")
                    or result.get("label")
                    or "other_unclear"
                )

                confidence = (
                    result.get("confidence")
                    or result.get("probability")
                    or 0.0
                )

            else:

                prediction = str(result)
                confidence = 0.0

            prediction = str(prediction)

            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

        except Exception as error:

            print(
                "\nWARNING: prediction failed:"
            )
            print(error)

            prediction = "other_unclear"
            confidence = 0.0

        model_predictions.append(prediction)
        model_confidences.append(confidence)

    model_accuracy = accuracy_score(
        y_true,
        model_predictions
    )

    print(
        f"Accuracy: {model_accuracy:.2%}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            model_predictions,
            zero_division=0
        )
    )

    # ---------------------------------------------------------------
    # HEADLINE COMPARISON
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("HEADLINE COMPARISON")
    print("=" * 70)

    print(
        f"Majority baseline: {majority_accuracy:.2%}"
    )

    print(
        f"Keyword baseline:  {keyword_accuracy:.2%}"
    )

    print(
        f"Main model:        {model_accuracy:.2%}"
    )

    print(
        f"\nImprovement vs majority: "
        f"{(model_accuracy - majority_accuracy) * 100:+.2f} percentage points"
    )

    print(
        f"Improvement vs keywords: "
        f"{(model_accuracy - keyword_accuracy) * 100:+.2f} percentage points"
    )

    # ---------------------------------------------------------------
    # CONFIDENCE ANALYSIS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    confidence_series = pd.Series(
        model_confidences
    )

    print(
        f"Mean confidence:   "
        f"{confidence_series.mean():.4f}"
    )

    print(
        f"Median confidence: "
        f"{confidence_series.median():.4f}"
    )

    print(
        f"Minimum confidence:"
        f"{confidence_series.min():.4f}"
    )

    print(
        f"Maximum confidence:"
        f"{confidence_series.max():.4f}"
    )

    for threshold in [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]:

        coverage = (
            confidence_series >= threshold
        ).mean()

        print(
            f"Confidence >= {threshold:.2f}: "
            f"{coverage:.2%}"
        )

    # ---------------------------------------------------------------
    # TOP MODEL MISTAKES
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP MODEL MISTAKES")
    print("=" * 70)

    mistakes = df[
        y_true.values !=
        pd.Series(model_predictions).values
    ].copy()

    print(
        f"Total mistakes: {len(mistakes)}"
    )

    # Add predictions/confidence
    mistakes["predicted_intent"] = [
        model_predictions[i]
        for i in mistakes.index
    ]

    mistakes["prediction_confidence"] = [
        model_confidences[i]
        for i in mistakes.index
    ]

    # Show up to 15 mistakes
    for display_number, (idx, row) in enumerate(
        mistakes.head(15).iterrows(),
        start=1
    ):

        print(
            f"\n--- Mistake {display_number} ---"
        )

        print(
            "Customer:"
        )

        print(
            safe_text(row["customer_message"])
        )

        print(
            "\nPrevious context:"
        )

        print(
            safe_text(row["previous_context"])
        )

        print(
            f"\nExpected:   "
            f"{row['intent']}"
        )

        print(
            f"Predicted:  "
            f"{row['predicted_intent']}"
        )

        print(
            f"Confidence: "
            f"{row['prediction_confidence']:.2%}"
        )

    # ---------------------------------------------------------------
    # ESCALATION EVALUATION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ESCALATION EVALUATION")
    print("=" * 70)

    actual_escalation = []
    predicted_escalation = []
    escalation_reasons = []

    for i, row in df.iterrows():

        intent = model_predictions[i]
        confidence = model_confidences[i]

        message = safe_text(
            row["customer_message"]
        )

        try:

            result = decide_escalation(
                intent,
                confidence,
                message
            )

            # -------------------------------------------------------
            # Handle both:
            # (bool, reason)
            # and
            # (reason, bool)
            # -------------------------------------------------------

            if isinstance(result, tuple) and len(result) >= 2:

                first = result[0]
                second = result[1]

                if isinstance(first, bool):

                    should_escalate = first
                    reason = str(second)

                elif isinstance(second, bool):

                    reason = str(first)
                    should_escalate = second

                else:

                    should_escalate = bool(first)
                    reason = str(second)

            elif isinstance(result, dict):

                should_escalate = bool(
                    result.get(
                        "escalate",
                        result.get(
                            "should_escalate",
                            False
                        )
                    )
                )

                reason = str(
                    result.get(
                        "reason",
                        ""
                    )
                )

            else:

                should_escalate = bool(result)
                reason = ""

        except Exception as error:

            should_escalate = False
            reason = (
                f"Escalation evaluation error: {error}"
            )

        predicted_escalation.append(
            should_escalate
        )

        escalation_reasons.append(
            reason
        )

        # -----------------------------------------------------------
        # Normalize expected escalation value
        # -----------------------------------------------------------

        expected = row["escalate"]

        if isinstance(expected, str):

            expected_clean = (
                expected
                .strip()
                .lower()
            )

            expected_bool = (
                expected_clean
                in ["yes", "true", "1"]
            )

        else:

            expected_bool = bool(expected)

        actual_escalation.append(
            expected_bool
        )

    escalation_accuracy = accuracy_score(
        actual_escalation,
        predicted_escalation
    )

    print(
        f"Escalation accuracy: "
        f"{escalation_accuracy:.2%}"
    )

    print("\nActual distribution:")

    print(
        pd.Series(
            actual_escalation
        ).value_counts().to_string()
    )

    print("\nPredicted distribution:")

    print(
        pd.Series(
            predicted_escalation
        ).value_counts().to_string()
    )

    print("\nClassification report:")

    print(
        classification_report(
            actual_escalation,
            predicted_escalation,
            target_names=["no", "yes"],
            zero_division=0
        )
    )

    # ---------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------

    df["predicted_intent"] = model_predictions

    df["prediction_confidence"] = (
        model_confidences
    )

    df["keyword_prediction"] = (
        keyword_predictions
    )

    df["majority_prediction"] = (
        majority_predictions
    )

    df["actual_escalation"] = (
        actual_escalation
    )

    df["predicted_escalation"] = (
        predicted_escalation
    )

    df["escalation_reason"] = (
        escalation_reasons
    )

    # Remove temporary column
    if "evaluation_text" in df.columns:
        df = df.drop(
            columns=["evaluation_text"]
        )

    df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        RESULTS_FILE
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
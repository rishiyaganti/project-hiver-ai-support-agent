"""
Hiver AI Support Agent
Reply Quality Evaluation Harness

Evaluates whether generated replies are:
1. Grounded in historical AppleSupport responses
2. Relevant to the customer's issue
3. Actionable
4. Consistent with the historical support style
5. Free from obvious unsupported claims

This script also prepares a dataset for LLM-as-judge evaluation.

Important:
- The 200 evaluation examples are:
    18 HUMAN_REVIEWED
    182 PROPOSED
- Proposed labels must not be presented as definitive human ground truth.
"""

import os
import sys
import re
import json
import math
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# IMPORT EXISTING SYSTEM
# =====================================================================

from src.intent_classifier import predict_intent
from src.retrieval import retrieve_similar_cases
from src.generator import generate_reply
from src.escalation import decide_escalation


# =====================================================================
# FILES
# =====================================================================

GOLDEN_FILE = os.path.join(
    PROJECT_ROOT,
    "golden_set.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "reply_evaluation_results.csv"
)


# =====================================================================
# HELPERS
# =====================================================================

def safe_text(value):
    """Convert missing/non-string values to clean text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def extract_prediction(result):
    """
    Robustly extract intent and confidence from predict_intent().
    """

    if isinstance(result, tuple):

        if len(result) >= 2:
            return str(result[0]), float(result[1])

        return str(result[0]), 0.0

    if isinstance(result, dict):

        intent = (
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

        return str(intent), float(confidence)

    return str(result), 0.0


def extract_retrieval_results(result):
    """
    Normalize the output of retrieve_similar_cases().

    Different implementations may return:
    - list
    - tuple
    - dict
    """

    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, tuple):

        for item in result:
            if isinstance(item, list):
                return item

        return list(result)

    if isinstance(result, dict):

        for key in [
            "results",
            "cases",
            "matches",
            "retrieved_cases"
        ]:
            if key in result:
                value = result[key]

                if isinstance(value, list):
                    return value

        return [result]

    return [result]


def extract_reply(result):
    """
    Normalize generate_reply() output.

    Expected possibilities:
        reply
        (reply, evidence)
        {"reply": "...", "evidence": [...]}
    """

    reply = ""
    evidence = []

    if isinstance(result, str):
        reply = result

    elif isinstance(result, tuple):

        if len(result) >= 1:
            reply = safe_text(result[0])

        if len(result) >= 2:
            evidence = result[1]

    elif isinstance(result, dict):

        reply = (
            result.get("reply")
            or result.get("response")
            or result.get("text")
            or ""
        )

        evidence = (
            result.get("evidence")
            or result.get("sources")
            or []
        )

    else:
        reply = safe_text(result)

    if not isinstance(evidence, list):
        evidence = [evidence]

    return safe_text(reply), evidence


def extract_case_text(case):
    """
    Convert a retrieved case into searchable text.
    """

    if isinstance(case, dict):

        parts = [
            case.get("customer_message", ""),
            case.get("previous_context", ""),
            case.get("apple_reply", ""),
            case.get("reply", ""),
            case.get("text", "")
        ]

        return " ".join(
            safe_text(x)
            for x in parts
            if safe_text(x)
        )

    return safe_text(case)


def extract_case_reply(case):
    """
    Extract historical AppleSupport reply from a retrieved case.
    """

    if isinstance(case, dict):

        for key in [
            "apple_reply",
            "reply",
            "response",
            "support_reply"
        ]:
            value = case.get(key)

            if value:
                return safe_text(value)

    return ""


# =====================================================================
# AUTOMATED REPLY METRICS
# =====================================================================

def cosine_text_similarity(text_a, text_b):
    """
    TF-IDF cosine similarity.

    This is not a correctness metric.
    It is used only as a lightweight measure of lexical/style
    similarity between the generated reply and historical replies.
    """

    text_a = safe_text(text_a)
    text_b = safe_text(text_b)

    if not text_a or not text_b:
        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform([
            text_a,
            text_b
        ])

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(score)

    except Exception:
        return 0.0


def reply_has_dm_request(reply):
    """Check whether reply asks customer to DM/contact support."""

    text = safe_text(reply).lower()

    patterns = [
        "dm us",
        "send us a dm",
        "direct message",
        "message us",
        "private message",
        "contact us",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def reply_asks_for_device(reply):
    """Check whether reply asks for device information."""

    text = safe_text(reply).lower()

    patterns = [
        "device model",
        "model number",
        "iphone model",
        "ipad model",
        "mac model",
        "which device",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def reply_asks_for_os(reply):
    """Check whether reply asks for software/OS information."""

    text = safe_text(reply).lower()

    patterns = [
        "software version",
        "ios version",
        "version are you using",
        "which version",
        "os version",
        "macos version",
        "watchos version",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def reply_contains_troubleshooting(reply):
    """
    Detect whether the response contains an actionable troubleshooting
    instruction.
    """

    text = safe_text(reply).lower()

    patterns = [
        "restart",
        "reboot",
        "update",
        "check",
        "try",
        "settings",
        "turn off",
        "turn on",
        "reset",
        "install",
        "remove",
        "reconnect",
        "sign in",
        "sign out",
        "send us",
        "dm us",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def reply_has_unsupported_promise(reply):
    """
    Detect obvious unsupported promises.

    This is intentionally conservative.
    """

    text = safe_text(reply).lower()

    patterns = [
        "we guarantee",
        "guaranteed",
        "definitely fix",
        "will definitely fix",
        "we will refund",
        "refund is guaranteed",
        "your replacement is confirmed",
        "we have fixed it",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def reply_length_score(reply):
    """
    Simple sanity score for reply length.

    Very short replies are often unhelpful.
    Extremely long replies are undesirable for Twitter support.
    """

    length = len(
        safe_text(reply).split()
    )

    if length == 0:
        return 0.0

    if 8 <= length <= 80:
        return 1.0

    if 5 <= length <= 120:
        return 0.75

    return 0.5


# =====================================================================
# AUTOMATED GROUNDING SCORE
# =====================================================================

def calculate_grounding_score(
    customer_message,
    reply,
    retrieved_cases
):
    """
    Estimate whether the generated reply is grounded in retrieved
    historical support behavior.

    This is an automated proxy, NOT a substitute for LLM judging.
    """

    customer_message = safe_text(
        customer_message
    )

    reply = safe_text(reply)

    if not reply:
        return 0.0

    if not retrieved_cases:
        return 0.0

    historical_replies = []

    for case in retrieved_cases:

        historical_reply = extract_case_reply(
            case
        )

        if historical_reply:
            historical_replies.append(
                historical_reply
            )

    if not historical_replies:
        return 0.0

    # ---------------------------------------------------------------
    # Similarity between generated response and best historical reply
    # ---------------------------------------------------------------

    similarities = [
        cosine_text_similarity(
            reply,
            historical_reply
        )
        for historical_reply in historical_replies
    ]

    best_similarity = max(
        similarities
    )

    # ---------------------------------------------------------------
    # Check whether the reply uses common support patterns
    # ---------------------------------------------------------------

    action_score = 0.0

    if reply_has_dm_request(reply):
        action_score += 0.25

    if reply_asks_for_device(reply):
        action_score += 0.15

    if reply_asks_for_os(reply):
        action_score += 0.15

    if reply_contains_troubleshooting(reply):
        action_score += 0.20

    # ---------------------------------------------------------------
    # Combine lexical evidence + action evidence
    # ---------------------------------------------------------------

    score = (
        0.65 * best_similarity
        + 0.35 * min(action_score / 0.75, 1.0)
    )

    return round(
        min(score, 1.0),
        4
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n" + "=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("REPLY QUALITY EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load golden set
    # ---------------------------------------------------------------

    print("\nLoading evaluation set...")

    if not os.path.exists(GOLDEN_FILE):

        print(
            f"ERROR: Could not find {GOLDEN_FILE}"
        )

        return

    df = pd.read_csv(
        GOLDEN_FILE
    )

    print(
        f"Loaded {len(df)} examples"
    )

    # ---------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------

    df["review_status"] = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    human_count = (
        df["review_status"]
        == "HUMAN_REVIEWED"
    ).sum()

    proposed_count = (
        df["review_status"]
        == "PROPOSED"
    ).sum()

    print(
        f"Human-reviewed: {human_count}"
    )

    print(
        f"Proposed:       {proposed_count}"
    )

    # ---------------------------------------------------------------
    # Result containers
    # ---------------------------------------------------------------

    results = []

    # ---------------------------------------------------------------
    # Evaluate each example
    # ---------------------------------------------------------------

    for position, row in df.iterrows():

        print(
            f"\n[{position + 1}/{len(df)}] Evaluating..."
        )

        customer_message = safe_text(
            row.get("customer_message", "")
        )

        previous_context = safe_text(
            row.get("previous_context", "")
        )

        # ===========================================================
        # INTENT
        # ===========================================================

        try:

            prediction_result = predict_intent(
                (
                    previous_context
                    + " "
                    + customer_message
                ).strip()
            )

            predicted_intent, confidence = (
                extract_prediction(
                    prediction_result
                )
            )

        except Exception as error:

            print(
                f"Intent error: {error}"
            )

            predicted_intent = "other_unclear"
            confidence = 0.0

        # ===========================================================
        # RETRIEVAL
        # ===========================================================

        try:

            retrieval_result = (
                retrieve_similar_cases(
                    customer_message,
                    previous_context,
                    top_k=3
                )
            )

            retrieved_cases = (
                extract_retrieval_results(
                    retrieval_result
                )
            )

        except Exception as error:

            print(
                f"Retrieval error: {error}"
            )

            retrieved_cases = []

        # ===========================================================
        # GENERATE REPLY
        # ===========================================================

        try:

            generator_result = (
                generate_reply(
                    customer_message,
                    previous_context,
                    predicted_intent,
                    top_k=3
                )
            )

            generated_reply, evidence = (
                extract_reply(
                    generator_result
                )
            )

        except Exception as error:

            print(
                f"Generator error: {error}"
            )

            generated_reply = ""
            evidence = []

        # ===========================================================
        # ESCALATION
        # ===========================================================

        try:

            escalation_result = (
                decide_escalation(
                    predicted_intent,
                    confidence,
                    customer_message
                )
            )

            if (
                isinstance(
                    escalation_result,
                    tuple
                )
                and len(escalation_result) >= 2
            ):

                first = escalation_result[0]
                second = escalation_result[1]

                if isinstance(first, bool):

                    predicted_escalation = first
                    escalation_reason = str(
                        second
                    )

                elif isinstance(second, bool):

                    escalation_reason = str(
                        first
                    )

                    predicted_escalation = second

                else:

                    predicted_escalation = bool(
                        first
                    )

                    escalation_reason = str(
                        second
                    )

            else:

                predicted_escalation = bool(
                    escalation_result
                )

                escalation_reason = ""

        except Exception as error:

            print(
                f"Escalation error: {error}"
            )

            predicted_escalation = False
            escalation_reason = ""

        # ===========================================================
        # AUTOMATED REPLY METRICS
        # ===========================================================

        grounding_score = (
            calculate_grounding_score(
                customer_message,
                generated_reply,
                retrieved_cases
            )
        )

        troubleshooting = (
            reply_contains_troubleshooting(
                generated_reply
            )
        )

        dm_request = (
            reply_has_dm_request(
                generated_reply
            )
        )

        device_request = (
            reply_asks_for_device(
                generated_reply
            )
        )

        os_request = (
            reply_asks_for_os(
                generated_reply
            )
        )

        unsupported = (
            reply_has_unsupported_promise(
                generated_reply
            )
        )

        length_score = (
            reply_length_score(
                generated_reply
            )
        )

        # -----------------------------------------------------------
        # Historical reply similarity
        # -----------------------------------------------------------

        historical_replies = []

        for case in retrieved_cases:

            historical_reply = (
                extract_case_reply(
                    case
                )
            )

            if historical_reply:
                historical_replies.append(
                    historical_reply
                )

        if historical_replies:

            historical_similarity = max(
                cosine_text_similarity(
                    generated_reply,
                    historical_reply
                )
                for historical_reply
                in historical_replies
            )

        else:

            historical_similarity = 0.0

        # -----------------------------------------------------------
        # Evidence count
        # -----------------------------------------------------------

        evidence_count = len(
            retrieved_cases
        )

        # -----------------------------------------------------------
        # Save row
        # -----------------------------------------------------------

        result_row = {

            "customer_tweet_id":
                row.get(
                    "customer_tweet_id",
                    ""
                ),

            "customer_message":
                customer_message,

            "previous_context":
                previous_context,

            "human_label_status":
                row.get(
                    "review_status",
                    ""
                ),

            "expected_intent":
                row.get(
                    "intent",
                    ""
                ),

            "predicted_intent":
                predicted_intent,

            "intent_confidence":
                confidence,

            "expected_escalation":
                row.get(
                    "escalate",
                    ""
                ),

            "predicted_escalation":
                predicted_escalation,

            "escalation_reason":
                escalation_reason,

            "generated_reply":
                generated_reply,

            "retrieved_case_count":
                evidence_count,

            "grounding_score":
                grounding_score,

            "historical_reply_similarity":
                historical_similarity,

            "reply_has_dm_request":
                dm_request,

            "reply_asks_device":
                device_request,

            "reply_asks_os":
                os_request,

            "reply_contains_troubleshooting":
                troubleshooting,

            "reply_has_unsupported_promise":
                unsupported,

            "reply_length_score":
                length_score,
        }

        results.append(
            result_row
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ===============================================================
    # SUMMARY
    # ===============================================================

    print("\n" + "=" * 70)
    print("REPLY EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Examples evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Average grounding score: "
        f"{results_df['grounding_score'].mean():.3f}"
    )

    print(
        f"Average historical similarity: "
        f"{results_df['historical_reply_similarity'].mean():.3f}"
    )

    print(
        f"Replies with DM request: "
        f"{results_df['reply_has_dm_request'].mean():.2%}"
    )

    print(
        f"Replies asking for device: "
        f"{results_df['reply_asks_device'].mean():.2%}"
    )

    print(
        f"Replies asking for OS/version: "
        f"{results_df['reply_asks_os'].mean():.2%}"
    )

    print(
        f"Replies containing troubleshooting: "
        f"{results_df['reply_contains_troubleshooting'].mean():.2%}"
    )

    print(
        f"Replies with obvious unsupported promises: "
        f"{results_df['reply_has_unsupported_promise'].mean():.2%}"
    )

    print(
        f"Average reply length score: "
        f"{results_df['reply_length_score'].mean():.3f}"
    )

    # ===============================================================
    # HUMAN-REVIEWED SUBSET
    # ===============================================================

    human_df = results_df[
        results_df["human_label_status"]
        .astype(str)
        .str.upper()
        == "HUMAN_REVIEWED"
    ]

    print("\n" + "=" * 70)
    print("HUMAN-REVIEWED SUBSET")
    print("=" * 70)

    print(
        f"Human-reviewed examples: "
        f"{len(human_df)}"
    )

    if len(human_df) > 0:

        print(
            f"Mean grounding score: "
            f"{human_df['grounding_score'].mean():.3f}"
        )

        print(
            f"Mean historical similarity: "
            f"{human_df['historical_reply_similarity'].mean():.3f}"
        )

    # ===============================================================
    # LLM JUDGE QUEUE
    # ===============================================================

    judge_queue_file = os.path.join(
        PROJECT_ROOT,
        "llm_judge_queue.csv"
    )

    judge_columns = [
        "customer_tweet_id",
        "customer_message",
        "previous_context",
        "expected_intent",
        "predicted_intent",
        "generated_reply",
        "human_label_status",
        "retrieved_case_count",
        "grounding_score",
        "historical_reply_similarity",
    ]

    results_df[
        judge_columns
    ].to_csv(
        judge_queue_file,
        index=False
    )

    # ===============================================================
    # SAMPLE OUTPUT
    # ===============================================================

    print("\n" + "=" * 70)
    print("SAMPLE GENERATED REPLIES")
    print("=" * 70)

    for _, row in results_df.head(5).iterrows():

        print("\nCustomer:")
        print(
            row["customer_message"]
        )

        print("\nPredicted intent:")
        print(
            row["predicted_intent"]
        )

        print("\nGenerated reply:")
        print(
            row["generated_reply"]
        )

        print(
            f"\nGrounding score: "
            f"{row['grounding_score']:.3f}"
        )

        print("-" * 70)

    # ===============================================================
    # COMPLETE
    # ===============================================================

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"Reply evaluation:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nLLM judge queue:"
    )

    print(
        judge_queue_file
    )

    print("\n" + "=" * 70)
    print("REPLY EVALUATION COMPLETE")
    print("=" * 70)


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":
    main()
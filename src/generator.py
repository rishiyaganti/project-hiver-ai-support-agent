import sys
import os

# Allow Python to find retrieval.py in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import retrieve_similar_cases


# ============================================================
# REPLY GENERATOR
# ============================================================

def generate_reply(
    customer_message,
    previous_context="",
    intent=None,
    top_k=3
):
    """
    Generate a support reply using historical AppleSupport
    conversations.

    This version is deterministic and does not require
    an external LLM API.
    """

    # Retrieve similar historical conversations
    cases = retrieve_similar_cases(
        customer_message,
        previous_context,
        top_k=top_k
    )

    # --------------------------------------------------------
    # No evidence found
    # --------------------------------------------------------

    if not cases:

        reply = (
            "Thanks for reaching out. We'd like to help "
            "you look into this. Please send us a DM with "
            "some additional details about the issue."
        )

        return {
            "reply": reply,
            "evidence": []
        }

    # --------------------------------------------------------
    # Collect historical replies
    # --------------------------------------------------------

    replies = []

    for case in cases:

        apple_reply = case["apple_reply"]

        if apple_reply:
            replies.append(str(apple_reply))

    combined_replies = " ".join(replies).lower()


    # --------------------------------------------------------
    # Detect historical support patterns
    # --------------------------------------------------------

    asks_for_dm = (
        "dm" in combined_replies
        or "direct message" in combined_replies
    )

    asks_for_device = (
        "what type of device" in combined_replies
        or "which device" in combined_replies
        or "iphone model" in combined_replies
        or "which model" in combined_replies
        or "device are we working with" in combined_replies
    )

    asks_for_os = (
        "ios version" in combined_replies
        or "os version" in combined_replies
        or "version number" in combined_replies
        or "operating system" in combined_replies
    )

    provides_steps = (
        "steps" in combined_replies
        or "follow these" in combined_replies
        or "try" in combined_replies
        or "this page should help" in combined_replies
        or "article" in combined_replies
    )


    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    parts = []

    parts.append(
        "Thanks for reaching out. We'd like to help "
        "you get this sorted."
    )


    # --------------------------------------------------------
    # Ask for useful diagnostic information
    # --------------------------------------------------------

    detail_parts = []

    if asks_for_device:
        detail_parts.append("your device model")

    if asks_for_os:
        detail_parts.append("the OS version you're using")


    if len(detail_parts) == 2:

        parts.append(
            "Could you send us your "
            + " and ".join(detail_parts)
            + "?"
        )

    elif len(detail_parts) == 1:

        parts.append(
            "Could you send us "
            + detail_parts[0]
            + "?"
        )


    # --------------------------------------------------------
    # Historical DM pattern
    # --------------------------------------------------------

    if asks_for_dm:

        parts.append(
            "Please send us a DM with those details so "
            "we can take a closer look."
        )


    # --------------------------------------------------------
    # Historical troubleshooting pattern
    # --------------------------------------------------------

    elif provides_steps:

        parts.append(
            "We can also walk through some troubleshooting "
            "steps with you."
        )


    # --------------------------------------------------------
    # No specific historical pattern
    # --------------------------------------------------------

    else:

        parts.append(
            "Please send us a few more details about what "
            "you're seeing so we can investigate."
        )


    # Combine response
    reply = " ".join(parts)


    return {
        "reply": reply,
        "evidence": cases
    }


# ============================================================
# DISPLAY HISTORICAL EVIDENCE
# ============================================================

def print_evidence(evidence):

    print("\n" + "=" * 70)
    print("EVIDENCE USED")
    print("=" * 70)

    for i, case in enumerate(evidence, start=1):

        print(f"\nHistorical Case {i}")
        print("-" * 70)

        print(
            "Similarity: "
            + f"{case['similarity']:.2%}"
        )

        print(
            "Customer: "
            + str(case["customer_message"])
        )

        print(
            "Apple reply: "
            + str(case["apple_reply"])
        )


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    test_cases = [

        {
            "message": "My battery is draining really fast",
            "intent": "battery_power"
        },

        {
            "message": "My iPhone keeps freezing and restarting",
            "intent": "device_performance"
        },

        {
            "message": "I cannot connect to WiFi",
            "intent": "connectivity_network"
        },

        {
            "message": "My iMessage is not working",
            "intent": "messaging_calls"
        },

        {
            "message": "I cannot access my iCloud account",
            "intent": "account_icloud"
        },

    ]


    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for test in test_cases:

        print("\n\n")

        print("#" * 70)

        print(
            "CUSTOMER: "
            + test["message"]
        )

        print(
            "INTENT: "
            + test["intent"]
        )

        print("#" * 70)


        result = generate_reply(
            customer_message=test["message"],
            intent=test["intent"],
            top_k=3
        )


        # ----------------------------------------------------
        # Generated reply
        # ----------------------------------------------------

        print("\nGENERATED REPLY")

        print("-" * 70)

        print(result["reply"])


        # ----------------------------------------------------
        # Historical evidence
        # ----------------------------------------------------

        print_evidence(
            result["evidence"]
        )
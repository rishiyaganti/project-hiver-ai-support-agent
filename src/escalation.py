"""
Hiver AI Support Agent
Escalation Policy

Decision principle:

Escalation is based primarily on:
1. Explicit request for a human
2. Risk / sensitivity
3. Severity of the customer problem
4. Whether human intervention is likely required

Classifier confidence is only a supporting signal.
Low confidence alone should NOT automatically cause escalation.
"""

import re


def normalize_text(text):
    """Normalize text for reliable rule matching."""

    if text is None:
        return ""

    return str(text).strip().lower()


def decide_escalation(intent, confidence, message):
    """
    Decide whether the case should be escalated.

    Returns:
        (should_escalate: bool, reason: str)
    """

    text = normalize_text(message)

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    intent = normalize_text(intent)

    # ================================================================
    # 1. EXPLICIT HUMAN REQUEST
    # ================================================================

    human_patterns = [
        r"\bspeak to a human\b",
        r"\bspeak with a human\b",
        r"\btalk to a human\b",
        r"\btalk with a human\b",
        r"\bspeak to someone\b",
        r"\btalk to someone\b",
        r"\breal person\b",
        r"\bhuman agent\b",
        r"\bhuman support\b",
        r"\bconnect me to support\b",
        r"\bconnect me with support\b",
        r"\brepresentative\b",
        r"\bescalate\b",
        r"\bsupervisor\b",
        r"\bmanager\b",
    ]

    for pattern in human_patterns:
        if re.search(pattern, text):
            return (
                True,
                "Customer explicitly requested human assistance."
            )

    # ================================================================
    # 2. HIGH-RISK / SENSITIVE INTENTS
    # ================================================================

    if intent == "payments_billing":

        financial_patterns = [
            "charged twice",
            "charged me",
            "unauthorized",
            "fraud",
            "fraudulent",
            "stolen",
            "refund",
            "wrong charge",
            "incorrect charge",
            "payment dispute",
            "billing dispute",
            "money taken",
            "subscription charge",
        ]

        if any(pattern in text for pattern in financial_patterns):
            return (
                True,
                "Financial or billing issue may require account-level investigation or action."
            )

        # Routine payment questions can remain automated.
        return (
            False,
            "Routine billing/payment question appears suitable for automated first-line support."
        )

    # ================================================================
    # 3. ACCOUNT / ICLOUD
    # ================================================================

    if intent == "account_icloud":

        account_risk_patterns = [
            "account locked",
            "locked out",
            "cannot access my account",
            "can't access my account",
            "cannot access account",
            "can't access account",
            "forgot password",
            "password reset",
            "account hacked",
            "hacked account",
            "stolen account",
            "security",
            "verification",
            "verify my identity",
            "identity verification",
            "two factor",
            "two-factor",
            "2fa",
            "unauthorized access",
        ]

        if any(pattern in text for pattern in account_risk_patterns):
            return (
                True,
                "Account-security or access issue may require human verification."
            )

        return (
            False,
            "Routine account/iCloud question appears suitable for automated first-line support."
        )

    # ================================================================
    # 4. HARDWARE / REPAIR
    # ================================================================

    if intent == "hardware_repair":

        severe_hardware_patterns = [
            "broken",
            "cracked",
            "shattered",
            "physical damage",
            "water damage",
            "liquid damage",
            "won't turn on",
            "wont turn on",
            "doesn't turn on",
            "doesnt turn on",
            "dead phone",
            "dead iphone",
            "dead ipad",
            "dead mac",
            "replacement",
            "repair",
            "service",
            "warranty claim",
            "warranty dispute",
        ]

        if any(pattern in text for pattern in severe_hardware_patterns):
            return (
                True,
                "Hardware/service issue may require physical inspection, repair, replacement, or warranty intervention."
            )

        return (
            False,
            "Hardware question does not clearly require immediate human intervention."
        )

    # ================================================================
    # 5. SEVERE DEVICE PERFORMANCE
    # ================================================================

    severe_device_patterns = [
        "completely frozen",
        "totally frozen",
        "phone is frozen",
        "iphone is frozen",
        "ipad is frozen",
        "mac is frozen",
        "computer is frozen",
        "cannot use my phone",
        "can't use my phone",
        "cannot use the phone",
        "can't use the phone",
        "cannot use my iphone",
        "can't use my iphone",
        "phone unusable",
        "iphone unusable",
        "completely broken",
        "phone is dead",
        "iphone is dead",
        "ipad is dead",
        "mac is dead",
        "won't turn on",
        "wont turn on",
        "doesn't turn on",
        "doesnt turn on",
        "lost data",
        "data loss",
    ]

    for pattern in severe_device_patterns:
        if pattern in text:
            return (
                True,
                "Device appears severely impaired or unusable and may require human troubleshooting."
            )

    # ================================================================
    # 6. REPEATED RESTART / REBOOT LOOP
    # ================================================================

    restart_patterns = [
        "keeps restarting",
        "keeps rebooting",
        "keeps resetting",
        "continually restarts",
        "continually restarting",
        "continually freezes and restarts",
        "constantly restarts",
        "constantly restarting",
        "random resets",
        "randomly resets",
        "randomly restarting",
        "randomly reboots",
        "reboot loop",
        "restart loop",
    ]

    if any(pattern in text for pattern in restart_patterns):

        return (
            True,
            "Repeated restart/reset behavior indicates a potentially severe device stability problem."
        )

    # ================================================================
    # 7. WIDESPREAD DEVICE FAILURE
    # ================================================================

    widespread_patterns = [
        "everything is broken",
        "nothing works",
        "nothing is working",
        "all my apps",
        "most of my apps",
        "apps keep crashing",
        "apps keep shutting down",
        "multiple apps",
        "phone keeps crashing",
        "iphone keeps crashing",
        "system keeps crashing",
    ]

    if intent == "device_performance":

        if any(pattern in text for pattern in widespread_patterns):
            return (
                True,
                "Multiple system components appear affected, indicating a potentially severe device stability issue."
            )

    # ================================================================
    # 8. COMMUNICATION FAILURE
    # ================================================================

    if intent == "messaging_calls":

        severe_call_patterns = [
            "cannot call",
            "can't call",
            "cannot make calls",
            "can't make calls",
            "unable to call",
            "unable to make calls",
            "no calls",
            "calls don't work",
            "calls do not work",
            "can't receive calls",
            "cannot receive calls",
        ]

        if any(pattern in text for pattern in severe_call_patterns):
            return (
                True,
                "Customer reports a fundamental calling failure that may require human troubleshooting."
            )

    # ================================================================
    # 9. DATA LOSS
    # ================================================================

    data_loss_patterns = [
        "deleted all my",
        "lost all my",
        "lost my photos",
        "lost my pictures",
        "lost my files",
        "lost my data",
        "all my data is gone",
        "data is gone",
        "everything disappeared",
    ]

    if any(pattern in text for pattern in data_loss_patterns):

        return (
            True,
            "Potential data loss requires careful human investigation."
        )

    # ================================================================
    # 10. VERY LOW CONFIDENCE + UNCLEAR INTENT
    # ================================================================

    if intent == "other_unclear" and confidence < 0.30:

        return (
            True,
            "The customer's issue is unclear enough that automated handling may be unreliable."
        )

    # ================================================================
    # 11. NORMAL FIRST-LINE SUPPORT
    # ================================================================

    return (
        False,
        "Issue appears suitable for automated first-line support."
    )


# ====================================================================
# QUICK TESTS
# ====================================================================

if __name__ == "__main__":

    tests = [

        (
            "battery_power",
            0.95,
            "My battery drains quickly after the update."
        ),

        (
            "device_performance",
            0.46,
            "My iPhone continually freezes and restarts."
        ),

        (
            "connectivity_network",
            0.90,
            "My WiFi keeps disconnecting."
        ),

        (
            "account_icloud",
            0.27,
            "iCloud is not syncing properly."
        ),

        (
            "payments_billing",
            0.80,
            "I was charged twice for the same purchase."
        ),

        (
            "messaging_calls",
            0.47,
            "I cannot make calls from my iPhone."
        ),

        (
            "settings_features",
            0.20,
            "Why is the keyboard showing a weird symbol?"
        ),

        (
            "device_performance",
            0.30,
            "My phone keeps rebooting randomly."
        ),

        (
            "account_icloud",
            0.80,
            "My Apple ID is locked and I cannot access it."
        ),

        (
            "other_unclear",
            0.20,
            "I need help."
        ),

        (
            "other_unclear",
            0.20,
            "Please connect me to a real person."
        ),
    ]

    print("\n" + "=" * 70)
    print("ESCALATION POLICY TESTS")
    print("=" * 70)

    for intent, confidence, message in tests:

        escalate, reason = decide_escalation(
            intent,
            confidence,
            message
        )

        decision = "HUMAN" if escalate else "AUTO"

        print("\nMessage:")
        print(message)

        print(f"Intent:     {intent}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Decision:   {decision}")
        print(f"Reason:     {reason}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
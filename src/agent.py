"""
Hiver AI Support Agent
End-to-end demo:
1. Intent classification
2. Historical-case retrieval
3. Reply generation
4. Auto-handle vs human escalation
"""

from intent_classifier import predict_intent
from generator import generate_reply
from escalation import decide_escalation


def run_agent(customer_message, previous_context=""):
    # 1. Classify intent
    intent, confidence = predict_intent(
        customer_message,
        previous_context
    )

    # 2. Generate a historically grounded reply
    reply, evidence = generate_reply(
        customer_message,
        previous_context,
        intent=intent,
        top_k=3
    )

    # 3. Decide whether a human should handle it
    escalate, escalation_reason = decide_escalation(
        intent,
        confidence,
        customer_message
    )

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "reply": reply,
        "escalate_to_human": escalate,
        "escalation_reason": escalation_reason,
        "evidence": evidence,
    }


def main():
    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("Apple Support prototype")
    print("=" * 70)

    customer_message = input("\nCustomer message: ").strip()

    previous_context = input(
        "Previous conversation context (optional): "
    ).strip()

    if not customer_message:
        print("Please enter a customer message.")
        return

    result = run_agent(
        customer_message,
        previous_context
    )

    print("\n" + "=" * 70)
    print("AGENT DECISION")
    print("=" * 70)

    print(f"\nIntent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nDraft reply:")
    print("-" * 70)
    print(result["reply"])
    print("-" * 70)

    print(
        "\nDecision:",
        "ESCALATE TO HUMAN"
        if result["escalate_to_human"]
        else "AUTO-HANDLE"
    )

    print(f"Reason: {result['escalation_reason']}")

    print("\nHistorical evidence:")
    for i, item in enumerate(result["evidence"], 1):
        print(f"\n[{i}]")
        print(item)


if __name__ == "__main__":
    main()
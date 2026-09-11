import pandas as pd
import os

INPUT_FILE = "golden_set_auto_labeled.csv"
OUTPUT_FILE = "golden_set.csv"

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

df = pd.read_csv(INPUT_FILE)

# Make sure required columns exist
for col in ["intent", "escalate", "label_reason", "review_status"]:
    if col not in df.columns:
        df[col] = ""

# Force text columns to object/string
for col in ["intent", "escalate", "label_reason", "review_status"]:
    df[col] = df[col].astype("string")

# Preserve the first 18 labels that were already reviewed
print("=" * 70)
print("HIVER GOLDEN SET HUMAN REVIEW")
print("=" * 70)

print(f"\nLoaded {len(df)} examples.")
print("\nControls:")
print("  Y = accept proposed label")
print("  N = change label")
print("  S = skip for now")
print("  Q = quit and save")
print()

def show_value(value):
    if pd.isna(value):
        return ""
    return str(value)

for i in range(len(df)):

    # Skip already human-reviewed examples
    if show_value(df.loc[i, "review_status"]).lower() == "reviewed":
        continue

    customer = show_value(df.loc[i, "customer_message"])
    context = show_value(df.loc[i, "previous_context"])
    apple_reply = show_value(df.loc[i, "apple_reply"])

    proposed_intent = show_value(df.loc[i, "auto_intent"])

    if not proposed_intent or proposed_intent not in INTENTS:
        proposed_intent = "other_unclear"

    proposed_escalate = show_value(df.loc[i, "auto_escalate"]).lower()

    print("\n" + "=" * 70)
    print(f"EXAMPLE {i + 1} / {len(df)}")
    print("=" * 70)

    if context:
        print("\nPREVIOUS CONTEXT:")
        print(context)

    print("\nCUSTOMER:")
    print(customer)

    if apple_reply:
        print("\nHISTORICAL APPLE REPLY:")
        print(apple_reply)

    print("\n" + "-" * 70)
    print(f"PROPOSED INTENT: {proposed_intent}")
    print(f"PROPOSED ESCALATION: {proposed_escalate}")
    print("-" * 70)

    action = input("\nAccept? [Y/N/S/Q]: ").strip().lower()

    if action == "q":
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved progress to {OUTPUT_FILE}")
        break

    if action == "s":
        continue

    if action == "y":
        final_intent = proposed_intent
    elif action == "n":

        print("\nChoose the correct intent:")

        for num, intent in enumerate(INTENTS, 1):
            print(f"{num:2}. {intent}")

        while True:
            choice = input("\nIntent number: ").strip()

            try:
                choice_num = int(choice)

                if 1 <= choice_num <= len(INTENTS):
                    final_intent = INTENTS[choice_num - 1]
                    break

            except ValueError:
                pass

            print("Invalid choice. Enter a number from 1 to 12.")

    else:
        print("Please enter Y, N, S, or Q.")
        continue

    # Escalation review
    while True:

        escalation_input = input(
            f"Escalate to human? [Y/N] "
            f"(proposed: {proposed_escalate}): "
        ).strip().lower()

        if escalation_input in ["y", "yes"]:
            final_escalate = "yes"
            break

        if escalation_input in ["n", "no"]:
            final_escalate = "no"
            break

        print("Please enter Y or N.")

    reason = input(
        "Short reason for your final label "
        "(e.g. 'battery drain after update'): "
    ).strip()

    if not reason:
        reason = f"Human-reviewed as {final_intent}"

    df.loc[i, "intent"] = final_intent
    df.loc[i, "escalate"] = final_escalate
    df.loc[i, "label_reason"] = reason
    df.loc[i, "review_status"] = "reviewed"

    # Save after every example
    df.to_csv(OUTPUT_FILE, index=False)

    reviewed = (
        df["review_status"]
        .fillna("")
        .str.lower()
        .eq("reviewed")
        .sum()
    )

    print(f"\n✓ Saved. Human-reviewed: {reviewed}/{len(df)}")

else:
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 70)
    print("REVIEW COMPLETE")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")
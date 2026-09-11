import os
import json
import csv
import time
import re
import urllib.request
import urllib.error

from intent_classifier import predict_intent
from retrieval import retrieve_similar_cases
from generator import generate_reply
from escalation import decide_escalation


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "golden_set.csv"
OUTPUT_FILE = "llm_judge_results.csv"

API_KEY = os.getenv("GEMINI_API_KEY")

# We set this from CMD with:
# set JUDGE_MODEL=gemini-3.6-flash
MODEL = os.getenv(
    "JUDGE_MODEL",
    "gemini-3.6-flash"
)

MAX_RETRIES = 5


# ============================================================
# LLM JUDGE RUBRIC
# ============================================================

SYSTEM_PROMPT = """
You are a strict evaluator for an AI customer-support agent.

The agent handles real Apple customer-support conversations.

The agent receives:
1. A customer message.
2. Optional previous conversation context.
3. A predicted support intent.
4. A generated reply.
5. Historical Apple Support examples retrieved for the same problem.

Your job is to evaluate ONLY the generated reply.

Do not give high scores merely because the reply is polite.

Score each dimension from 1 to 5.

1. relevance
Does the reply address the customer's actual problem?

1 = completely irrelevant
2 = mostly irrelevant
3 = partially addresses the problem
4 = addresses the problem well
5 = directly and precisely addresses the problem

2. groundedness
Is the reply supported by the historical Apple Support examples?

1 = largely invented/unsupported
2 = weakly grounded
3 = partly grounded
4 = mostly grounded
5 = strongly grounded in the evidence

Do not reward invented policies, capabilities, guarantees, or facts.

3. helpfulness
Does the reply provide a useful next step?

1 = not useful
2 = very limited
3 = somewhat useful
4 = useful
5 = highly actionable

4. brand_consistency
Does the reply resemble the tone and resolution style seen in the
historical Apple Support examples?

1 = very inconsistent
2 = somewhat inconsistent
3 = acceptable
4 = consistent
5 = highly consistent

5. safety
Does the reply avoid unsupported promises, risky instructions,
and overconfident claims?

1 = unsafe
2 = significant concern
3 = acceptable with concerns
4 = safe
5 = clearly safe

Also provide:

overall:
An overall reply-quality score from 1 to 5.

unsupported_claims:
true if the reply contains claims not supported by the evidence.
Otherwise false.

major_issue:
The most important weakness, or an empty string if there is no major issue.

reasoning:
A concise explanation of the scores.

Return ONLY valid JSON.
"""


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def extract_json(text):
    """
    Extract JSON even if the model accidentally wraps it
    in markdown code fences.
    """

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:
            return json.loads(
                match.group(0)
            )

        raise


# ============================================================
# GEMINI API
# ============================================================

def call_gemini(
    customer_message,
    context,
    intent,
    reply,
    evidence
):

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    # --------------------------------------------------------
    # Build historical evidence
    # --------------------------------------------------------

    evidence_text = ""

    for i, case in enumerate(
        evidence[:3],
        1
    ):

        if isinstance(case, dict):

            customer = str(
                case.get(
                    "customer_message",
                    ""
                )
            )

            historical_reply = str(
                case.get(
                    "apple_reply",
                    ""
                )
            )

        else:

            customer = ""
            historical_reply = str(case)

        evidence_text += (
            f"\n--- Historical Case {i} ---\n"
            f"Customer: {customer}\n"
            f"Apple Support reply: "
            f"{historical_reply}\n"
        )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
{SYSTEM_PROMPT}

CUSTOMER MESSAGE:
{customer_message}

PREVIOUS CONTEXT:
{context}

PREDICTED INTENT:
{intent}

GENERATED REPLY:
{reply}

HISTORICAL EVIDENCE:
{evidence_text}

Evaluate the generated reply.

Return JSON exactly in this structure:

{{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "brand_consistency": 1,
  "safety": 1,
  "overall": 1,
  "unsupported_claims": false,
  "major_issue": "",
  "reasoning": ""
}}
"""

    # --------------------------------------------------------
    # Gemini endpoint
    # --------------------------------------------------------

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {

        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],

        "generationConfig": {

            "temperature": 0,

            "responseMimeType":
                "application/json"
        }
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(

        url,

        data=data,

        headers={
            "x-goog-api-key":
                API_KEY,

            "Content-Type":
                "application/json"
        },

        method="POST"
    )

    # --------------------------------------------------------
    # Retry temporary errors
    # --------------------------------------------------------

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            with urllib.request.urlopen(
                request,
                timeout=90
            ) as response:

                result = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            text = (
                result[
                    "candidates"
                ][0][
                    "content"
                ][
                    "parts"
                ][0][
                    "text"
                ]
            )

            return extract_json(
                text
            )

        except urllib.error.HTTPError as e:

            body = e.read().decode(
                "utf-8",
                errors="ignore"
            )

            # ------------------------------------------------
            # Temporary errors
            # ------------------------------------------------

            if e.code in (
                429,
                500,
                502,
                503,
                504
            ):

                wait_seconds = (
                    5 * (attempt + 1)
                )

                print(
                    f"  Gemini temporary "
                    f"error {e.code}. "
                    f"Retry "
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES} "
                    f"in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            # ------------------------------------------------
            # Permanent error
            # ------------------------------------------------

            raise RuntimeError(
                f"Gemini API error "
                f"{e.code}: {body}"
            )

        except (
            urllib.error.URLError,
            TimeoutError
        ) as e:

            wait_seconds = (
                5 * (attempt + 1)
            )

            print(
                f"  Network/timeout "
                f"error. Retry "
                f"{attempt + 1}/"
                f"{MAX_RETRIES} "
                f"in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Gemini remained unavailable "
        f"after {MAX_RETRIES} retries."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # API key check
    # --------------------------------------------------------

    if not API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Run:\n"
            "set GEMINI_API_KEY=YOUR_KEY"
        )

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    print(
        "Loading golden set..."
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        f"Examples: {len(rows)}"
    )

    print(
        f"Judge model: {MODEL}"
    )

    print()

    # --------------------------------------------------------
    # Resume existing results
    # --------------------------------------------------------

    existing = {}

    if os.path.exists(
        OUTPUT_FILE
    ):

        print(
            f"Found existing results: "
            f"{OUTPUT_FILE}"
        )

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            old_results = list(
                csv.DictReader(f)
            )

        for result in old_results:

            tweet_id = str(
                result.get(
                    "customer_tweet_id",
                    ""
                )
            )

            if tweet_id:

                existing[
                    tweet_id
                ] = result

        print(
            f"Already completed: "
            f"{len(existing)}"
        )

    results = list(
        existing.values()
    )

    remaining = (
        len(rows)
        -
        len(existing)
    )

    print(
        f"Remaining: {remaining}"
    )

    print()

    # --------------------------------------------------------
    # Process examples
    # --------------------------------------------------------

    for index, row in enumerate(
        rows,
        1
    ):

        tweet_id = str(
            row.get(
                "customer_tweet_id",
                ""
            )
        )

        # ----------------------------------------------------
        # Skip completed examples
        # ----------------------------------------------------

        if tweet_id in existing:

            print(
                f"[{index}/{len(rows)}] "
                f"SKIP - already completed"
            )

            continue

        customer_message = str(
            row.get(
                "customer_message",
                ""
            )
        )

        previous_context = str(
            row.get(
                "previous_context",
                ""
            )
        )

        # ----------------------------------------------------
        # Intent classification
        # ----------------------------------------------------

        intent_result = predict_intent(
            customer_message,
            previous_context
        )

        if isinstance(
            intent_result,
            tuple
        ):

            intent = (
                intent_result[0]
            )

            confidence = safe_float(
                intent_result[1]
            )

        elif isinstance(
            intent_result,
            dict
        ):

            intent = (
                intent_result.get(
                    "intent",
                    "other_unclear"
                )
            )

            confidence = safe_float(
                intent_result.get(
                    "confidence",
                    0
                )
            )

        else:

            intent = str(
                intent_result
            )

            confidence = 0.0

        # ----------------------------------------------------
        # Generate support reply
        # ----------------------------------------------------

        reply_result = generate_reply(
            customer_message,
            previous_context,
            intent=intent,
            top_k=3
        )

        if isinstance(
            reply_result,
            tuple
        ):

            generated_reply = (
                reply_result[0]
            )

            evidence = (
                reply_result[1]
                if len(reply_result) > 1
                else []
            )

        elif isinstance(
            reply_result,
            dict
        ):

            generated_reply = (
                reply_result.get(
                    "reply",
                    ""
                )
            )

            evidence = (
                reply_result.get(
                    "evidence",
                    []
                )
            )

        else:

            generated_reply = str(
                reply_result
            )

            evidence = []

        # ----------------------------------------------------
        # Escalation decision
        # ----------------------------------------------------

        escalation_result = (
            decide_escalation(
                intent,
                confidence,
                customer_message
            )
        )

        if isinstance(
            escalation_result,
            tuple
        ):

            escalate = (
                escalation_result[0]
            )

            escalation_reason = (
                escalation_result[1]
                if len(
                    escalation_result
                ) > 1
                else ""
            )

        else:

            escalate = bool(
                escalation_result
            )

            escalation_reason = ""

        # ----------------------------------------------------
        # Gemini judge
        # ----------------------------------------------------

        try:

            judge = call_gemini(

                customer_message,

                previous_context,

                intent,

                generated_reply,

                evidence
            )

        except Exception as e:

            print()

            print(
                f"ERROR on example "
                f"{index}:"
            )

            print(
                str(e)
            )

            print()

            print(
                "Completed results "
                "have already been "
                "saved."
            )

            print(
                "Run the same command "
                "again to resume."
            )

            return

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        result = {

            "customer_tweet_id":
                tweet_id,

            "customer_message":
                customer_message,

            "intent":
                intent,

            "confidence":
                confidence,

            "generated_reply":
                generated_reply,

            "escalate":
                escalate,

            "escalation_reason":
                escalation_reason,

            "relevance":
                judge.get(
                    "relevance",
                    ""
                ),

            "groundedness":
                judge.get(
                    "groundedness",
                    ""
                ),

            "helpfulness":
                judge.get(
                    "helpfulness",
                    ""
                ),

            "brand_consistency":
                judge.get(
                    "brand_consistency",
                    ""
                ),

            "safety":
                judge.get(
                    "safety",
                    ""
                ),

            "overall":
                judge.get(
                    "overall",
                    ""
                ),

            "unsupported_claims":
                judge.get(
                    "unsupported_claims",
                    ""
                ),

            "major_issue":
                judge.get(
                    "major_issue",
                    ""
                ),

            "reasoning":
                judge.get(
                    "reasoning",
                    ""
                ),

            # Human intent/escalation labels
            # are kept separately from
            # reply-quality judging.
            "human_reviewed":
                row.get(
                    "review_status",
                    ""
                ),

            "human_intent":
                row.get(
                    "intent",
                    ""
                ),

            "human_escalate":
                row.get(
                    "escalate",
                    ""
                )
        }

        results.append(
            result
        )

        existing[
            tweet_id
        ] = result

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        save_results(
            results
        )

        print(
            f"[{index}/{len(rows)}] "
            f"intent={intent} "
            f"overall="
            f"{judge.get('overall', '?')} "
            f"--> SAVED"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "GEMINI LLM JUDGE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    scores = []

    for result in results:

        score = safe_float(
            result.get(
                "overall",
                0
            )
        )

        if score > 0:

            scores.append(
                score
            )

    if scores:

        average = (
            sum(scores)
            /
            len(scores)
        )

        high_quality = (
            sum(
                1
                for score in scores
                if score >= 4
            )
            /
            len(scores)
        )

        print(
            f"Examples judged: "
            f"{len(scores)}"
        )

        print(
            f"Average overall "
            f"reply quality: "
            f"{average:.2f}/5"
        )

        print(
            f"Replies rated >=4: "
            f"{high_quality * 100:.1f}%"
        )

    else:

        print(
            "No valid judge scores "
            "were produced."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
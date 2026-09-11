\# Hiver AI Support Agent — Evaluation Report



\## 1. Problem Framing



This project builds an AI customer-support agent for Apple Support using the Customer Support on Twitter dataset.



The agent has three responsibilities:



1\. Classify an incoming customer message into a small, interpretable set of support intents.

2\. Draft a reply grounded in historically similar Apple Support conversations.

3\. Decide whether the issue can be auto-handled or should be escalated to a human, with a reason.



The goal was not to build a production-grade support system. The goal was to build a small, reproducible system and evaluate where it works and where it fails on noisy real-world support conversations.



\### What I built



\- A 12-class intent taxonomy derived from Apple Support conversations.

\- A TF-IDF + Logistic Regression intent classifier.

\- A retrieval component using historical Apple Support conversations.

\- A retrieval-grounded deterministic reply generator.

\- A rule-based escalation policy.

\- An automated evaluation harness.

\- Majority and keyword baselines.

\- A Gemini-based LLM-as-judge harness for reply quality.

\- A 200-example golden evaluation set.



\### What I deliberately did not build



I did not attempt to:



\- train a large neural classifier from scratch;

\- process the full \~3M-row dataset;

\- build a production deployment;

\- create a fully autonomous customer-support agent;

\- optimize for maximum benchmark accuracy at the expense of interpretability;

\- claim production-level performance from a small development evaluation set.



The system is intended as an evidence-driven prototype.



\---



\## 2. Dataset and Sampling



The source dataset contains real customer-support conversations between customers and brands on Twitter.



I selected `AppleSupport` because it provides a large number of support interactions and contains repeated technical-support patterns.



The working dataset contains 10,000 Apple Support interactions.



The original Apple Support data contains approximately 106,000 usable customer/support interaction pairs in the extracted subset.



For development:



\- 10,000 examples were sampled for the working dataset.

\- 200 examples were reserved for the golden evaluation set.

\- Golden examples were excluded from classifier training.

\- Golden examples were also excluded from the retrieval index to avoid direct evidence leakage.



The remaining 9,800 examples were used for the silver training/retrieval pipeline.



The golden set currently contains:



\- 18 human-reviewed examples.

\- 182 provisional/proposed labels.



Therefore, the headline classification number should be interpreted as an early development result rather than definitive human-ground-truth performance.



\---



\## 3. Intent Taxonomy



The final taxonomy contains 12 intents:



| Intent | Description |

|---|---|

| `software\_update` | Problems/questions related to OS updates |

| `battery\_power` | Battery drain, charging, shutdown |

| `device\_performance` | Freezing, crashing, slowness, instability |

| `connectivity\_network` | Wi-Fi, cellular, Bluetooth, network issues |

| `messaging\_calls` | iMessage, SMS, calls, FaceTime |

| `apps\_media` | Apps, App Store, media, Photos, camera |

| `account\_icloud` | Apple ID, iCloud, account access |

| `hardware\_repair` | Hardware failures, repair, replacement, warranty |

| `payments\_billing` | Charges, subscriptions, refunds, billing |

| `settings\_features` | Settings and individual feature behavior |

| `feedback\_request` | Product feedback and feature requests |

| `other\_unclear` | Ambiguous or insufficient information |



A primary-intent rule was used rather than assigning every keyword to an intent.



For example:



\- "After iOS update my battery drains" → `battery\_power`

\- "After iOS update my phone freezes" → `device\_performance`

\- "Why can't I install the latest iOS?" → `software\_update`



This prevents the word "update" from automatically dominating the classification.



\---



\## 4. System Architecture



The system follows this pipeline:



```text

Customer message

&#x20;      |

&#x20;      v

Previous conversation context

&#x20;      |

&#x20;      v

Intent classifier

(TF-IDF + Logistic Regression)

&#x20;      |

&#x20;      +----------------------+

&#x20;      |                      |

&#x20;      v                      v

Historical retrieval       Escalation policy

&#x20;      |                      |

&#x20;      v                      v

Similar Apple cases       AUTO / HUMAN

&#x20;      |

&#x20;      v

Grounded reply generator

&#x20;      |

&#x20;      v

Final support response


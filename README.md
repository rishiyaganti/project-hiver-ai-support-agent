\# Hiver AI Support Agent — AppleSupport



An AI customer-support prototype built on real-world Twitter customer-support conversations.



The system takes an incoming customer message and:



1\. Classifies the customer's primary support intent.

2\. Retrieves historically similar AppleSupport cases.

3\. Drafts a reply grounded in those historical cases.

4\. Decides whether the request can be auto-handled or should be escalated to a human.



\## Problem



Customer-support conversations are noisy, short, contextual and often incomplete.



The goal is not to build a generic chatbot. The goal is to build a support agent whose decisions and replies are grounded in how the brand historically handled similar customer problems.



\## Dataset



Dataset:



Customer Support on Twitter



Source:

Kaggle — Customer Support on Twitter



The dataset contains approximately 2.8M tweets across multiple brands.



AppleSupport was selected because it has a large number of support interactions and contains recurring technical-support patterns.



The prototype uses a 10,000-example AppleSupport interaction subset.



After reserving 200 evaluation examples, 9,800 examples are used for model training/retrieval.



\## Intent Taxonomy



The system uses 12 intents:



1\. software\_update

2\. battery\_power

3\. device\_performance

4\. connectivity\_network

5\. messaging\_calls

6\. apps\_media

7\. account\_icloud

8\. hardware\_repair

9\. payments\_billing

10\. settings\_features

11\. feedback\_request

12\. other\_unclear



The classifier predicts the customer's primary support problem rather than assigning every possible topic mentioned in the message.



\## Architecture



```text

Customer message

&#x20;      |

&#x20;      v

Previous context + current message

&#x20;      |

&#x20;      v

TF-IDF + Logistic Regression

&#x20;      |

&#x20;      v

Intent + confidence

&#x20;      |

&#x20;      +----------------------+

&#x20;      |                      |

&#x20;      v                      v

Historical retrieval     Escalation policy

&#x20;      |                      |

&#x20;      v                      v

Grounded reply            Human / Auto

&#x20;      |

&#x20;      v

Final support response


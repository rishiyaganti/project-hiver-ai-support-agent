\# Decision Log



\## 1. Selected AppleSupport as the brand



\*\*Decision:\*\* Use AppleSupport rather than another brand.



\*\*Why:\*\* AppleSupport has a large number of customer-support interactions and repeated technical-support patterns. This provides enough examples for intent discovery, retrieval, and evaluation while keeping the scope manageable.



\---



\## 2. Used a 10,000-example working subset



\*\*Decision:\*\* Do not process the full dataset.



\*\*Why:\*\* The assignment explicitly allows/encourages subsampling. A 10,000-example working dataset makes experimentation fast and reproducible while still containing diverse real-world support conversations.



\---



\## 3. Created a 12-intent taxonomy



\*\*Decision:\*\* Use a small, interpretable intent taxonomy rather than dozens of highly specific labels.



\*\*Why:\*\* Support routing needs actionable categories. Too many labels would increase ambiguity and reduce the number of examples available per class.



\---



\## 4. Used primary-intent labeling



\*\*Decision:\*\* Assign the customer's primary support problem rather than every topic mentioned.



\*\*Why:\*\* A message can contain multiple keywords. For example, an issue occurring after an iOS update may actually be a battery or performance problem. The routing decision should reflect the problem the customer needs solved.



\---



\## 5. Included previous conversation context



\*\*Decision:\*\* Combine the current customer message with available previous context.



\*\*Why:\*\* Twitter support conversations are multi-turn. Short follow-up messages can be impossible to classify correctly without the previous conversation.



\---



\## 6. Used TF-IDF + Logistic Regression



\*\*Decision:\*\* Start with a lightweight classical classifier.



\*\*Why:\*\* It is fast, interpretable, reproducible, and establishes a strong baseline before introducing more complex models.



\---



\## 7. Used class balancing



\*\*Decision:\*\* Use `class\_weight="balanced"`.



\*\*Why:\*\* The intent distribution is highly imbalanced. Without class balancing, common intents could dominate the classifier.



\---



\## 8. Created a 200-example golden set



\*\*Decision:\*\* Reserve 200 examples for evaluation.



\*\*Why:\*\* The assignment asks for a 150–250 example golden set. A fixed evaluation set provides a repeatable benchmark.



\---



\## 9. Excluded golden examples from training



\*\*Decision:\*\* Remove golden-set IDs from the training data.



\*\*Why:\*\* Prevents direct leakage between evaluation and training.



\---



\## 10. Excluded golden examples from retrieval



\*\*Decision:\*\* Also remove golden examples from the retrieval index.



\*\*Why:\*\* Otherwise the reply generator could retrieve the exact evaluation conversation, producing artificially strong grounding results.



\---



\## 11. Used a retrieval-grounded reply generator



\*\*Decision:\*\* Ground replies in historical Apple Support responses.



\*\*Why:\*\* The assignment specifically asks the system to draft replies based on how the brand historically resolved similar issues.



\---



\## 12. Chose conservative reply generation



\*\*Decision:\*\* Prefer safe diagnostic questions and DM requests over unsupported claims.



\*\*Why:\*\* A customer-support agent should avoid inventing policies, refunds, guarantees, or capabilities that are not supported by historical evidence.



\---



\## 13. Used a separate escalation layer



\*\*Decision:\*\* Keep escalation logic separate from reply generation.



\*\*Why:\*\* A good reply and a good routing decision are different problems. Separating them makes the system easier to evaluate and modify.



\---



\## 14. Prioritized explicit risk signals for escalation



\*\*Decision:\*\* Escalate security, financial-risk, severe device, data-loss, repeated failure, and explicit human-support requests.



\*\*Why:\*\* In customer support, certain failures are more costly than unnecessary escalation. These signals provide a transparent safety layer.



\---



\## 15. Added an LLM-as-judge evaluation layer



\*\*Decision:\*\* Use Gemini to judge reply quality across relevance, groundedness, helpfulness, brand consistency, and safety.



\*\*Why:\*\* Automated lexical metrics alone cannot reliably determine whether a support reply is genuinely useful and grounded in historical evidence.



The judge is implemented separately from the agent so that the evaluator does not affect the agent's behavior.


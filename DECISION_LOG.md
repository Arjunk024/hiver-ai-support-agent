# Hiver AI Support Agent — Decision Log

## 1. Selected SpotifyCares

**Decision:** Build the prototype around SpotifyCares.

**Why:** SpotifyCares provides enough recurring customer/support interactions for intent discovery, retrieval and evaluation.

**Trade-off:** The prototype is specialised to one brand rather than immediately generalising across all brands.

---

## 2. Pair inbound customer tweets with direct support responses

**Decision:** Use customer tweets paired with corresponding SpotifyCares support replies.

**Why:** This directly captures how the brand historically handled customer issues.

**Trade-off:** Some longer multi-turn thread context is lost.

---

## 3. Use a compact six-intent taxonomy

**Decision:** Use `account_login`, `playback`, `playlist`, `subscription`, `security`, and `general`.

**Why:** These categories capture recurring support patterns while remaining small enough to evaluate consistently.

**Trade-off:** Some real customer messages naturally belong to multiple categories.

---

## 4. Keep a held-out test split

**Decision:** Evaluate on examples separated from the training data.

**Why:** Prevents the headline result from simply measuring memorisation.

**Trade-off:** Less data is available to the retrieval corpus.

---

## 5. Remove golden examples from retrieval

**Decision:** Exclude the 200 golden evaluation cases from the final retrieval index.

**Why:** An overlap check found exact-text leakage that could inflate the evaluation result.

**Trade-off:** The retrieval index becomes slightly smaller, but the final evaluation is substantially more credible.

---

## 6. Use MiniLM embeddings

**Decision:** Use `all-MiniLM-L6-v2`.

**Why:** It provides useful semantic retrieval while remaining lightweight and CPU-friendly.

**Trade-off:** Larger embedding models or dedicated rerankers could provide better retrieval quality.

---

## 7. Normalize embeddings

**Decision:** Normalize vectors before similarity search.

**Why:** This makes the retrieval score comparable to cosine similarity and provides an interpretable confidence signal.

**Trade-off:** Similarity alone is not sufficient to distinguish every closely related support intent.

---

## 8. Ground responses in historical support examples

**Decision:** Retrieve similar historical SpotifyCares interactions before drafting a response.

**Why:** The task requires learning from how the brand historically handled support issues rather than generating generic customer-service language.

**Trade-off:** Historical support behaviour can be incomplete or outdated.

---

## 9. Use a rule-based intent layer initially

**Decision:** Implement the initial intent classifier using interpretable rules and phrase patterns.

**Why:** It is fast, transparent, deterministic and easy to debug within the assignment timeframe.

**Trade-off:** It is vulnerable to wording variations and does not demonstrate learned-model generalisation.

---

## 10. Prioritise security

**Decision:** Security/fraud/payment-compromise signals receive the highest intent priority.

**Why:** These cases carry greater risk than routine support issues.

**Trade-off:** Broad security rules can create false positives.

---

## 11. Never request passwords

**Decision:** The response layer must not ask customers for passwords.

**Why:** Passwords are sensitive credentials and should not be requested by a support agent.

**Trade-off:** Account workflows requiring authenticated verification are outside this prototype.

---

## 12. Escalate high-risk cases

**Decision:** Security, fraud and payment-compromise cases are routed to human review.

**Why:** These cases may require authenticated investigation or operational actions unavailable to the prototype.

**Trade-off:** Some cases that could eventually be automated will still escalate.

---

## 13. Use retrieval confidence for escalation

**Decision:** Low-confidence retrieval can trigger human escalation.

**Why:** It is safer to escalate than generate an unsupported answer.

**Trade-off:** Conservative thresholds can increase escalation volume.

---

## 14. Use the human-labelled golden set as the primary evaluation

**Decision:** Use the 200 manually labelled examples as the main evaluation.

**Why:** Human labels provide a more defensible measure than treating another model as ground truth.

**Trade-off:** Manual annotation is slower and the set is still relatively small.

---

## 15. Use independent simple baselines

**Decision:** Compare against a majority-class baseline and TF-IDF + Logistic Regression.

**Why:** They provide clear trivial and simple machine-learning reference points.

**Trade-off:** They do not represent stronger modern classifiers.

---

## 16. Treat the local LLM judge as experimental

**Decision:** Do not use the local FLAN-T5 judge as the primary response-quality metric.

**Why:** Agreement with human labels was weak: 24.14% exact agreement and quadratic weighted kappa of 0.169.

**Trade-off:** Automated response-quality evaluation remains less mature, but the submission avoids treating an unreliable judge as ground truth.

---

## 17. Report limitations instead of inflating the headline result

**Decision:** Explicitly document retrieval leakage correction, evaluation scope and LLM-judge limitations.

**Why:** A credible engineering evaluation should explain what its metrics do and do not prove.

**Trade-off:** The submission may appear less polished than one reporting only its strongest number, but the evidence is more defensible.
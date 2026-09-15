# Hiver AI Support Agent — Final Report

## 1. Problem Framing

### Objective

The goal of this project is to build an AI support agent that learns from real customer-support conversations and can:

1. classify an incoming customer message into a small, data-derived set of support intents,
2. draft a response grounded in how the brand historically handled similar issues, and
3. decide whether to auto-handle or escalate to a human, with an explicit reason.

**Target brand:** SpotifyCares.

### What does "good" mean?

A good support agent should:

- identify the customer's intent correctly,
- use historical support behaviour as evidence rather than inventing unsupported solutions,
- provide a concise and actionable response,
- avoid requesting sensitive credentials such as passwords,
- escalate security, fraud, and payment-compromise cases appropriately,
- avoid leaking private identifiers or historical tweet metadata.

### What is not built?

This is a prototype rather than a production support platform. It does not:

- authenticate or access a customer's Spotify account,
- perform live refunds, subscription changes, or account actions,
- integrate with production CRM/helpdesk systems,
- maintain full multi-turn conversational state,
- provide a live knowledge base,
- provide production-grade multilingual support.

---

## 2. Data and System

### Dataset

The project uses the **Customer Support on Twitter** dataset and focuses on SpotifyCares interactions.

Final extracted support pairs:

- **27,135** customer/support pairs
- **21,707** training examples
- **2,714** held-out test examples

The golden evaluation set contains **200 manually labelled examples** sampled from the held-out test split using `random_state=42`.

### Intent Taxonomy

Six intents were defined from recurring SpotifyCares support patterns:

| Intent | Description |
|---|---|
| `account_login` | Login, sign-in, password and access problems |
| `playback` | Songs not playing, buffering, pausing, skipping and playback errors |
| `playlist` | Playlist/song management and playlist-related problems |
| `subscription` | Premium, family, student and subscription-related issues |
| `security` | Hacked accounts, fraud, unauthorized payments and compromised payment information |
| `general` | Other support questions, feedback and issues not fitting the above |

### Architecture

The pipeline is:

**Incoming message → intent detection → historical retrieval → grounded response → escalation decision**

Retrieval uses `all-MiniLM-L6-v2` sentence embeddings with normalized vectors and similarity search.

Similar historical SpotifyCares interactions provide evidence for the drafted response.

The intent layer is deliberately rule-based and interpretable. Security has the highest priority so that a high-risk signal is not overridden by a routine category.

The escalation layer routes security/fraud/payment-compromise cases to human review and can also escalate when retrieval confidence is insufficient.

A key safety rule is that the agent never asks customers for their password.

### Leakage Prevention

An earlier evaluation found substantial exact-text overlap between golden examples and the retrieval corpus.

The final evaluation therefore removes golden examples from the retrieval index.

**Final leakage-free retrieval index: 21,465 examples.**

---

## 3. Evaluation

### Golden Evaluation Set

The final evaluation contains **200 manually labelled messages** from the held-out SpotifyCares test split.

| Intent | Cases |
|---|---:|
| account_login | 29 |
| playback | 30 |
| playlist | 32 |
| subscription | 35 |
| security | 25 |
| general | 49 |
| **Total** | **200** |

### Main Results

| Metric | AI Support Agent |
|---|---:|
| Intent accuracy | **97.50%** |
| Macro precision | **97.95%** |
| Macro recall | **97.29%** |
| Macro F1 | **97.55%** |
| Correct predictions | **195 / 200** |
| Average retrieval similarity | **0.8651** |

Per-intent F1:

| Intent | F1 |
|---|---:|
| account_login | 98.31% |
| playback | 94.74% |
| playlist | 95.24% |
| subscription | 100.00% |
| security | 100.00% |
| general | 97.03% |

### Baselines

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 24.50% | 6.56% |
| TF-IDF + Logistic Regression | 41.00% | 32.10% |
| **AI Support Agent** | **97.50%** | **97.55%** |

The agent substantially outperforms both simple baselines on the manually labelled evaluation set.

---

## 4. Response Quality and Safety

A secondary automated response-quality evaluation was run on the 200-case evaluation set.

It reported approximately:

- **99% response-quality pass rate**
- **100% safety/no-leakage pass rate**
- approximately **97.5% overall pass rate**

These are secondary metrics. The primary headline result remains intent classification performance on the manually labelled golden set.

The response layer is deliberately conservative: it uses retrieved historical support behaviour, avoids requesting passwords, and gives security/payment-compromise issues a human escalation path.

---

## 5. LLM-as-Judge Validation

A local **FLAN-T5-base** experiment was used to test whether an LLM could reliably judge response quality without requiring an external API.

The rubric considered:

- correctness,
- helpfulness,
- actionability,
- grounding,
- safety,
- tone.

On a 29-case comparison against human labels:

- Exact agreement: **24.14%**
- Within one point: **62.07%**
- Quadratic weighted Cohen's kappa: **0.169**
- Mean human score: **3.07**
- Mean LLM score: **3.59**

### Interpretation

The local judge was not reliable enough to serve as the primary quality metric.

This result is therefore reported as a limitation rather than presented as a success metric. Human-labelled evaluation and deterministic safety checks are given greater weight.

---

## 6. Top Five Failure Modes

### 1. Taxonomy Boundary / Noisy Labels

**Example:**

> The site was not letting my reset my passwordbut I finally got it!

Expected: `playlist`  
Predicted: `account_login`  
Similarity: **0.8914**

The password-reset signal strongly supports the model's login prediction, suggesting a possible taxonomy or annotation-boundary issue.

**Hypothesis:** Define intent boundaries more explicitly and introduce treatment for ambiguous or multi-issue messages.

---

### 2. Long Emotional Messages Containing Multiple Issues

**Example:**

> after your piss poor performance... I've paid 3 months with no service... want my money back now.

Expected: `playlist`  
Predicted: `general`  
Similarity: **0.8140**

The message mixes service failure, dissatisfaction and refund language.

**Hypothesis:** Extract the actionable support issue before classification and reduce the influence of emotional wording.

---

### 3. Feature Requests Expressed as General Feedback

**Example:**

> any plans to use the playback interface from the Time Capsule mode for normal playback?

Expected: `playback`  
Predicted: `general`  
Similarity: **0.6953**

The customer is discussing playback functionality but phrases the request as product feedback.

**Hypothesis:** Add feature-request patterns and distinguish product feedback from unrelated/general support.

---

### 4. Cross-Intent Mentions

**Example:**

> this specific playlist doesn't show the playback bar

Expected: `playback`  
Predicted: `playlist`  
Similarity: **0.8254**

The message mentions a playlist, but the actual failure concerns playback UI.

**Hypothesis:** Classify the customer's actual failure/action rather than entity keywords. A learned classifier or reranker should improve this boundary.

---

### 5. Missing Phrase Coverage for Playback Errors

**Example:**

> It sometimes appears a playback failed message, but not all the time

Expected: `playback`  
Predicted: `general`  
Similarity: **0.8395**

The phrase "playback failed" is a strong signal, but the current rule set did not cover this wording.

**Hypothesis:** Expand phrase coverage and eventually replace hand-maintained rules with a trained classifier.

---

## 7. What Is Misleading About My Headline Number?

The **97.5% intent accuracy** should not be interpreted as:

> "The AI support agent is 97.5% ready for production."

There are several reasons:

1. The evaluation contains only **200 examples**.
2. The intent classifier is **rule-based**, so the result does not establish learned-model generalisation.
3. The system is specialised to **SpotifyCares**.
4. The six-intent taxonomy may not cover production support complexity.
5. Some customer messages are inherently ambiguous or multi-issue.
6. Intent accuracy does not measure operational correctness, customer satisfaction, or long-term response quality.
7. Production traffic would introduce distribution shift, new issues, multilingual traffic and richer multi-turn context.

The defensible claim is:

> **On a 200-example manually labelled SpotifyCares evaluation set, this implementation correctly classified 195 messages and substantially outperformed the selected simple baselines.**

---

## 8. What I Would Do With One More Week

### 1. Replace rules with a learned intent classifier

Train a lightweight classifier on the labelled training data and focus hard negatives on playback vs playlist and login vs general.

### 2. Add thread-aware context

Include previous customer/support turns for multi-turn cases instead of treating every message independently.

### 3. Improve retrieval ranking

Add a cross-encoder or lightweight reranker after MiniLM retrieval.

### 4. Expand the golden set

Increase the evaluation set and deliberately include difficult boundary cases, multi-issue messages, security cases and noisy customer messages.

### 5. Improve response-quality evaluation

Use multiple human-labelled dimensions and validate any LLM judge against those labels before using it at scale.

### 6. Add production monitoring

Track confidence, escalation rate, retrieval similarity, safety violations and disagreement cases over time.

---

## 9. Conclusion

The project demonstrates a practical support-agent workflow grounded in real customer-support history.

The strongest evidence is the manually evaluated result:

**97.50% intent accuracy and 97.55% macro F1 on 200 held-out, manually labelled SpotifyCares messages**, compared with **24.50%** for the majority baseline and **41.00%** for TF-IDF + Logistic Regression.

The system also addresses an important support-agent decision beyond classification: **when not to automate**.

Security, fraud and payment-compromise cases are routed toward human review, while the response layer avoids requesting passwords or exposing historical identifiers.

The main limitations are explicit: the headline evaluation is narrow, the classifier is rule-based, and response-quality judging needs stronger human validation.

Those limitations define the next engineering steps rather than being treated as solved problems.
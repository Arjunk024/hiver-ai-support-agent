# Hiver AI Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment.

The system uses historical SpotifyCares customer-support conversations to:

1. Classify an incoming customer message into a support intent.
2. Retrieve historically similar support interactions.
3. Draft a response grounded in how SpotifyCares historically handled similar issues.
4. Decide whether the request can be auto-handled or should be escalated to a human.

---

## 1. Problem

Customer-support agents repeatedly handle similar issues such as login problems, playback failures, playlists, subscriptions, and account/security concerns.

The goal of this project is to build a support agent that learns from historical customer-support conversations rather than generating generic answers.

For this implementation, **SpotifyCares** was selected from the Customer Support on Twitter dataset because it contains a large number of real customer-support interactions.

---

## 2. What the System Does

Given a customer message:

```text
Someone used my card through my Spotify account
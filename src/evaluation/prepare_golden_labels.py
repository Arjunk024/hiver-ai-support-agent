import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/evaluation/golden_test_cases_200.csv")
OUTPUT = Path("data/processed/evaluation/golden_test_cases_200.csv")


def suggest_intent(text):
    text = str(text).lower()

    security_keywords = [
        "hacked", "hack", "stolen", "compromised",
        "unauthorized", "fraud", "payment information",
        "payment details", "card information", "card details",
        "debit card", "credit card", "used my card",
        "card was used", "someone used my card",
        "unknown payment", "unauthorized payment",
        "unauthorized charge", "don't recognize this charge",
        "charged without permission"
    ]

    login_keywords = [
        "can't login", "cannot login", "can't log in",
        "cannot log in", "login", "log in", "sign in",
        "signin", "password", "forgot my password"
    ]

    playback_keywords = [
        "buffer", "buffering", "won't play",
        "not playing", "stopping", "stops playing",
        "stopping after", "pausing", "pause",
        "skipping", "skip songs", "playback"
    ]

    playlist_keywords = [
        "playlist", "song missing", "songs missing",
        "songs disappeared", "playlist disappeared",
        "add song", "remove song"
    ]

    subscription_keywords = [
        "premium", "subscription", "family plan",
        "student plan", "student discount", "premium account"
    ]

    if any(k in text for k in security_keywords):
        return "security"

    if any(k in text for k in login_keywords):
        return "account_login"

    if any(k in text for k in playback_keywords):
        return "playback"

    if any(k in text for k in playlist_keywords):
        return "playlist"

    if any(k in text for k in subscription_keywords):
        return "subscription"

    return "general"


df = pd.read_csv(INPUT)

df["suggested_intent"] = df["customer_text"].apply(suggest_intent)

# Keep gold_intent blank.
# This is important: suggested_intent is NOT the final human label.
df["gold_intent"] = ""

# Put suggested label next to the customer message
columns = [
    "case_id",
    "customer_text",
    "suggested_intent",
    "gold_intent",
    "support_text"
]

df = df[columns]

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("SUGGESTED LABELS CREATED")
print("=" * 60)
print("Rows:", len(df))
print()
print(df["suggested_intent"].value_counts())
print()
print("IMPORTANT:")
print("suggested_intent = AI/rule suggestion")
print("gold_intent      = HUMAN-CONFIRMED FINAL LABEL")
print()
print("Open the CSV and review every row.")
print("=" * 60)
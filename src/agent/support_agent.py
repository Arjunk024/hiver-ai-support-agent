import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.retrival.search import search


# ---------------------------------------------------------
# INTENT DETECTION
# ---------------------------------------------------------

def detect_intent(query):
    """
    Lightweight rule-based intent detection.
    This keeps the prototype local, explainable, and deterministic.
    """

    text = query.lower()

    security_keywords = [
       "hacked",
    "hack",
    "stolen",
    "compromised",
    "someone changed",
    "unauthorized",
    "fraud",
    "payment information",
    "payment details",
    "card information",
    "card details",
    "debit card",
    "credit card",
    "used my card",
    "card was used",
    "someone used my card",
    "unknown payment",
    "unauthorized payment",
    "unauthorized charge",
    "charge i don't recognize",
    "don't recognize this charge",
    "someone charged",
    "charged without permission"
    ]

    login_keywords = [
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "login",
        "log in",
        "sign in",
        "password"
    ]

    playback_keywords = [
        "buffer",
        "buffering",
        "won't play",
        "not playing",
        "stopping",
        "stops playing",
        "pausing",
        "pause",
        "skipping",
        "skip songs"
    ]

    playlist_keywords = [
        "playlist",
        "song missing",
        "songs missing",
        "add song",
        "remove song"
    ]

    subscription_keywords = [
        "premium",
        "subscription",
        "family plan",
        "student plan",
        "student discount",
        "premium account"
    ]

    if any(keyword in text for keyword in security_keywords):
        return "security"

    if any(keyword in text for keyword in login_keywords):
        return "account_login"

    if any(keyword in text for keyword in playback_keywords):
        return "playback"

    if any(keyword in text for keyword in playlist_keywords):
        return "playlist"

    if any(keyword in text for keyword in subscription_keywords):
        return "subscription"

    return "general"


# ---------------------------------------------------------
# RESPONSE SANITIZATION
# ---------------------------------------------------------

def sanitize_response(response):
    """Clean historical support text before showing it to the user."""

    response = re.sub(r"https?://\S+", "", response)
    response = re.sub(r"@\d+", "", response)
    response = re.sub(r"\s/[A-Z]{2}\b", "", response)

    response = response.replace("&gt;", ">")
    response = response.replace("&lt;", "<")
    response = response.replace("&amp;", "&")

    # Fix accidental word-joining in generated/displayed text
    replacements = {
        "security guidance": "security guidance",
        "Family plan": "Family plan",
        "Premiumaccount": "Premium account",
        "Spotifyversion": "Spotify version",
        "OfflineMode": "Offline Mode",
    }

    for old, new in replacements.items():
        response = response.replace(old, new)

    response = re.sub(r"\s+", " ", response).strip()

    return response

# ---------------------------------------------------------
# INTENT-SPECIFIC RESPONSES
# ---------------------------------------------------------

def generate_intent_response(query, intent, results, similarity):

    if intent == "security":

        return (
            "I'm sorry you're dealing with this. Since your Spotify "
            "account or payment information may have been compromised, "
            "please secure your account immediately and follow Spotify's "
            "official account-recovery and security guidance. "
            "If you see an unauthorized payment or believe your card "
            "details were exposed, contact your payment provider as well. "
            "If you still have access to the account, let us know and "
            "we can help with the next steps."
        )

    if intent == "account_login":

        if similarity >= 0.85:
            historical = sanitize_response(
                str(results.iloc[0]["support_text"])
            )

            return (
                "Sorry you're having trouble signing in. "
                "Based on similar support cases, we'd recommend "
                "providing your account username or email so the "
                "account can be checked securely. "
                "Please don't share your password here."
            )

        return (
            "Sorry you're having trouble signing in. "
            "Could you confirm whether you're receiving an error "
            "message or being returned to the login screen? "
            "Please don't share your password."
        )

    if intent == "playback":

        return (
            "Thanks for reaching out! Let's troubleshoot the playback "
            "issue. Could you tell us what device you're using and "
            "which version of Spotify is installed? Also check whether "
            "Spotify is in Offline Mode and try restarting the app or "
            "your device. If the issue continues, let us know what "
            "happens when playback stops."
        )

    if intent == "playlist":

        return (
            "Sorry you're having trouble with your playlist or songs. "
            "Could you tell us what device you're using and exactly "
            "what happens when you try to view, add, or play the songs?"
        )

    if intent == "subscription":

        return (
            "We can help with your Spotify subscription. "
            "Could you tell us whether the issue is with Premium, "
            "a Family plan, or a Student plan, and what happens when "
            "you try to use or manage it?"
        )

    return (
        "Thanks for reaching out! Could you provide a few more details "
        "about the issue, including the device you're using and what "
        "happens when you try?"
    )


# ---------------------------------------------------------
# MAIN RESPONSE GENERATOR
# ---------------------------------------------------------
def decide_handling(intent, similarity):
    """
    Decide whether the AI should auto-handle the request
    or escalate it to a human agent.
    """

    # Security/account-compromise issues should always reach a human.
    if intent == "security":
        return {
            "decision": "ESCALATE",
            "reason": (
                "Security, fraud, or payment-compromise issues "
                "require human review."
            ),
        }

    # Low-confidence retrieval should not be auto-handled.
    if similarity < 0.75:
        return {
            "decision": "ESCALATE",
            "reason": (
                "Low retrieval confidence; a human should review "
                "the customer's issue."
            ),
        }

    # Routine, high-confidence support requests can be auto-handled.
    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            "Routine support intent with sufficiently strong "
            "historical evidence."
        ),
    }
def generate_response(query, top_k=5):
    
    results = search(query, top_k=top_k)

    if results.empty:
        handling = decide_handling(intent, similarity)

        return {
       "response": response,
        "similarity": similarity,
          "intent": intent,
         "decision": handling["decision"],
        "decision_reason": handling["reason"],
        "sources": results
        
            }
    
    best_match = results.iloc[0]
    similarity = float(best_match["similarity"])

    intent = detect_intent(query)

    response = generate_intent_response(
        query,
        intent,
        results,
        similarity
    )
    response = sanitize_response(response)

    handling = decide_handling(intent, similarity)

    return {
        "response": response,
        "similarity": similarity,
        "intent": intent,
        "decision": handling["decision"],
        "decision_reason": handling["reason"],
        "sources": results
    }

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

def display_result(query, result):

    print("\n")
    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)

    print("\nCUSTOMER QUERY")
    print("-" * 70)
    print(query)

    print("\nDETECTED INTENT")
    print("-" * 70)
    print(result["intent"])

    print("\nGENERATED SUPPORT RESPONSE")
    print("-" * 70)
    print(result["response"])
    print("\nRETRIEVAL CONFIDENCE")
    print("-" * 70)
    print(f"{result['similarity']:.4f}")

    print("\nHANDLING DECISION")
    print("-" * 70)
    print(result["decision"])

    print("\nDECISION REASON")
    print("-" * 70)
    print(result["decision_reason"])

    sources = result["sources"]

    for i, (_, row) in enumerate(
        sources.head(3).iterrows(),
        start=1
    ):

        print(f"\n[{i}] Similarity: {row['similarity']:.4f}")

        print("Historical customer:")
        print(row["customer_text"])

        print("\nHistorical support:")
        print(row["support_text"])

        print("-" * 70)


# ---------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("Retrieval-Grounded Support Assistant V3")
    print("=" * 70)

    print("\nType a customer problem.")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nCustomer: ").strip()

        if query.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not query:
            continue

        result = generate_response(
            query,
            top_k=5
        )

        display_result(
            query,
            result
        )


if __name__ == "__main__":
    main()
import sys
import html
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.support_agent import generate_response


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# MARKDOWN HELPER
# =========================================================
#
# IMPORTANT: st.markdown() treats any line that starts with 4+ spaces
# of leading whitespace as a preformatted code block. Since our HTML
# strings below are indented to match the surrounding Python code,
# passing them straight to st.markdown() causes Streamlit to render
# the raw HTML/CSS as literal text instead of parsing it.
#
# textwrap.dedent() strips the common leading whitespace from every
# line in the string first, so the HTML is no longer "indented" from
# Markdown's point of view and renders normally.
#
def md(text: str):
    """st.markdown wrapper that strips ALL leading whitespace from
    every line before rendering.

    textwrap.dedent() alone is not enough here: it only removes the
    *common minimum* indentation across the whole block, so nested
    tags (e.g. a <div> inside another <div>) can still end up with a
    few leftover spaces relative to their parent. Markdown treats any
    line indented 4+ spaces after a blank line as an indented code
    block, so those nested lines were still being rendered as literal
    text. Stripping every line individually removes indentation
    entirely, so nothing is left for Markdown to misinterpret.
    """
    stripped = "\n".join(line.strip() for line in text.split("\n"))
    st.markdown(stripped, unsafe_allow_html=True)


# =========================================================
# CUSTOM STYLING
# =========================================================

md(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f7f8fa;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1, h2, h3 {
        color: #17191c;
        letter-spacing: -0.02em;
    }

    p {
        color: #59636e;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e7e9ec;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .brand {
        font-size: 30px;
        font-weight: 800;
        color: #151719;
        letter-spacing: -1.5px;
        margin-bottom: 2px;
    }

    .brand-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .sidebar-footer {
        margin-top: 80px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
        color: #718096;
        font-size: 13px;
        line-height: 1.8;
    }

    /* ---------- HEADER ---------- */

    .page-kicker {
        color: #e5484d;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .page-title {
        font-size: 34px;
        font-weight: 800;
        color: #17191c;
        margin-bottom: 4px;
    }

    .page-description {
        font-size: 15px;
        color: #66717d;
        margin-bottom: 28px;
    }

    /* ---------- HERO ---------- */

    .hero-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 28px 30px;
        margin-bottom: 26px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .hero-title {
        font-size: 20px;
        font-weight: 750;
        color: #17191c;
        margin-bottom: 7px;
    }

    .hero-text {
        font-size: 14px;
        line-height: 1.7;
        color: #68737f;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #17191c;
        margin-top: 30px;
        margin-bottom: 14px;
    }

    .section-description {
        color: #707985;
        font-size: 14px;
        margin-bottom: 16px;
    }

    /* ---------- RESPONSE CARD ---------- */

    .response-card {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 18px;
        padding: 28px 30px;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.05);
        margin-top: 10px;
        margin-bottom: 22px;
    }

    .response-label {
        font-size: 12px;
        font-weight: 750;
        color: #e5484d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 15px;
    }

    .response-text {
        font-size: 16px;
        line-height: 1.75;
        color: #252a30;
    }

    /* ---------- INTENT BADGE ---------- */

    .intent-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #fff1f2;
        border: 1px solid #ffd5d8;
        color: #c9373d;
        font-size: 13px;
        font-weight: 700;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e4e7eb;
        border-radius: 16px;
        padding: 22px;
        min-height: 145px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.035);
    }

    .metric-label {
        color: #727b86;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #17191c;
    }

    .metric-description {
        margin-top: 7px;
        color: #7b8490;
        font-size: 12px;
        line-height: 1.5;
    }

    /* ---------- CONFIDENCE ---------- */

    .confidence-good {
        color: #16855b;
        font-size: 28px;
        font-weight: 800;
    }

    .confidence-medium {
        color: #a66b00;
        font-size: 28px;
        font-weight: 800;
    }

    .confidence-low {
        color: #c9373d;
        font-size: 28px;
        font-weight: 800;
    }

    /* ---------- WHY RESPONSE ---------- */

    .why-card {
        background: #ffffff;
        border: 1px solid #e4e7eb;
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
    }

    .why-title {
        font-size: 15px;
        font-weight: 750;
        color: #252a30;
        margin-bottom: 10px;
    }

    .why-text {
        font-size: 14px;
        color: #69737e;
        line-height: 1.7;
    }

    /* ---------- EVIDENCE ---------- */

    .evidence-note {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 12px 15px;
        color: #68737f;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ---------- INPUT ---------- */

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid #dfe3e8;
        background: #ffffff;
        font-size: 15px;
        padding: 15px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #e5484d;
        box-shadow: 0 0 0 1px #e5484d;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    /* ---------- STATUS ---------- */

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 11px;
        border-radius: 999px;
        background: #eefbf5;
        color: #16855b;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #d2f1e1;
    }

    /* ---------- EVALUATION ---------- */

    .eval-success {
        background: #eefbf5;
        border: 1px solid #d5f0e2;
        color: #176b4b;
        padding: 14px 16px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .architecture-step {
        background: #ffffff;
        border: 1px solid #e4e7eb;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        color: #343a40;
        font-size: 14px;
        font-weight: 600;
    }

    /* ---------- REMOVE STREAMLIT CHROME ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_text(value):
    """Safely escape text before placing it inside HTML."""
    if value is None:
        return ""
    return html.escape(str(value))


def intent_label(intent):
    labels = {
        "account_login": "Account Login",
        "playback": "Playback",
        "playlist": "Playlist",
        "subscription": "Subscription",
        "security": "Security",
        "general": "General Support",
    }
    return labels.get(intent, intent.replace("_", " ").title())


def confidence_class(score):
    if score >= 0.85:
        return "confidence-good"
    if score >= 0.70:
        return "confidence-medium"
    return "confidence-low"


def confidence_text(score):
    if score >= 0.85:
        return "Strong historical match"
    if score >= 0.70:
        return "Moderate historical match"
    return "Lower historical match"


def load_evaluation_results():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "evaluation"
        / "response_quality_results.csv"
    )

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None

def render_response(result):
    response = str(result["response"])
    intent = intent_label(result["intent"])

    st.markdown("### Recommended customer response")

    st.success(f"Detected intent: {intent}")

    st.write(response)

    md(
        f"""
        <div class="response-card">

            <div class="response-label">
                Recommended customer response
            </div>

            <div style="margin-bottom:18px;">
                <span class="intent-badge">
                    {safe_text(intent)}
                </span>
            </div>

            <div class="response-text">
                {response}
            </div>

        </div>
        """
    )


def render_analysis(result):
    score = float(result["similarity"])
    intent = intent_label(result["intent"])
    cclass = confidence_class(score)

    col1, col2 = st.columns(2)

    with col1:
        md(
            f"""
            <div class="metric-card">
                <div class="metric-label">Detected Intent</div>
                <div style="margin-top:12px;">
                    <span class="intent-badge">
                        {safe_text(intent)}
                    </span>
                </div>
                <div class="metric-description">
                    Classified from the customer's message using
                    an explainable intent layer.
                </div>
            </div>
            """
        )

    with col2:
        md(
            f"""
            <div class="metric-card">
                <div class="metric-label">Retrieval Confidence</div>

                <div class="{cclass}">
                    {score:.4f}
                </div>

                <div class="metric-description">
                    {confidence_text(score)} based on historical
                    SpotifyCares support conversations.
                </div>
            </div>
            """
        )


def render_why_response():
    md(
        """
        <div class="why-card">

            <div class="why-title">
                Retrieval-grounded reasoning
            </div>

            <div class="why-text">
                The agent first identifies the customer's support intent,
                then retrieves semantically similar historical SpotifyCares
                conversations. The final response uses the detected intent
                and retrieved support evidence while avoiding exposure of
                raw historical identifiers.
            </div>

        </div>
        """
    )


def render_evidence(results):
    if results is None or results.empty:
        st.info("No historical support evidence was retrieved.")
        return

    md(
        """
        <div class="evidence-note">
            Historical conversations used as retrieval evidence.
            Raw identifiers are intentionally not shown in the response.
        </div>
        """
    )

    for i, (_, row) in enumerate(results.head(3).iterrows(), start=1):

        similarity = float(row["similarity"])

        customer = safe_text(row.get("customer_text", ""))
        support = safe_text(row.get("support_text", ""))

        with st.expander(
            f"Evidence {i}  •  Similarity {similarity:.4f}"
        ):

            md(
                f"""
                <div style="
                    color:#68737f;
                    font-size:12px;
                    font-weight:700;
                    text-transform:uppercase;
                    letter-spacing:.06em;
                    margin-bottom:7px;
                ">
                    Historical customer message
                </div>

                <div style="
                    background:#f8fafc;
                    border-radius:10px;
                    padding:14px;
                    color:#343a40;
                    font-size:14px;
                    line-height:1.6;
                    margin-bottom:18px;
                ">
                    {customer}
                </div>

                <div style="
                    color:#68737f;
                    font-size:12px;
                    font-weight:700;
                    text-transform:uppercase;
                    letter-spacing:.06em;
                    margin-bottom:7px;
                ">
                    Historical support response
                </div>

                <div style="
                    background:#f8fafc;
                    border-radius:10px;
                    padding:14px;
                    color:#343a40;
                    font-size:14px;
                    line-height:1.6;
                ">
                    {support}
                </div>
                """
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    md(
        """
        <div class="brand">Hiver</div>
        <div class="brand-subtitle">AI Support Workspace</div>
        """
    )

    page = st.radio(
        "Workspace",
        [
            "Support Agent",
            "Evaluation",
            "About",
        ],
        label_visibility="collapsed",
    )

    md(
        """
        <div class="sidebar-footer">
            AI-powered customer support<br>
            Built for <b>SpotifyCares</b><br><br>
            Retrieval-grounded • Explainable • Safe
        </div>
        """
    )


# =========================================================
# SUPPORT AGENT PAGE
# =========================================================

if page == "Support Agent":

    md(
        """
        <div class="page-kicker">Hiver AI Support</div>
        <div class="page-title">Support Agent</div>
        <div class="page-description">
            Generate grounded customer-support responses using
            historical SpotifyCares conversations.
        </div>
        """
    )

    md(
        """
        <div class="hero-card">

            <div class="hero-title">
                AI-assisted support workspace
            </div>

            <div class="hero-text">
                Enter a customer's issue below. The agent retrieves
                semantically similar historical support cases,
                identifies the support intent, applies a safety layer,
                and produces a concise response for the support team.
            </div>

        </div>
        """
    )

    # -----------------------------------------------------
    # CUSTOMER INPUT
    # -----------------------------------------------------

    md(
        """
        <div class="section-title">
            Customer message
        </div>

        <div class="section-description">
            Describe the customer's issue in natural language.
        </div>
        """
    )

    query = st.text_area(
        "Customer message",
        placeholder=(
            "Example: I can't log into my Spotify account "
            "and I forgot my password."
        ),
        height=140,
        label_visibility="collapsed",
    )

    generate_button = st.button(
        "Generate Response",
        type="primary",
        use_container_width=False,
    )

    # -----------------------------------------------------
    # GENERATE
    # -----------------------------------------------------

    if generate_button:

        if not query.strip():

            st.warning(
                "Please enter a customer message before generating a response."
            )

        else:

            with st.spinner("Analyzing customer issue..."):

                try:
                    result = generate_response(
                        query.strip(),
                        top_k=5,
                    )

                    st.session_state["latest_result"] = result
                    st.session_state["latest_query"] = query.strip()

                except Exception as error:

                    st.error(
                        "The support agent could not process this request."
                    )

                    st.exception(error)

    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    if "latest_result" in st.session_state:

        result = st.session_state["latest_result"]
        latest_query = st.session_state.get(
            "latest_query",
            query,
        )

        md(
            """
            <div class="section-title">
                AI-generated response
            </div>
            """
        )

        render_response(result)

        # Copy-friendly plain response
        with st.expander("View response text"):

            st.text_area(
                "Response",
                value=result["response"],
                height=130,
                label_visibility="collapsed",
            )

        # -------------------------------------------------
        # AI ANALYSIS
        # -------------------------------------------------

        md(
            """
            <div class="section-title">
                AI analysis
            </div>

            <div class="section-description">
                Signals used by the agent to produce the response.
            </div>
            """
        )

        render_analysis(result)

        # -------------------------------------------------
        # WHY
        # -------------------------------------------------

        md(
            """
            <div class="section-title">
                Why this response?
            </div>
            """
        )

        render_why_response()

        # -------------------------------------------------
        # EVIDENCE
        # -------------------------------------------------

        md(
            """
            <div class="section-title">
                Historical support evidence
            </div>

            <div class="section-description">
                The most semantically similar support conversations
                retrieved from the training set.
            </div>
            """
        )

        render_evidence(result["sources"])

        md(
            """
            <div style="
                margin-top:28px;
                padding:14px 16px;
                border-top:1px solid #e5e7eb;
                color:#7b8490;
                font-size:12px;
            ">
                Safety note: customer passwords and sensitive account
                credentials should never be requested or displayed.
            </div>
            """
        )


# =========================================================
# EVALUATION PAGE
# =========================================================

elif page == "Evaluation":

    md(
        """
        <div class="page-kicker">Model Evaluation</div>
        <div class="page-title">Evaluation Dashboard</div>
        <div class="page-description">
            Offline evaluation of intent classification, response
            quality, retrieval confidence, and safety.
        </div>
        """
    )

    evaluation = load_evaluation_results()

    if evaluation is None:

        st.warning(
            "Evaluation results were not found. Run the response-quality "
            "evaluation script first."
        )

    else:

        total = len(evaluation)

        intent_accuracy = (
            evaluation["intent_correct"].mean()
            if "intent_correct" in evaluation.columns
            else 0
        )

        response_quality = (
            evaluation["response_quality"].mean()
            if "response_quality" in evaluation.columns
            else 0
        )

        safety = (
            evaluation["no_sensitive_leakage"].mean()
            if "no_sensitive_leakage" in evaluation.columns
            else 0
        )

        avg_similarity = (
            evaluation["similarity"].mean()
            if "similarity" in evaluation.columns
            else 0
        )

        md(
            """
            <div class="eval-success">
                ✓ Evaluation completed successfully on the golden test set.
            </div>
            """
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            md(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Intent Accuracy</div>
                    <div class="metric-value">
                        {intent_accuracy * 100:.0f}%
                    </div>
                    <div class="metric-description">
                        Correct support-intent classification.
                    </div>
                </div>
                """
            )

        with c2:
            md(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Response Quality</div>
                    <div class="metric-value">
                        {response_quality * 100:.0f}%
                    </div>
                    <div class="metric-description">
                        Intent-appropriate generated responses.
                    </div>
                </div>
                """
            )

        with c3:
            md(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Safety / No Leakage</div>
                    <div class="metric-value">
                        {safety * 100:.0f}%
                    </div>
                    <div class="metric-description">
                        No raw URLs or historical identifiers exposed.
                    </div>
                </div>
                """
            )

        with c4:
            md(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg. Retrieval Similarity</div>
                    <div class="metric-value">
                        {avg_similarity:.4f}
                    </div>
                    <div class="metric-description">
                        Mean semantic similarity of retrieved evidence.
                    </div>
                </div>
                """
            )

        # -------------------------------------------------
        # TEST RESULTS
        # -------------------------------------------------

        md(
            """
            <div class="section-title">
                Test case results
            </div>
            """
        )

        display_columns = [
            "query",
            "expected_intent",
            "predicted_intent",
            "similarity",
            "intent_correct",
            "response_quality",
            "no_sensitive_leakage",
            "overall_pass",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in evaluation.columns
        ]

        if available_columns:

            table = evaluation[available_columns].copy()

            rename_map = {
                "query": "Customer Query",
                "expected_intent": "Expected Intent",
                "predicted_intent": "Predicted Intent",
                "similarity": "Similarity",
                "intent_correct": "Intent Correct",
                "response_quality": "Response Quality",
                "no_sensitive_leakage": "Safety",
                "overall_pass": "Overall Pass",
            }

            table = table.rename(columns=rename_map)

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        md(
            """
            <div class="section-title">
                Evaluation methodology
            </div>
            """
        )

        md(
            """
            <div class="why-card">

                <div class="why-title">
                    Golden test set
                </div>

                <div class="why-text">
                    The evaluation set contains representative support
                    scenarios covering account login, playback, security,
                    subscriptions, playlists, and general support.
                    Each case is checked for intent correctness,
                    response quality, and sensitive-information leakage.
                </div>

            </div>
            """
        )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    md(
        """
        <div class="page-kicker">System Overview</div>
        <div class="page-title">About the AI Support Agent</div>
        <div class="page-description">
            Architecture, methodology, and design principles behind
            the Hiver AI Support Agent.
        </div>
        """
    )

    md(
        """
        <div class="hero-card">

            <div class="hero-title">
                Retrieval-grounded customer support
            </div>

            <div class="hero-text">
                This prototype demonstrates how historical customer-support
                conversations can be transformed into a practical,
                explainable AI-assisted support workflow.
                Rather than generating unsupported answers, the system
                retrieves semantically similar historical conversations
                and combines that evidence with an explicit intent and
                safety layer.
            </div>

        </div>
        """
    )

    md(
        """
        <div class="section-title">
            Architecture
        </div>
        """
    )

    architecture = [
        "Customer Support Dataset",
        "Data Cleaning",
        "Conversation Pair Extraction",
        "Train / Validation / Test Split",
        "Sentence Transformer",
        "384-D Embeddings",
        "Semantic Retrieval",
        "Intent Detection",
        "Safety Layer",
        "Response Generation",
        "Evaluation",
        "Streamlit Support Workspace",
    ]

    for i, step in enumerate(architecture):

        md(
            f"""
            <div class="architecture-step">
                {i + 1:02d}
                &nbsp;&nbsp;
                {safe_text(step)}
            </div>
            """
        )

    md(
        """
        <div class="section-title">
            Key design principles
        </div>
        """
    )

    principles = [
        (
            "Retrieval grounded",
            "Responses are informed by semantically similar historical support conversations."
        ),
        (
            "Explainable",
            "The interface exposes detected intent, retrieval confidence, and supporting evidence."
        ),
        (
            "Safety aware",
            "Sensitive credentials are not requested, and historical identifiers are sanitized."
        ),
        (
            "Deterministic",
            "Intent detection is implemented using a transparent rule-based layer."
        ),
        (
            "Evaluated",
            "A golden test set measures intent accuracy, response quality, and safety."
        ),
    ]

    for title, description in principles:

        md(
            f"""
            <div class="why-card" style="margin-bottom:12px;">

                <div class="why-title">
                    {safe_text(title)}
                </div>

                <div class="why-text">
                    {safe_text(description)}
                </div>

            </div>
            """
        )

    md(
        """
        <div style="
            margin-top:35px;
            padding-top:20px;
            border-top:1px solid #e5e7eb;
            color:#7b8490;
            font-size:13px;
            line-height:1.7;
        ">
            Hiver AI Support Agent<br>
            Retrieval-Grounded Support Assistant<br>
            Built for SpotifyCares support workflows
        </div>
        """
    )

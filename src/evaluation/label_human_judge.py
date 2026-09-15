import streamlit as st
import pandas as pd
from pathlib import Path

# Always resolve paths from the project root
ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "evaluation" / "llm_judge_results.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "evaluation" / "human_judge_results.csv"

st.set_page_config(
    page_title="Human Response Judge",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Human Judge — Response Quality")
st.write(
    "Rate the quality of the AI-generated support response using the same 1–5 rubric."
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if not INPUT_PATH.exists():
    st.error(f"Input file not found:\n{INPUT_PATH}")
    st.stop()

if OUTPUT_PATH.exists():
    df = pd.read_csv(OUTPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH)

# Make sure required column exists
if "human_score" not in df.columns:
    df["human_score"] = ""

df["human_score"] = df["human_score"].fillna("").astype(str)

# ---------------------------------------------------------
# PROGRESS
# ---------------------------------------------------------

labelled = df["human_score"].str.strip().ne("").sum()
total = len(df)

st.progress(labelled / total if total else 0)

st.write(f"**Progress: {labelled}/{total} cases judged**")

if labelled == total:
    st.success("🎉 All cases have been human-judged!")

    st.download_button(
        "Download human judge results",
        data=df.to_csv(index=False),
        file_name="human_judge_results.csv",
        mime="text/csv"
    )

    st.stop()

# ---------------------------------------------------------
# CURRENT CASE
# ---------------------------------------------------------

unlabelled = df.index[df["human_score"].str.strip() == ""]

row_index = unlabelled[0]
row = df.loc[row_index]

st.divider()

st.subheader(f"Case {int(row['case_id'])}")

# Customer query
st.markdown("### Customer message")
st.info(str(row.get("customer_query", "")))

# Intent
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Expected intent")
    st.code(str(row.get("expected_intent", "")))

with col2:
    st.markdown("### Predicted intent")
    st.code(str(row.get("predicted_intent", "")))

# Existing automated checks
st.markdown("### Automated evaluation")
st.write(
    f"Intent correct: **{row.get('intent_correct', '')}**"
)

st.divider()

# ---------------------------------------------------------
# HUMAN RUBRIC
# ---------------------------------------------------------

st.markdown("### Human response-quality rubric")

st.write("""
**5 — Excellent:** Correct, helpful, grounded in historical support behaviour,
actionable, safe, and appropriate in tone.

**4 — Good:** Helpful and appropriate with only minor omissions.

**3 — Acceptable:** Partially useful but missing important detail or specificity.

**2 — Poor:** Significant problems, weak grounding, or limited usefulness.

**1 — Unacceptable:** Incorrect, unsafe, misleading, or essentially unhelpful.
""")

score = st.radio(
    "Your score",
    ["1", "2", "3", "4", "5"],
    index=None,
    key=f"human_score_{int(row['case_id'])}"
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

if st.button("✅ Save Score & Next", width="stretch"):

    if score is None:
        st.warning("Please select a score before continuing.")
        st.stop()

    df.loc[row_index, "human_score"] = score

    df.to_csv(OUTPUT_PATH, index=False)

    st.success(
        f"Case {int(row['case_id'])} saved with human score {score}/5."
    )

    st.rerun()
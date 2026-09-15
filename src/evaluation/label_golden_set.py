import streamlit as st
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/evaluation/golden_test_cases_clean_200.csv")
OUTPUT_PATH = Path("data/processed/evaluation/golden_test_cases_clean_200_labelled.csv")
INTENTS = [
    "account_login",
    "playback",
    "playlist",
    "subscription",
    "security",
    "general",
]

st.set_page_config(
    page_title="Golden Set Labeler",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Golden Evaluation Set — Human Labeling")

st.write(
    "Review each SpotifyCares customer message and assign the correct intent."
)

# ---------------------------------------------------------
# LOAD SAVED DATA
# ---------------------------------------------------------

if Path(OUTPUT_PATH).exists():
    df = pd.read_csv(OUTPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH)

if "gold_intent" not in df.columns:
    df["gold_intent"] = ""

df["gold_intent"] = df["gold_intent"].fillna("").astype(str)

# ---------------------------------------------------------
# FIND FIRST UNLABELLED CASE
# ---------------------------------------------------------

unlabelled = df[
    df["gold_intent"].str.strip() == ""
]

labelled_count = len(df) - len(unlabelled)
total = len(df)

st.progress(
    labelled_count / total,
    text=f"Progress: {labelled_count}/{total} cases labelled"
)

# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

if labelled_count == total:

    st.success("🎉 All 200 cases have been labelled!")

    st.download_button(
        "Download labelled golden set",
        data=df.to_csv(index=False),
        file_name="golden_test_cases_200_labelled.csv",
        mime="text/csv",
    )

    st.stop()

# ---------------------------------------------------------
# CURRENT CASE
# ---------------------------------------------------------

row_index = unlabelled.index[0]
row = df.loc[row_index]

st.divider()

st.subheader(f"Case {int(row['case_id'])}")

st.markdown("### Customer message")

st.info(str(row["customer_text"]))

st.markdown("### Suggested intent")

st.code(str(row["suggested_intent"]))

st.markdown("### Historical support response")

st.write(str(row["support_text"]))

st.divider()

st.markdown("### Select the correct human label")

suggested = str(row["suggested_intent"])

default_index = (
    INTENTS.index(suggested)
    if suggested in INTENTS
    else 0
)

selected = st.radio(
    "Intent",
    INTENTS,
    index=default_index,
    key=f"label_case_{int(row['case_id'])}"
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

if st.button("✅ Confirm & Next", width="stretch"):

    df.loc[row_index, "gold_intent"] = selected

    # SAVE TO DISK IMMEDIATELY
    df.to_csv(OUTPUT_PATH, index=False)

    st.success(
        f"Case {int(row['case_id'])} saved as '{selected}'."
    )

    st.rerun()

st.caption(
    "Each confirmed label is written to disk immediately. "
    "You can safely restart the app without losing progress."
)
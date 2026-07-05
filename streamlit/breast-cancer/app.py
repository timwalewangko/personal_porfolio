import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("breast_cancer_pipeline.pkl")

# --------------------------------------------------
# Load Default Patient
# --------------------------------------------------

with open("defaults.json", "r") as f:
    defaults = json.load(f)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "patient" not in st.session_state:
    st.session_state.patient = defaults.copy()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction Demo")

st.markdown("""
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a Logistic Regression model trained on the Wisconsin Breast Cancer Dataset.

Modify any measurements below or use the default example patient.
""")

# --------------------------------------------------
# Buttons
# --------------------------------------------------

left, right = st.columns([1, 1])

with left:
    if st.button("Load Example Patient"):
        st.session_state.patient = defaults.copy()
        st.rerun()

with right:
    if st.button("Reset All Values"):
        st.session_state.patient = {k: 0.0 for k in defaults.keys()}
        st.rerun()

st.divider()

# --------------------------------------------------
# Input Fields
# --------------------------------------------------

features = list(defaults.keys())

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    with col1 if i % 2 == 0 else col2:

        st.session_state.patient[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(st.session_state.patient[feature]),
            format="%.5f",
            key=feature,
        )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary"):

    df = pd.DataFrame([st.session_state.patient])

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    benign_prob = probabilities[0]
    malignant_prob = probabilities[1]

    st.header("Prediction")

    if prediction == 1:

        st.error(
            f"### Malignant\n\n" f"The model predicts this tumor is **Malignant**."
        )

    else:

        st.success(f"### Benign\n\n" f"The model predicts this tumor is **Benign**.")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Benign Probability", f"{benign_prob*100:.2f}%")

    with metric2:
        st.metric("Malignant Probability", f"{malignant_prob*100:.2f}%")

    st.subheader("Prediction Confidence")

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Benign", "Malignant"]
    probs = [benign_prob, malignant_prob]

    bars = ax.bar(labels, probs)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Model Confidence")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2%}",
            ha="center",
            fontsize=11,
        )

    st.pyplot(fig)

    st.subheader("Input Summary")

    st.dataframe(df.T.rename(columns={0: "Value"}), use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit • Model: Logistic Regression + PCA • "
    "Dataset: Wisconsin Breast Cancer Diagnostic Dataset"
)

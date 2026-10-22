
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="From Beans to Bytes",
    page_icon="☕",
    layout="centered"
)

st.title("☕ From Beans to Bytes")
st.subheader("Starbucks Beverage Prediction")
st.write(
    "Enter beverage details to predict its category "
    "using the trained Decision Tree model."
)


# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = (
    Path(__file__).parent / "starbucks_decision_tree.pkl"
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("Trained model file was not found.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    expected_columns = list(preprocessor.feature_names_in_)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# IDENTIFY FEATURE TYPES
# ==========================================
numeric_features = []
categorical_features = []
categorical_options = {}

for name, transformer, columns in preprocessor.transformers_:

    if name == "numeric":
        numeric_features = list(columns)

    elif name == "categorical":
        categorical_features = list(columns)

        encoder = transformer.named_steps.get("encoder")

        if encoder is not None and hasattr(encoder, "categories_"):
            for col, options in zip(columns, encoder.categories_):
                categorical_options[col] = [
                    value for value in options
                    if pd.notna(value)
                ]


# ==========================================
# USER INPUT FORM
# ==========================================
st.markdown("### Beverage Measurements")

with st.form("beverage_form"):
    input_values = {}

    if numeric_features:
        st.markdown("#### Nutritional Information")

        for col in numeric_features:
            label = col.replace("_", " ").title()

            input_values[col] = st.number_input(
                label,
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"num_{col}"
            )

    if categorical_features:
        st.markdown("#### Beverage Details")

        for col in categorical_features:
            label = col.replace("_", " ").title()
            options = categorical_options.get(col, [])

            if options:
                selected = st.selectbox(
                    label,
                    options=options,
                    key=f"cat_{col}"
                )
                input_values[col] = selected
            else:
                input_values[col] = st.text_input(
                    label,
                    key=f"text_{col}"
                )

    submitted = st.form_submit_button(
        "Predict Beverage Category",
        type="primary",
        use_container_width=True
    )


# ==========================================
# PREDICT
# ==========================================
if submitted:

    input_df = pd.DataFrame(
        [[input_values[col] for col in expected_columns]],
        columns=expected_columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            st.metric(
                "Model Confidence",
                f"{probabilities.max() * 100:.1f}%"
            )

            probability_df = pd.DataFrame({
                "Category": classifier.classes_,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=False
            )

            st.markdown("### Category Probabilities")
            st.bar_chart(
                probability_df.set_index("Category")["Probability"]
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("From Beans to Bytes | Starbucks Data Analysis Project")

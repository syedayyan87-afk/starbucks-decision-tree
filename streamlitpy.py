
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


# ==========================================
# HEADER
# ==========================================
st.title("☕ From Beans to Bytes")
st.subheader("Starbucks Beverage Category Prediction")

st.write(
    "Enter the nutritional measurements below "
    "to predict the beverage category."
)


# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = (
    Path(__file__).parent
    / "starbucks_decision_tree.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please upload "
        "starbucks_decision_tree.pkl to the same folder."
    )
    st.stop()

try:
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ==========================================
# GET MODEL FEATURES
# ==========================================
numeric_features = []
categorical_features = []
categorical_defaults = {}

for name, transformer, columns in preprocessor.transformers_:
    if name == "numeric":
        numeric_features = list(columns)

    elif name == "categorical":
        categorical_features = list(columns)
        imputer = transformer.named_steps.get("imputer")

        if hasattr(imputer, "statistics_"):
            categorical_defaults = dict(
                zip(columns, imputer.statistics_)
            )


# ==========================================
# INPUT FORM
# ==========================================
st.markdown("### Beverage Measurements")

with st.form("prediction_form"):
    input_values = {}

    for col in numeric_features:
        input_values[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"input_{col}"
        )

    # Use training defaults for categorical features
    for col in categorical_features:
        input_values[col] = categorical_defaults.get(col, "")

    submitted = st.form_submit_button(
        "Predict Beverage Category",
        type="primary",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================
if submitted:
    input_df = pd.DataFrame(
        [[input_values[col] for col in expected_columns]],
        columns=expected_columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted Beverage Category: {prediction}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("From Beans to Bytes | Starbucks Data Analysis Project")

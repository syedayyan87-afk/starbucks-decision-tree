
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Starbucks Beverage Predictor",
    page_icon="☕",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "starbucks_decision_tree.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean_column_name(column):
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


st.title("☕ My Starbucks Decision Tree Model")
st.write("Enter the nutritional values to predict the beverage category.")


if not MODEL_PATH.exists():
    st.error("Model file not found.")
    st.stop()

try:
    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    expected_columns = list(preprocessor.feature_names_in_)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# Show only the numerical features as simple input fields
numeric_features = []

for name, transformer, columns in preprocessor.transformers_:
    if name == "numeric":
        numeric_features = list(columns)
        break

if not numeric_features:
    st.error("No numerical input features found in the model.")
    st.stop()


st.subheader("Beverage Measurements")

with st.form("prediction_form"):
    values = {}

    for col in numeric_features:
        label = col.replace("_", " ").title()
        values[col] = st.number_input(
            label,
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

    submitted = st.form_submit_button("Predict Beverage Category")

if submitted:
    # Build a row containing all features expected by the model.
    # Categorical features use their most frequent training value.
    input_data = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            for col in columns:
                input_data[col] = values[col]

        elif name == "categorical":
            imputer = transformer.named_steps.get("imputer")

            if imputer is not None and hasattr(imputer, "statistics_"):
                for col, default_value in zip(columns, imputer.statistics_):
                    input_data[col] = default_value
            else:
                for col in columns:
                    input_data[col] = ""

    input_df = pd.DataFrame(
        [[input_data[col] for col in expected_columns]],
        columns=expected_columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Beverage Category: **{prediction}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

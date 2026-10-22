
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Starbucks Decision Tree",
    page_icon="☕",
    layout="wide"
)

st.title("☕ From Beans to Bytes")
st.subheader("Starbucks Beverage Category Prediction")

st.write(
    "Upload your Starbucks dataset to predict beverage "
    "categories using a trained Decision Tree."
)


# ==========================================
# MODEL LOADING
# ==========================================
MODEL_PATH = (
    Path(__file__).parent
    / "starbucks_decision_tree.pkl"
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please upload "
        "'starbucks_decision_tree.pkl' to the same "
        "folder as this Python file."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

if not hasattr(model, "named_steps"):
    st.error(
        "The saved model does not contain the expected "
        "preprocessing pipeline."
    )
    st.stop()

if (
    "preprocessor" not in model.named_steps
    or "classifier" not in model.named_steps
):
    st.error(
        "The model must contain 'preprocessor' and "
        "'classifier' pipeline steps."
    )
    st.stop()

st.success("Trained model loaded successfully!")


# ==========================================
# COLUMN NAME CLEANING
# ==========================================
def clean_column_name(column):
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


# ==========================================
# CSV UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Upload Starbucks CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        # Clean incoming column names
        uploaded_df.columns = [
            clean_column_name(col)
            for col in uploaded_df.columns
        ]

        # Avoid ambiguous duplicated column names
        if uploaded_df.columns.duplicated().any():
            st.error(
                "Duplicate column names were found after "
                "cleaning. Please check your CSV headers."
            )
            st.stop()

        st.write("### Dataset Preview")
        st.dataframe(uploaded_df.head())

        preprocessor = model.named_steps["preprocessor"]

        expected_columns = list(
            preprocessor.feature_names_in_
        )

        # Check required model input features
        missing_columns = [
            col for col in expected_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                "Some required model features are missing:"
            )
            st.write(missing_columns)

            st.info(
                "Upload the original Starbucks nutrition "
                "CSV with the same features used during "
                "model training."
            )

            st.write("### Columns detected in your CSV")
            st.write(list(uploaded_df.columns))

        else:
            # Use exactly the features and order expected
            # by the trained pipeline
            X = uploaded_df[expected_columns].copy()

            # Match training-time Boolean conversion
            for col in X.select_dtypes(
                include=["bool"]
            ).columns:
                X[col] = X[col].astype(str)

            st.write(
                f"Ready to predict {len(X)} beverage records."
            )

            if st.button(
                "Predict Beverage Categories",
                type="primary"
            ):
                predictions = model.predict(X)

                results = uploaded_df.copy()
                results["predicted_category"] = predictions

                st.write("### Prediction Results")
                st.dataframe(results, use_container_width=True)

                # Download prediction results
                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="starbucks_predictions.csv",
                    mime="text/csv"
                )

                # Optional: show prediction counts
                st.write("### Predicted Category Counts")
                st.bar_chart(
                    results["predicted_category"]
                    .value_counts()
                )

    except Exception as e:
        st.error(
            f"Could not process the uploaded CSV: {e}"
        )

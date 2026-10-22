
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Starbucks Decision Tree",
    page_icon="☕",
    layout="wide"
)

st.title("☕ From Beans to Bytes")
st.subheader("Starbucks Beverage Category Prediction")

st.write(
    "Upload your Starbucks nutrition dataset to predict "
    "beverage categories using a trained Decision Tree."
)


# ==========================================
# 2. MODEL LOADING
# ==========================================
MODEL_PATH = (
    Path(__file__).parent / "starbucks_decision_tree.pkl"
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please upload "
        "'starbucks_decision_tree.pkl' to the same folder "
        "as this Python file."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

if not hasattr(model, "named_steps"):
    st.error("The saved model is not the expected pipeline.")
    st.stop()

if not {"preprocessor", "classifier"}.issubset(
    model.named_steps.keys()
):
    st.error(
        "Model must contain 'preprocessor' and "
        "'classifier' steps."
    )
    st.stop()

preprocessor = model.named_steps["preprocessor"]

if not hasattr(preprocessor, "feature_names_in_"):
    st.error(
        "The model does not contain its original feature names."
    )
    st.stop()

expected_columns = list(preprocessor.feature_names_in_)

st.success("Trained model loaded successfully!")


# ==========================================
# 3. COLUMN NAME CLEANING
# ==========================================
def clean_column_name(column):
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


# ==========================================
# 4. NUMERIC DATA CLEANING
# ==========================================
def clean_numeric_column(series):
    """
    Convert numeric text into numbers.
    Invalid values such as '3 2' become NaN rather
    than being guessed as 32 or another number.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(
            r"(?<=\d)\s*(mg|mcg|g|ml|oz)$",
            "",
            regex=True,
            case=False
        )
        .str.strip()
    )

    # Only remove internal spaces when the remaining
    # value is made entirely of digits and spaces.
    # Example: "3 2" becomes "32".
    # Remove this conversion if such values may be typos.
    digit_space_pattern = r"\d[\d\s]*"

    mask = cleaned.str.fullmatch(
        digit_space_pattern,
        na=False
    )

    cleaned.loc[mask] = (
        cleaned.loc[mask].str.replace(
            r"\s+", "", regex=True
        )
    )

    return pd.to_numeric(cleaned, errors="coerce")


# ==========================================
# 5. CSV UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Upload Starbucks CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        # Normalize column names
        uploaded_df.columns = [
            clean_column_name(col)
            for col in uploaded_df.columns
        ]

        if uploaded_df.columns.duplicated().any():
            st.error(
                "Duplicate column names were found after "
                "cleaning. Please check your CSV headers."
            )
            st.stop()

        st.write("### Dataset Preview")
        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        st.write(
            f"**Rows:** {len(uploaded_df)} | "
            f"**Columns:** {len(uploaded_df.columns)}"
        )

        # ==========================================
        # 6. VALIDATE REQUIRED FEATURES
        # ==========================================
        missing_columns = [
            col for col in expected_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                "Required feature columns are missing:"
            )
            st.write(missing_columns)

            st.info(
                "Please upload the original Starbucks CSV "
                "with the same features used during training."
            )

            st.write("### Columns detected in uploaded CSV")
            st.write(list(uploaded_df.columns))

        else:
            # Use features in the exact training order
            X = uploaded_df[expected_columns].copy()

            # ==========================================
            # 7. CLEAN NUMERIC FEATURES
            # ==========================================
            numeric_features = []

            for name, transformer, columns in (
                preprocessor.transformers_
            ):
                if name == "numeric":
                    numeric_features = list(columns)
                    break

            converted_columns = []

            for col in numeric_features:
                original = X[col].copy()
                converted = clean_numeric_column(original)

                invalid_mask = (
                    original.notna()
                    & original.astype(str).str.strip().ne("")
                    & converted.isna()
                )

                if invalid_mask.any():
                    converted_columns.append(
                        (col, int(invalid_mask.sum()))
                    )

                X[col] = converted

            if converted_columns:
                with st.expander(
                    "Data cleaning details"
                ):
                    st.write(
                        "Some invalid numeric values were "
                        "converted to missing values and will "
                        "be handled by the model's imputer."
                    )

                    for col, count in converted_columns:
                        st.write(
                            f"- {col}: {count} invalid value(s)"
                        )

            # Convert boolean features consistently
            for col in X.select_dtypes(
                include=["bool"]
            ).columns:
                X[col] = X[col].astype(str)

            st.write(
                f"**Records ready for prediction:** {len(X)}"
            )

            # ==========================================
            # 8. PREDICTIONS
            # ==========================================
            if st.button(
                "Predict Beverage Categories",
                type="primary"
            ):

                predictions = model.predict(X)

                results = uploaded_df.copy()
                results["predicted_category"] = predictions

                st.success("Predictions completed successfully!")

                st.write("### Prediction Results")
                st.dataframe(
                    results,
                    use_container_width=True
                )

                # Category counts
                st.write("### Predicted Category Counts")
                st.bar_chart(
                    results["predicted_category"]
                    .value_counts()
                )

                # Download results
                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="starbucks_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(
            f"Could not process the uploaded CSV: {e}"
        )

st.caption(
    "From Beans to Bytes | Starbucks Data Analysis Project"
)

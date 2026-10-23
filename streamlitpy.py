from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="From Beans to Bytes",
    page_icon="☕",
    layout="centered"
)

# Custom Starbucks Green UI Styling
st.markdown("""
    <style>
    .main-title {
        color: #00704A;
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #555555;
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton > button {
        background-color: #00704A;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #005538;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">☕ From Beans to Bytes</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Starbucks Beverage Prediction & Analysis App</div>', unsafe_allow_html=True)
st.divider()


# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = Path(__file__).parent / "starbucks_decision_tree.pkl"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error("Trained model file ('starbucks_decision_tree.pkl') was not found in the repository.")
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
                    value for value in options if pd.notna(value)
                ]


# ==========================================
# USER INPUT FORM (UI)
# ==========================================
st.markdown("### 📝 Enter Beverage Measurements")

with st.form("beverage_form"):
    input_values = {}

    if numeric_features:
        st.markdown("#### 📊 Nutritional Information")
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
        st.markdown("#### 🏷️ Beverage Details")
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
        "🚀 Predict Beverage Category",
        use_container_width=True
    )


# ==========================================
# PREDICTION LOGIC
# ==========================================
if submitted:
    input_df = pd.DataFrame(
        [[input_values[col] for col in expected_columns]],
        columns=expected_columns
    )

    try:
        prediction = model.predict(input_df)[0]
        
        st.success(f"🎉 **Predicted Category:** {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            st.metric(
                label="Model Confidence",
                value=f"{probabilities.max() * 100:.1f}%"
            )

            probability_df = pd.DataFrame({
                "Category": classifier.classes_,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=False
            )

            st.markdown("### 📈 Category Probabilities Breakdown")
            st.bar_chart(
                probability_df.set_index("Category")["Probability"]
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==========================================
# EDA / DATA INSIGHTS SECTION (Optional)
# ==========================================
st.divider()
with st.expander("🔍 View Project Dataset EDA Insights"):
    st.write("This application uses a Decision Tree model trained on Starbucks beverage data to classify and predict drink types based on nutritional features.")
    
    # Try loading csv if available in repo
    csv_path = Path(__file__).parent / "starbucks.csv"
    if csv_path.exists():
        try:
            df_eda = pd.read_csv(csv_path)
            st.subheader("Top Highest Calorie Drinks Sample")
            if "Calories" in df_eda.columns and "Beverage" in df_eda.columns:
                st.dataframe(df_eda.nlargest(5, "Calories")[["Beverage", "Calories"]], use_container_width=True)
        except Exception:
            pass
    else:
        st.info("Tip: You can upload your dataset CSV file to GitHub as `starbucks.csv` to display raw data insights here.")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("✨ Built with Streamlit & Machine Learning | From Beans to Bytes")

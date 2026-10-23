from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIGURATION & COLOURFUL UI STYLING
# ==========================================
st.set_page_config(
    page_title="From Beans to Bytes",
    page_icon="☕",
    layout="centered"
)

# Vibrant & Colourful Custom CSS
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
    }

    /* Gradient Title */
    .main-title {
        background: linear-gradient(45deg, #00704A, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #4b5563;
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 600;
    }

    /* Stylish Colourful Predict Button */
    .stButton > button {
        background: linear-gradient(135deg, #00704A 0%, #10b981 100%);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        font-size: 18px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
        transform: translateY(-2px);
    }

    /* Form Card Styling */
    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
    }

    /* Headers inside form */
    h3, h4 {
        color: #1f2937;
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
# FOOTER
# ==========================================
st.divider()
st.caption("✨ Built with Streamlit & Machine Learning | From Beans to Bytes")

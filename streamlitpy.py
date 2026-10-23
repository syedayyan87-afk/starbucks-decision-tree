from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Starbucks Beverage Customizer",
    page_icon="☕",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-title {
        background: linear-gradient(45deg, #00704A, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
    }
    .sub-title {
        color: #9ca3af;
        font-size: 16px;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00704A 0%, #10b981 100%);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
        padding: 0.6rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">☕ Starbucks Customizer & Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Choose your preferred input style: Sliders or Manual Typing</div>', unsafe_allow_html=True)
st.divider()

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = Path(__file__).parent / "starbucks_decision_tree.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("Model file (.pkl) nahi mili GitHub repository mein!")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    expected_columns = list(preprocessor.feature_names_in_)
except Exception as e:
    st.error(f"Model load hone mein masla hua: {e}")
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
                categorical_options[col] = [val for val in options if pd.notna(val)]

# ==========================================
# USER INPUT INTERFACE
# ==========================================
input_values = {}

if categorical_features:
    st.markdown("### 🏷️ Beverage Category:")
    for col in categorical_features:
        label = col.replace("_", " ").title()
        options = categorical_options.get(col, [])
        if options:
            input_values[col] = st.selectbox(label, options=options, key=f"cat_{col}")

if numeric_features:
    st.markdown("### ⚙️ Select Input Method for Nutrition:")
    # User ko choose karne ka option ke slider chahiye ya manual text box
    input_mode = st.radio(
        "Choose Mode:",
        ["Sliders 🎛️", "Manual Typing ⌨️"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    for col in numeric_features:
        label = col.replace("_", " ").title()
        
        if input_mode == "Sliders 🎛️":
            input_values[col] = st.slider(
                f"{label}:",
                min_value=0.0,
                max_value=500.0,
                value=20.0,
                step=1.0,
                key=f"slider_{col}"
            )
        else:
            input_values[col] = st.number_input(
                f"{label}:",
                min_value=0.0,
                value=20.0,
                step=1.0,
                key=f"num_{col}"
            )

st.divider()
submitted = st.button("🚀 Predict Beverage Details")

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
        
        st.success(f"🎉 **Prediction Result:** {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.metric(label="Model Confidence", value=f"{probabilities.max() * 100:.1f}%")

            probability_df = pd.DataFrame({
                "Category": classifier.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)

            st.markdown("### 📈 Probabilities Breakdown")
            st.bar_chart(probability_df.set_index("Category")["Probability"])

    except Exception as e:
        st.error(f"Prediction fail ho gayi: {e}")

st.divider()
st.caption("✨ Built with Streamlit | Sliders & Manual Mode Integrated")

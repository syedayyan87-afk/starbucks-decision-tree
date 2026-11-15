"""From Beans to Bytes — evidence-based Starbucks nutrition explorer + ML demo.

Deploy alongside starbucks.csv and the EXISTING trained Decision Tree .pkl.
The existing model predicts beverage CATEGORY, not nutrition values.
"""
from pathlib import Path
import re
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "starbucks.csv"
MODEL_FILES = [ROOT / "starbucks_decision_tree.pkl", ROOT / "starbucks_decision_tree (1).pkl"]

st.set_page_config(page_title="From Beans to Bytes | Starbucks", page_icon="☕", layout="wide")
st.markdown("""<style>
.block-container {max-width:1100px; padding-top:2rem}
[data-testid="stMetric"] {border:1px solid rgba(140,140,140,.25); border-radius:12px; padding:12px}
</style>""", unsafe_allow_html=True)
st.title("☕ From Beans to Bytes")
st.caption("Starbucks beverage nutrition explorer + original Decision Tree classification demo")

@st.cache_data
def load_data(path):
    raw = pd.read_csv(path)
    # Keep the original headings out of the training model; these are display-only normalized copies.
    raw.columns = raw.columns.str.strip()
    essential = ["Beverage_category", "Beverage", "Beverage_prep", "Calories", "Sugars (g)"]
    absent = [col for col in essential if col not in raw.columns]
    if absent:
        raise ValueError("Missing columns: " + ", ".join(absent))
    raw["Calories"] = pd.to_numeric(raw["Calories"], errors="coerce")
    raw["Sugars (g)"] = pd.to_numeric(raw["Sugars (g)"], errors="coerce")
    raw["_row"] = range(2, len(raw) + 2)  # Actual CSV line number (including header)
    return raw

if not CSV_FILE.is_file():
    st.error("starbucks.csv was not found. Put it next to this Python file in your GitHub repository.")
    st.stop()
try:
    df = load_data(CSV_FILE)
except Exception as exc:
    st.error(f"Unable to read dataset: {exc}")
    st.stop()

st.info("Nutrition figures are from the supplied dataset, **not predictions** or verified current Starbucks menu values. Some rows do not identify a distinct drink size.")
summary1, summary2, summary3 = st.columns(3)
summary1.metric("Dataset records", f"{len(df):,}")
summary2.metric("Unique drink names", df["Beverage"].nunique())
summary3.metric("Drink categories", df["Beverage_category"].nunique())

explore_tab, discover_tab, ml_tab = st.tabs(["☕ Explore nutrition", "🔎 Find a drink", "🌳 Decision Tree ML demo"])

with explore_tab:
    st.subheader("Select a beverage")
    drink = st.selectbox("Beverage", sorted(df["Beverage"].dropna().unique()), key="explore_drink")
    options = df[df["Beverage"] == drink]
    prep = st.selectbox("Preparation (as recorded in CSV)", sorted(options["Beverage_prep"].dropna().unique()), key="explore_prep")
    matches = options[options["Beverage_prep"] == prep].copy()
    st.write(f"**Category/categories in dataset:** {', '.join(sorted(matches['Beverage_category'].unique()))}")
    if len(matches) > 1:
        st.warning(f"{len(matches)} different dataset rows match this beverage and preparation. The CSV does not distinguish their sizes or variants sufficiently; no single exact nutrition value can be inferred.")
        st.write("**Nutrition range across matching records:**")
        fields = [c for c in ["Calories", "Sugars (g)", "Total Fat (g)", "Protein (g)", "Caffeine (mg)"] if c in matches.columns]
        summary = matches[fields].apply(pd.to_numeric, errors="coerce").agg(["min", "max"]).T
        summary.columns = ["Minimum", "Maximum"]
        st.dataframe(summary, use_container_width=True)
        st.write("**Individual matching dataset records:**")
        shown = ["_row", "Beverage_category", "Beverage", "Beverage_prep"] + fields
        st.dataframe(matches[shown].rename(columns={"_row": "CSV row"}), hide_index=True, use_container_width=True)
    else:
        r = matches.iloc[0]
        a,b,c = st.columns(3)
        a.metric("Calories", f"{r['Calories']:g} kcal" if pd.notna(r['Calories']) else "Unknown")
        b.metric("Sugar", f"{r['Sugars (g)']:g} g" if pd.notna(r['Sugars (g)']) else "Unknown")
        fat = pd.to_numeric(r.get("Total Fat (g)"), errors="coerce")
        c.metric("Total fat", f"{fat:g} g" if pd.notna(fat) else "Unknown")
        st.dataframe(matches.drop(columns=["_row"]), hide_index=True, use_container_width=True)

with discover_tab:
    st.subheader("Find lower-calorie or lower-sugar records")
    st.caption("These are straightforward dataset filters, not AI recommendations or medical advice.")
    category_opts = ["All categories"] + sorted(df["Beverage_category"].dropna().unique())
    category = st.selectbox("Category filter", category_opts)
    max_cal = st.slider("Maximum calories (kcal)", 0, max(100, int(df["Calories"].max())), 200, step=10)
    max_sugar = st.slider("Maximum sugar (g)", 0, max(20, int(df["Sugars (g)"].max())), 25, step=5)
    subset = df[(df["Calories"] <= max_cal) & (df["Sugars (g)"] <= max_sugar)].copy()
    if category != "All categories":
        subset = subset[subset["Beverage_category"] == category]
    subset = subset.sort_values(["Calories", "Sugars (g)", "Beverage"])
    st.write(f"**{len(subset)} dataset records match your filters.**")
    columns = ["Beverage_category", "Beverage", "Beverage_prep", "Calories", "Sugars (g)"]
    st.dataframe(subset[columns], hide_index=True, use_container_width=True)
    st.download_button("Download filtered records", subset[columns].to_csv(index=False).encode("utf-8"), file_name="starbucks_filtered.csv", mime="text/csv")

with ml_tab:
    st.subheader("Original Decision Tree: predict beverage category")
    st.caption("This is the original classification task. It predicts category from recorded drink and nutrition features; it does NOT predict calories or sugar.")
    st.warning("Notebook reported **44.90% held-out accuracy (22 / 49)**. Predictions are for educational demonstration, not reliable beverage identification. A probability score is not a calibrated accuracy guarantee.")
    model_path = next((p for p in MODEL_FILES if p.is_file()), None)
    if model_path is None:
        st.warning("No .pkl file present. Add starbucks_decision_tree.pkl beside this script to enable the ML demo. Nutrition explorer still works.")
    else:
        @st.cache_resource
        def load_model(path):
            # Only load a pickle created by you / a trusted source. Never allow arbitrary user uploads.
            import joblib
            return joblib.load(path)
        try:
            model = load_model(str(model_path))
            expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)
            # CSV column stripping above is for display; normalize ML input names back to training names.
            # Original training notebook stripped headings at the data-cleaning stage.
            demo_drink = st.selectbox("Choose a recorded beverage", sorted(df["Beverage"].unique()), key="demo_drink")
            candidate_rows = df[df["Beverage"] == demo_drink].copy()
            candidate_rows["_label"] = candidate_rows.apply(
                lambda r: f"CSV row {int(r['_row'])} · {r['Beverage_prep']} · {r['Calories']:g} kcal · {r['Sugars (g)']:g} g sugar", axis=1
            )
            chosen_label = st.selectbox("Choose an individual dataset record", candidate_rows["_label"].tolist())
            selected = candidate_rows[candidate_rows["_label"] == chosen_label].iloc[[0]]
            if st.button("🌳 Predict category using Decision Tree", type="primary"):
                # Reconstruct the notebook's preprocessing for the selected raw CSV row.
                # The training model expects lowercase underscore-separated column names.
                features = selected.drop(columns=["_row", "_label"], errors="ignore").copy()
                features.columns = [
                    re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                    for c in features.columns
                ]
                # The notebook also stripped strings and converted numeric-looking columns.
                # Use numeric vs categorical feature lists from the fitted preprocessor.
                pre = model.named_steps["preprocessor"]
                numeric_cols = []
                categorical_cols = []
                for name, transformer, cols in pre.transformers_:
                    if name == "numeric":
                        numeric_cols = list(cols)
                    elif name == "categorical":
                        categorical_cols = list(cols)
                missing = [col for col in expected_cols if col not in features.columns]
                if missing:
                    st.error("Required model columns absent after normalization: " + ", ".join(missing))
                else:
                    features = features.reindex(columns=expected_cols)
                    for col in numeric_cols:
                        features[col] = pd.to_numeric(
                            features[col].astype("string")
                            .str.replace(",", "", regex=False)
                            .str.replace(r"(?<=\d)\s*(mg|mcg|g|ml|oz)$", "", regex=True, case=False)
                            .str.replace("%", "", regex=False).str.strip(),
                            errors="coerce"
                        )
                    for col in categorical_cols:
                        features[col] = features[col].map(
                            lambda x: x.strip() if isinstance(x, str) else x
                        )
                    for bool_col in features.select_dtypes(include=["bool"]).columns:
                        features[bool_col] = features[bool_col].astype(str)
                    result = str(model.predict(features)[0])
                    actual = str(selected.iloc[0]["Beverage_category"])
                    st.success(f"Model-predicted category: **{result}**")
                    st.write(f"Original dataset category: **{actual}**")
                    st.write("Comparison: " + ("Matches the dataset label" if result == actual else "Does not match the dataset label"))
                    st.caption("This selected dataset row may have been in model training data. This demonstration is not a separate test-set evaluation.")
        except Exception as exc:
            st.error(f"ML demo unavailable: {exc}")

st.divider()
st.caption("Dataset: user-provided Starbucks CSV (242 historical records). Training notebook and existing .pkl model are unchanged.")

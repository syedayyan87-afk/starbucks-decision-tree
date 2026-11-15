        a, b, c = st.columns(3)
        a.metric("Calories", f"{r['Calories']:g} kcal" if pd.notna(r['Calories']) else "Unknown")
        b.metric("Sugar", f"{r['Sugars (g)']:g} g" if pd.notna(r['Sugars (g)']) else "Unknown")
        fat = r.get("Total Fat (g)")
        c.metric("Total fat", f"{fat:g} g" if pd.notna(fat) else "Unknown")
        st.dataframe(values.reset_index(drop=True), hide_index=True, use_container_width=True)
    else:
        st.info("Several entries share this beverage and preparation. The dataset does not identify every size or variation, so a nutritional range is shown instead of a made-up exact value.")
        limits = values.agg(["min", "max"]).T.rename(columns={"min": "Minimum", "max": "Maximum"})
        st.dataframe(limits, use_container_width=True)
    st.caption("Historical dataset values, not ML predictions or verified current Starbucks menu information.")

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

def build_model_features(records, model):
    """Reconstruct original notebook feature normalization without changing user-facing data."""
    expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)
    features = records.drop(columns=["_row"], errors="ignore").copy()
    features.columns = [re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_") for col in features.columns]
    missing = [col for col in expected_cols if col not in features.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    features = features.reindex(columns=expected_cols)
    numeric_cols, categorical_cols = [], []
    for name, transformer, columns in model.named_steps["preprocessor"].transformers_:
        if name == "numeric":
            numeric_cols = list(columns)
        elif name == "categorical":
            categorical_cols = list(columns)
    for col in numeric_cols:
        features[col] = pd.to_numeric(
            features[col].astype("string")
            .str.replace(",", "", regex=False)
            .str.replace(r"(?<=\d)\s*(mg|mcg|g|ml|oz)$", "", regex=True, case=False)
            .str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )
    for col in categorical_cols:
        features[col] = features[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    for col in features.select_dtypes(include=["bool"]).columns:
        features[col] = features[col].astype(str)
    return features


with ml_tab:
    st.subheader("Decision Tree beverage category demo")
    st.caption("Choose a beverage and preparation. The app retrieves the required nutrition features internally — no spreadsheet rows to select.")
    st.warning("Original notebook test accuracy: 44.90% (22/49). This model predicts beverage CATEGORY, not calories or sugar. Results can be wrong.")
    demo_drink = st.selectbox("Beverage", sorted(df["Beverage"].dropna().unique()), key="demo_drink")
    drink_records = df[df["Beverage"] == demo_drink]
    demo_prep = st.selectbox("Preparation", sorted(drink_records["Beverage_prep"].dropna().unique()), key="demo_prep")
    chosen_records = drink_records[drink_records["Beverage_prep"] == demo_prep].copy()
    st.caption("Model input is assembled automatically using matching historical dataset records.")
    model_path = next((candidate for candidate in MODEL_FILES if candidate.is_file()), None)
    if model_path is None:
        st.warning("Trained model file is missing. Keep starbucks_decision_tree.pkl next to streamlitpy.py.")
    else:
        @st.cache_resource
        def load_model(path):
            import joblib
            # Only load trusted, project-owned model files.
            return joblib.load(path)

        if st.button("🌳 Predict beverage category", type="primary", key="predict_category"):
            try:
                model = load_model(str(model_path))
                input_features = build_model_features(chosen_records, model)
                predictions = pd.Series(model.predict(input_features)).astype(str)
                actual_categories = sorted(chosen_records["Beverage_category"].dropna().astype(str).unique())
                counts = predictions.value_counts()
                if len(chosen_records) > 1:
                    st.info("Multiple historical entries match these choices. The model was evaluated on each matching entry; results may differ because some nutrition details are ambiguous.")
                if len(counts) == 1:
                    st.success("Model-predicted category: **" + str(counts.index[0]) + "**")
                else:
                    st.write("**Model predictions across matching historical entries:**")
                    st.dataframe(pd.DataFrame({"Predicted category": counts.index, "Matching entries": counts.values}), hide_index=True, use_container_width=True)
                st.write("**Dataset category:** " + ", ".join(actual_categories))
                if len(counts) == 1 and len(actual_categories) == 1:
                    if counts.index[0] == actual_categories[0]:
                        st.caption("Prediction matches the dataset label for these entries.")
                    else:
                        st.error("This prediction does not match the dataset category.")
                st.caption("Educational demo: matched records may have appeared in training; this is not a held-out accuracy test.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Dataset: user-provided Starbucks CSV (242 historical records). Training notebook and existing .pkl model are unchanged.")

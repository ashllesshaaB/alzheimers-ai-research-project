import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Alzheimer's AI Risk Stratification",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/final_xgboost_pipeline.pkl"
    )


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/GSE33000_AD_vs_control.csv"
    )


# ==========================================
# LOAD GENE MAPPING
# ==========================================

@st.cache_data
def load_gene_mapping():

    mapping_file = "results/selected_probe_gene_mapping.csv"

    try:
        mapping = pd.read_csv(mapping_file)
        return mapping

    except Exception:
        return pd.DataFrame()


# ==========================================
# LOAD EVERYTHING
# ==========================================

model = load_model()
data = load_data()
gene_mapping = load_gene_mapping()


# ==========================================
# TITLE
# ==========================================

st.title(
    "🧠 Explainable AI-Based Alzheimer's Risk Stratification"
)

st.markdown(
    """
    **Research Prototype**

    This system uses transcriptomic gene-expression data
    from the GSE33000 research dataset to classify samples
    as Alzheimer's disease or non-demented control.
    """
)

st.warning(
    "⚠️ Research prototype only — this system is NOT a clinical diagnostic tool."
)


# ==========================================
# DATASET INFORMATION
# ==========================================

st.header("📊 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Samples",
        "467"
    )

with col2:
    st.metric(
        "Transcriptomic Features",
        "39,280"
    )

with col3:
    st.metric(
        "Selected Features",
        "100"
    )


# ==========================================
# SAMPLE PREDICTION
# ==========================================

st.header("🔬 Sample Prediction")

sample_ids = data.columns[1:]

selected_sample = st.selectbox(
    "Select a GSE33000 sample:",
    sample_ids
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button("🔍 Analyze Sample"):

    # ======================================
    # PREPARE SAMPLE
    # ======================================

    sample = data[
        ["ID_REF", selected_sample]
    ].copy()

    X_sample = sample.drop(
        columns=["ID_REF"]
    ).T

    X_sample.columns = (
        data["ID_REF"]
        .astype(str)
    )


    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        X_sample
    )[0]

    probability = model.predict_proba(
        X_sample
    )[0][1]

    probability_percent = (
        probability * 100
    )


    # ======================================
    # RISK BAND
    # ======================================

    if probability < 0.33:

        risk = "Low"

    elif probability < 0.66:

        risk = "Moderate"

    else:

        risk = "High"


    # ======================================
    # PREDICTION RESULT
    # ======================================

    st.header("📋 Prediction Result")

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        if prediction == 1:

            st.error(
                "Predicted Class: Alzheimer's"
            )

        else:

            st.success(
                "Predicted Class: Non-demented"
            )


    with result_col2:

        st.metric(
            "Model Probability",
            f"{probability_percent:.2f}%"
        )


    with result_col3:

        st.metric(
            "Prototype Risk Band",
            risk
        )


    st.caption(
        "The displayed probability is the model's predicted "
        "probability for the Alzheimer's class. It is not a "
        "clinical probability or diagnosis."
    )


    # ======================================
    # SHAP EXPLANATION
    # ======================================

    st.header("🧠 SHAP Explanation")

    st.info(
        """
        SHAP explains which transcriptomic features contributed
        most strongly to this individual model prediction.

        Positive SHAP values push the prediction toward
        Alzheimer's, while negative SHAP values push the
        prediction toward the non-demented class.
        """
    )


    # ======================================
    # GET PIPELINE COMPONENTS
    # ======================================

    imputer = model.named_steps[
        "imputer"
    ]

    selector = model.named_steps[
        "feature_selection"
    ]

    xgb_model = model.named_steps[
        "model"
    ]


    # ======================================
    # TRANSFORM SAMPLE
    # ======================================

    X_imputed = imputer.transform(
        X_sample
    )

    X_selected = selector.transform(
        X_imputed
    )


    # ======================================
    # GET SELECTED FEATURES
    # ======================================

    selected_indices = selector.get_support(
        indices=True
    )

    original_feature_names = (
        X_sample.columns
    )

    selected_feature_names = (
        original_feature_names[
            selected_indices
        ]
    )


    # ======================================
    # CALCULATE SHAP
    # ======================================

    with st.spinner(
        "Calculating SHAP explanation..."
    ):

        explainer = shap.TreeExplainer(
            xgb_model
        )

        shap_values = explainer.shap_values(
            X_selected
        )


    # ======================================
    # HANDLE SHAP OUTPUT
    # ======================================

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[1]


    shap_values = np.asarray(
        shap_values
    )


    if shap_values.ndim == 2:

        shap_values = shap_values[0]


    # ======================================
    # CREATE EXPLANATION DATAFRAME
    # ======================================

    explanation = pd.DataFrame({

        "Probe": selected_feature_names,

        "SHAP Value": shap_values,

        "Absolute SHAP": np.abs(
            shap_values
        )

    })


    # ======================================
    # ADD GENE NAMES
    # ======================================

    if not gene_mapping.empty:

        mapping = gene_mapping.copy()

        mapping["ID"] = (
            mapping["ID"]
            .astype(str)
        )

        explanation["Probe"] = (
            explanation["Probe"]
            .astype(str)
        )

        explanation = explanation.merge(

            mapping[
                [
                    "ID",
                    "Gene symbol",
                    "Gene title"
                ]
            ],

            left_on="Probe",

            right_on="ID",

            how="left"
        )


        explanation["Feature"] = (
            explanation["Gene symbol"]
            .fillna("")
        )


        explanation.loc[
            explanation["Feature"] == "",
            "Feature"
        ] = (
            "Probe_"
            + explanation["Probe"]
        )


    else:

        explanation["Feature"] = (
            "Probe_"
            + explanation["Probe"]
            .astype(str)
        )


    # ======================================
    # TOP 10 FEATURES
    # ======================================

    st.subheader(
        "Top 10 Features Influencing This Prediction"
    )

    top_features = (
        explanation
        .sort_values(
            "Absolute SHAP",
            ascending=False
        )
        .head(10)
        .copy()
    )


    # ======================================
    # CLEAN DISPLAY TABLE
    # ======================================

    display_table = pd.DataFrame({

        "Feature":
            top_features["Feature"].values,

        "SHAP Contribution":
            top_features["SHAP Value"].values
    })


    display_table["Direction"] = np.where(

        display_table["SHAP Contribution"] > 0,

        "Toward Alzheimer's",

        "Toward Non-demented"
    )


    display_table[
        "SHAP Contribution"
    ] = (
        display_table[
            "SHAP Contribution"
        ]
        .round(4)
    )


    st.dataframe(

        display_table,

        use_container_width=True,

        hide_index=True
    )


    # ======================================
    # FEATURE CONTRIBUTION CHART
    # ======================================

    st.subheader(
        "📊 Feature Contribution to the Prediction"
    )


    chart_data = top_features[
        [
            "Feature",
            "SHAP Value"
        ]
    ].copy()


    chart_data = chart_data.sort_values(
        "SHAP Value"
    )


    chart_data = chart_data.set_index(
        "Feature"
    )


    st.bar_chart(
        chart_data,
        horizontal=True
    )


    # ======================================
    # POSITIVE / NEGATIVE FEATURES
    # ======================================

    positive_features = top_features[
        top_features["SHAP Value"] > 0
    ]

    negative_features = top_features[
        top_features["SHAP Value"] < 0
    ]


    # ======================================
    # SIMPLE INTERPRETATION
    # ======================================

    st.subheader(
        "🔎 How the Model Reached This Prediction"
    )


    st.write(
        f"""
        Among the top 10 contributing features,
        **{len(positive_features)}** pushed the model
        prediction toward Alzheimer's, while
        **{len(negative_features)}** pushed the prediction
        toward the non-demented class.
        """
    )


    # ======================================
    # IMPORTANT SCIENTIFIC NOTE
    # ======================================

    st.info(
        """
        **Important:** SHAP explains how the machine-learning
        model arrived at its prediction. It does not prove that
        a gene causes Alzheimer's disease and should not be
        interpreted as a clinical diagnosis.
        """
    )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.header("🏆 Model Performance")


performance = pd.DataFrame({

    "Model": [

        "Random Forest",

        "XGBoost",

        "LightGBM"

    ],

    "5-Fold CV Accuracy": [

        "91.23% ± 3.24%",

        "92.09% ± 2.81%",

        "91.44% ± 2.67%"

    ],

    "F1 Score": [

        "93.38%",

        "94.10%",

        "93.64%"

    ]

})


st.dataframe(

    performance,

    use_container_width=True,

    hide_index=True
)


# ==========================================
# PROJECT PIPELINE
# ==========================================

st.header("🔄 Project Pipeline")

st.markdown(
    """
    **GSE33000**
    → **467 AD + Control Samples**
    → **39,280 Transcriptomic Features**
    → **Missing Value Handling**
    → **Top 100 Features**
    → **XGBoost**
    → **Prediction Probability**
    → **SHAP Explainability**
    → **Prototype Risk Stratification**
    """
)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Dataset: GSE33000 | "
    "Model: XGBoost | "
    "Explainability: SHAP | "
    "Research prototype — not for clinical diagnosis."
)
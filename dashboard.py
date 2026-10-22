import pandas as pd
import matplotlib.pyplot as plt

print("Creating project dashboard...")

# --------------------------------------------------
# 1. LOAD RISK RESULTS
# --------------------------------------------------

risk_data = pd.read_csv(
    "results/risk_stratification_results.csv"
)

# --------------------------------------------------
# 2. MODEL PERFORMANCE
# --------------------------------------------------

models = [
    "Random Forest",
    "XGBoost",
    "LightGBM"
]

accuracies = [
    88.30,
    90.43,
    89.36
]

# --------------------------------------------------
# 3. RISK DISTRIBUTION
# --------------------------------------------------

risk_counts = (
    risk_data["Risk_Category"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High"],
        fill_value=0
    )
)

# --------------------------------------------------
# 4. CREATE DASHBOARD
# --------------------------------------------------

fig = plt.figure(
    figsize=(14, 9)
)

fig.suptitle(
    "Explainable AI-Based Alzheimer's Risk Stratification",
    fontsize=20,
    fontweight="bold"
)

# --------------------------------------------------
# PANEL 1 — MODEL PERFORMANCE
# --------------------------------------------------

ax1 = plt.subplot(2, 2, 1)

bars = ax1.bar(
    models,
    accuracies
)

ax1.set_title(
    "Model Performance",
    fontsize=14,
    fontweight="bold"
)

ax1.set_ylabel("Accuracy (%)")

ax1.set_ylim(0, 100)

for bar, value in zip(bars, accuracies):

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

# --------------------------------------------------
# PANEL 2 — RISK STRATIFICATION
# --------------------------------------------------

ax2 = plt.subplot(2, 2, 2)

bars = ax2.bar(
    risk_counts.index,
    risk_counts.values
)

ax2.set_title(
    "Model Risk Stratification",
    fontsize=14,
    fontweight="bold"
)

ax2.set_ylabel("Number of Test Samples")

for bar, value in zip(
    bars,
    risk_counts.values
):

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        str(value),
        ha="center",
        fontweight="bold"
    )

# --------------------------------------------------
# PANEL 3 — PROJECT PIPELINE
# --------------------------------------------------

ax3 = plt.subplot(2, 2, 3)

ax3.axis("off")

pipeline_text = """
PROJECT PIPELINE

GSE33000
     ↓
467 AD + Control Samples
     ↓
39,281 Transcriptomic Probes
     ↓
Missing Value Handling
     ↓
Top 100 Features
     ↓
XGBoost Classifier
     ↓
90.43% Test Accuracy
     ↓
SHAP Explainability
     ↓
Risk Probability
     ↓
Low / Moderate / High
"""

ax3.text(
    0.5,
    0.5,
    pipeline_text,
    ha="center",
    va="center",
    fontsize=12,
    family="monospace"
)

# --------------------------------------------------
# PANEL 4 — KEY RESULTS
# --------------------------------------------------

ax4 = plt.subplot(2, 2, 4)

ax4.axis("off")

results_text = f"""
KEY RESULTS

Best Model
XGBoost

Test Accuracy
90.43%

Test Samples
94

Selected Features
100

SHAP
Gene-level explanations

Risk Bands

Low       {risk_counts["Low"]}
Moderate  {risk_counts["Moderate"]}
High      {risk_counts["High"]}
"""

ax4.text(
    0.5,
    0.5,
    results_text,
    ha="center",
    va="center",
    fontsize=13,
    family="monospace"
)

# --------------------------------------------------
# SAVE DASHBOARD
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.savefig(
    "results/final_project_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n===== DASHBOARD COMPLETE =====")
print(
    "Saved: results/final_project_dashboard.png"
)
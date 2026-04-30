import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Improve overall figure style
# -----------------------------
plt.rcParams.update({'font.size': 12})
sns.set_style("white")

# -----------------------------
# Model Performance Data
# -----------------------------
data = {
    "Model": [
        "Random Forest",
        "Logistic Regression",
        "SVM (RBF)",
        "Decision Tree",
        "KNN",
        "XGBoost"
    ],
    "Accuracy": [0.9496, 0.9345, 0.9423, 0.9354, 0.9451, 0.9491],
    "AUC": [0.9355, 0.8718, 0.7964, 0.7624, 0.8536, 0.9303]
}

df = pd.DataFrame(data)
df.set_index("Model", inplace=True)

# -----------------------------
# Plot Heatmap
# -----------------------------
plt.figure(figsize=(9,5))

sns.heatmap(
    df,
    annot=True,
    cmap="viridis",
    linewidths=1,
    linecolor="white",
    fmt=".4f",
    cbar_kws={'label': 'Performance Score'}
)

plt.title("Performance Comparison of Machine Learning Models for Phishing Detection", fontsize=14)
plt.xlabel("Evaluation Metrics")
plt.ylabel("Models")

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------
# Save High Quality Figure
# -----------------------------
plt.savefig("PhishNet_Heatmap.png", dpi=600, bbox_inches='tight')

plt.show()
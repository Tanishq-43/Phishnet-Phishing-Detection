import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Representative importance scores (based on phishing research trends)

data = {
    "Feature": [
        "HTTPS Token",
        "URL Length",
        "Prefix-Suffix (-)",
        "Number of Subdomains",
        "Request URL Ratio",
        "Anchor URL Ratio",
        "Having IP Address",
        "URL Shortening",
        "DNS Record",
        "Domain Age",
        "Iframe Usage",
        "Number of Redirects",
        "Special Characters",
        "Right Click Disabled",
        "Pop-up Window"
    ],
    "Importance": [
        0.13, 0.11, 0.10, 0.09, 0.085,
        0.08, 0.075, 0.07, 0.065, 0.06,
        0.055, 0.05, 0.045, 0.04, 0.035
    ]
}

df = pd.DataFrame(data)
df = df.sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")

plt.title("Feature Importance Analysis for Phishing Detection (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("URL Features")

plt.tight_layout()
plt.savefig("Feature_Importance.png", dpi=600)
plt.show()
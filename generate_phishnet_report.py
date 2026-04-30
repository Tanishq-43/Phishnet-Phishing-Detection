# generate_phishnet_report.py
# Run this once to generate a full evaluation report for PhishNet

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from data_preprocessing import DataPreprocessor


def main():
    # Make sure we are in project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # -------- 1. Load + preprocess data --------
    print("🔄 Loading and preprocessing datasets from 'data/'...")
    pre = DataPreprocessor()
    combined_df = pre.load_and_combine_datasets()

    if combined_df is None or combined_df.empty:
        print("❌ No valid datasets found. Check your 'data' folder.")
        return

    print(f"✅ Combined dataset shape: {combined_df.shape}")

    # (Optional) subsample if huge – comment out if you want full data
    MAX_SAMPLES = 20000
    if combined_df.shape[0] > MAX_SAMPLES:
        combined_df = combined_df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"✂️ Subsampled to {combined_df.shape[0]} rows for fast evaluation.")

    X_train, X_test, y_train, y_test, scaler = pre.preprocess_data(combined_df)
    print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -------- 2. Define models (same as your project) --------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=20
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
    }

    os.makedirs("static", exist_ok=True)

    results = {}

    # -------- 3. Train + evaluate each model --------
    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"🚀 Training model: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cls_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "cls": cls_dict,
            "cm": cm,
        }

        print(f"✅ {name} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

        # ---- Confusion matrix figure (for paper) ----
        plt.figure()
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha="center", va="center")
        cm_path = os.path.join(
            "static",
            f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🖼 Saved confusion matrix: {cm_path}")

        # ---- ROC curve figure (for paper) ----
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (AUC={auc:.4f})")
        roc_path = os.path.join(
            "static",
            f"roc_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🖼 Saved ROC curve: {roc_path}")

    # -------- 4. Build markdown report --------
    lines = []
    lines.append("# PhishNet Model Evaluation Report")
    lines.append("")
    lines.append(
        f"_Generated automatically on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Total samples used: **{combined_df.shape[0]}**")
    lines.append(f"- Features after preprocessing: **{X_train.shape[1]}**")
    lines.append(f"- Train size: **{X_train.shape[0]}**")
    lines.append(f"- Test size: **{X_test.shape[0]}**")
    lines.append("")
    lines.append("**Class encoding:** `0 = Legitimate / Safe`, `1 = Phishing`")
    lines.append("")
    lines.append("## Summary Metrics (All Models)")
    lines.append("")
    lines.append("| Model | Accuracy | AUC |")
    lines.append("|-------|----------|-----|")
    for name, m in results.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['auc']:.4f} |")
    lines.append("")

    # Per–model sections
    for name, m in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Accuracy: **{m['accuracy']:.4f}**")
        lines.append(f"- AUC: **{m['auc']:.4f}**")
        lines.append("")

        cm = m["cm"]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            lines.append("### Confusion Matrix")
            lines.append("")
            lines.append("| Actual \\ Predicted | Safe (0) | Phishing (1) |")
            lines.append("|--------------------|----------|--------------|")
            lines.append(f"| **Safe (0)** | {tn} | {fp} |")
            lines.append(f"| **Phishing (1)** | {fn} | {tp} |")
            lines.append("")

        lines.append("### Classification Report")
        lines.append("")
        lines.append("| Label | Precision | Recall | F1-score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")

        cls_dict = m["cls"]
        for label, stats in cls_dict.items():
            if label == "accuracy":
                continue
            if isinstance(stats, dict):
                prec = stats.get("precision", 0.0)
                rec = stats.get("recall", 0.0)
                f1 = stats.get("f1-score", 0.0)
                sup = int(stats.get("support", 0))
                lines.append(
                    f"| {label} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {sup} |"
                )

        # Figure references
        cm_name = f"static/confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        roc_name = f"static/roc_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        lines.append("")
        lines.append("### Figures")
        lines.append(f"- Confusion Matrix: `{cm_name}`")
        lines.append(f"- ROC Curve: `{roc_name}`")
        lines.append("")

    report_path = os.path.join(project_root, "phishnet_model_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"✅ Report generated at: {report_path}")
    print("   You can open this file, convert it to PDF, or share it as-is with your mentor.")
    print("   All figure images are inside the 'static/' folder.")


if __name__ == "__main__":
    main()

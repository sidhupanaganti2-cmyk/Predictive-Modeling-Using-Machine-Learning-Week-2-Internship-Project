
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

# ==========================================================
# WEEK 2 PROJECT: PREDICTIVE MODELING USING MACHINE LEARNING
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"

for folder in [CHART_DIR, REPORT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# 1. LOAD THE DATASET

def load_data():
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data.copy()
    y = dataset.target.copy()

    # In this dataset:
    # 0 = malignant
    # 1 = benign
    # We convert the target to:
    # 1 = malignant, 0 = benign
    y = (y == 0).astype(int)
    y.name = "malignant"

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Number of records: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Missing feature values: {X.isnull().sum().sum()}")

    print("\nFirst five records:")
    print(X.head())

    print("\nTarget distribution:")
    print(y.map({
        0: "Benign",
        1: "Malignant"
    }).value_counts())

    return X, y


# 2. SPLIT DATA INTO TRAINING AND TESTING SETS

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# 3. DEFINE MACHINE LEARNING MODELS

def create_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    return models


# 4. TRAIN AND EVALUATE MODELS

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    predictions = {}
    probabilities = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC AUC": roc_auc_score(y_test, y_prob),
        })

        predictions[name] = y_pred
        probabilities[name] = y_prob
        fitted_models[name] = model

        print(f"\nClassification report: {name}")
        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "ROC AUC", ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))

    return results_df, predictions, probabilities, fitted_models


# 5. PLOT CONFUSION MATRICES

def plot_confusion_matrices(predictions, y_test):
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )

        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        plt.savefig(
            CHART_DIR / f"confusion_matrix_{filename}.png",
            dpi=300,
        )
        plt.close()


# 6. PLOT ROC CURVES

def plot_roc_curves(probabilities, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, y_prob in probabilities.items():
        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            name=name,
            ax=ax,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "roc_curve_comparison.png",
        dpi=300,
    )
    plt.close()


# 7. PLOT MODEL PERFORMANCE

def plot_model_comparison(results_df):
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
    ]

    plot_df = results_df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
    )

    plt.title("Machine Learning Model Performance")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=10)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "model_performance_comparison.png",
        dpi=300,
    )
    plt.close()


# 8. SAVE THE BEST MODEL AND TEST PREDICTIONS

def save_best_model(
    results_df,
    fitted_models,
    X_test,
    y_test,
):
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    prediction_df = pd.DataFrame({
        "actual_malignant": y_test.to_numpy(),
        "predicted_malignant": y_pred,
        "predicted_malignant_probability": y_prob,
    })

    prediction_df["actual_label"] = prediction_df[
        "actual_malignant"
    ].map({0: "Benign", 1: "Malignant"})

    prediction_df["predicted_label"] = prediction_df[
        "predicted_malignant"
    ].map({0: "Benign", 1: "Malignant"})

    prediction_df.to_csv(
        REPORT_DIR / "best_model_predictions.csv",
        index=False,
    )

    return best_name, model_path


# 9. GENERATE PROJECT SUMMARY

def generate_summary(results_df, best_name, X, y):
    best = results_df.iloc[0]

    report = f"""
WEEK 2: PREDICTIVE MODELING USING MACHINE LEARNING
==================================================

Dataset:
Breast Cancer Wisconsin Diagnostic dataset
Source: Scikit-learn built-in educational dataset

Dataset records: {len(X)}
Number of features: {X.shape[1]}

Target labels:
0 = Benign
1 = Malignant

Models evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

Best model by ROC AUC on the held-out test set:
{best_name}

Test-set performance of selected model:
Accuracy:  {best['Accuracy']:.4f}
Precision: {best['Precision']:.4f}
Recall:    {best['Recall']:.4f}
F1 Score:  {best['F1 Score']:.4f}
ROC AUC:   {best['ROC AUC']:.4f}

INTERPRETATION
--------------
Accuracy measures the fraction of correct predictions.
Precision measures how many predicted positive cases were positive.
Recall measures how many actual positive cases were identified.
F1 Score balances precision and recall.
ROC AUC measures how well the model ranks positive cases above
negative cases across classification thresholds.

IMPORTANT LIMITATIONS
---------------------
This is an educational experiment, not a medical diagnostic tool.
Results depend on the selected dataset and train-test split.
Model selection uses a single held-out test set and is exploratory.
No model should be considered clinically validated based on this
project. Clinical use requires appropriate expert validation.

The selected model is chosen by ROC AUC, not accuracy alone.
"""

    (REPORT_DIR / "project_summary.txt").write_text(
        report.strip(),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)
    print(report)


# 10. MAIN FUNCTION

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = create_models()

    results_df, predictions, probabilities, fitted_models = (
        train_and_evaluate(
            models,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    plot_confusion_matrices(predictions, y_test)
    plot_roc_curves(probabilities, y_test)
    plot_model_comparison(results_df)

    best_name, model_path = save_best_model(
        results_df,
        fitted_models,
        X_test,
        y_test,
    )

    generate_summary(results_df, best_name, X, y)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Charts folder: {CHART_DIR}")
    print(f"Reports folder: {REPORT_DIR}")


if __name__ == "__main__":
    main()

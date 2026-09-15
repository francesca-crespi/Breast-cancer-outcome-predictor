import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compile_metrics_summary(results):
    """Compile model accuracy and ROC-AUC into a dataframe."""
    return pd.DataFrame(
        {
            model_name: {
                "Accuracy": results[model_name]["accuracy"],
                "ROC-AUC": results[model_name]["roc_auc"],
            }
            for model_name in results
        }
    ).T


def plot_confusion_matrices(results):
    """Plot confusion matrices for all evaluated models."""
    for name, model_result in results.items():
        cm = model_result["confusion_matrix"]

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix — {name}")
        plt.show()

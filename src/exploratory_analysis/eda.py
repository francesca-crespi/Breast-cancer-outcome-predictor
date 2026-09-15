import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


def summary_statistics(df):
    """Return descriptive statistics and missing-value counts."""
    eda_summary = df.describe().T
    eda_summary["missing"] = df.isnull().sum()
    return eda_summary


def plot_feature_distributions(df):
    """Plot distributions for the first 10 numeric features."""
    subset_features = df.columns[1:11]
    n_features = len(subset_features)
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for i, feature in enumerate(subset_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(
            data=df,
            x=feature,
            hue="diagnosis",
            kde=True,
            palette={0: "green", 1: "red"},
            alpha=0.5,
        )
        plt.title(feature)
        plt.xlabel("")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Plot the correlation heatmap for numeric predictors."""
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        df.drop(columns="diagnosis").corr(),
        cmap="coolwarm",
        annot=False,
    )
    plt.title("Feature Correlation Heatmap")
    plt.show()


def feature_statistics_by_diagnosis(df):
    """Calculate group means, standard deviations and independent t-tests."""
    diagnosis_groups = df.groupby("diagnosis")
    feature_stats = []

    for feature in df.columns[1:]:
        benign = diagnosis_groups.get_group(0)[feature]
        malignant = diagnosis_groups.get_group(1)[feature]

        t_stat, p_value = ttest_ind(benign, malignant)

        feature_stats.append(
            {
                "Feature": feature,
                "Benign_Mean": benign.mean(),
                "Malignant_Mean": malignant.mean(),
                "Benign_STD": benign.std(),
                "Malignant_STD": malignant.std(),
                "Difference": malignant.mean() - benign.mean(),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(feature_stats).sort_values(
        by="Difference",
        key=abs,
        ascending=False,
    )

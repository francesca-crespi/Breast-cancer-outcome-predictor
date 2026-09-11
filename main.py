from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.extend(
    [
        str(Path(__file__).resolve().parent / "01_data_collection"),
        str(Path(__file__).resolve().parent / "02_data_cleaning"),
        str(Path(__file__).resolve().parent / "03_eda"),
        str(Path(__file__).resolve().parent / "04_modeling"),
        str(Path(__file__).resolve().parent / "05_results"),
    ]
)

from load_data import load_breast_cancer
from clean_data import clean_breast_cancer_data
from exploratory_analysis import (
    summary_statistics,
    plot_feature_distributions,
    plot_correlation_heatmap,
    feature_statistics_by_diagnosis,
)
from train_models import (
    prepare_data,
    train_models,
    evaluate_models,
    random_forest_feature_importance,
)
from results import compile_metrics_summary, plot_confusion_matrices


DATA_ZIP_PATH = PROJECT_ROOT / "data" / "archive.zip"
EXTRACT_DIR = PROJECT_ROOT / "data_extracted"


def main():
    # 1. Data collection
    df = load_breast_cancer(DATA_ZIP_PATH, EXTRACT_DIR)

    # 2. Data cleaning
    df_clean = clean_breast_cancer_data(df)

    # 3. Exploratory data analysis
    print("\nSummary statistics:")
    print(summary_statistics(df_clean))

    plot_feature_distributions(df_clean)
    plot_correlation_heatmap(df_clean)

    feature_stats_df = feature_statistics_by_diagnosis(df_clean)
    print("\nTop 10 features by difference between diagnosis groups:")
    print(feature_stats_df.head(10))

    # 4. Model building and evaluation
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df_clean)
    models = train_models(X_train_scaled, y_train)
    results = evaluate_models(models, X_test_scaled, y_test)

    random_forest_feature_importance(
        models["Random Forest"],
        X.columns,
    )

    # 5. Results
    metrics_summary = compile_metrics_summary(results)
    print("\nModel Evaluation Summary:")
    print(metrics_summary)

    plot_confusion_matrices(results)


if __name__ == "__main__":
    main()

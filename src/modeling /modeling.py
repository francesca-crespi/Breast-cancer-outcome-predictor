import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df, test_size=0.2, random_state=42):
    """Split features and diagnosis into stratified train/test sets."""
    X = df.drop(columns="diagnosis")
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training shape:", X_train_scaled.shape)
    print("Testing shape:", X_test_scaled.shape)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test


def train_models(X_train_scaled, y_train):
    """Train Random Forest and Logistic Regression models."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        print(f"{name} trained successfully")

    return models


def evaluate_models(models, X_test_scaled, y_test):
    """Evaluate models and return classification metrics and confusion matrices."""
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )
        acc = report["accuracy"]
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": acc,
            "roc_auc": auc,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        print(f"\n{name} Metrics:")
        print("Accuracy:", acc)
        print("ROC-AUC:", auc)
        print("Classification Report:\n", results[name]["classification_report"])
        print("Confusion Matrix:\n", results[name]["confusion_matrix"])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}")
        plt.legend()
        plt.show()

    return results


def random_forest_feature_importance(model, feature_names, top_n=15):
    """Plot the top Random Forest feature importances."""
    importances = __import__("pandas").DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importances.head(top_n),
        x="Importance",
        y="Feature",
    )
    plt.title("Top 15 Feature Importances — Random Forest")
    plt.show()

    return importances

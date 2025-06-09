# Model training and evaluation script
# evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import main as load_data

def main():
    # 1. LOAD PREPROCESSED DATA
    X_train, X_test, y_train, y_test = load_data()

    # 2. DEFINE MODELS
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    # 3. TRAIN & COLLECT METRICS
    comparison = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        comparison["Model"].append(name)
        comparison["Accuracy"].append(np.round(model.score(X_test, y_test), 3))
        comparison["Precision"].append(np.round(precision_score(y_test, y_pred, average="weighted"), 3))
        comparison["Recall"].append(np.round(recall_score(y_test, y_pred, average="weighted"), 3))
        comparison["F1 Score"].append(np.round(f1_score(y_test, y_pred, average="weighted"), 3))

    results_df = pd.DataFrame(comparison)

    # 4. PLOT COMPARISON
    results_melted = pd.melt(results_df, id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=results_melted)
    plt.title("Model Comparison Across Metrics")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

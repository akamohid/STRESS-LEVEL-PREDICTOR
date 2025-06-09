# Performance metrics plotting script
# modeling.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ConfusionMatrixDisplay
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from data_preprocessing import main as load_data
import matplotlib.pyplot as plt

def main():
    # 1. LOAD PREPROCESSED DATA
    X_train, X_test, y_train, y_test = load_data()

    # 2. TRAIN DECISION TREE
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # 3. EVALUATION REPORT
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4. CONFUSION MATRIX
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    # 5. BAR PLOT OF METRICS
    dt_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted")
    }
    plt.figure(figsize=(8, 5))
    plt.bar(dt_metrics.keys(), dt_metrics.values())
    plt.ylim(0, 1)
    plt.title("Decision Tree Performance Metrics")
    plt.ylabel("Score")
    plt.grid(True, axis="y")
    plt.show()

if __name__ == "__main__":
    main()

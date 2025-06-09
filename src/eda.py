# Exploratory Data Analysis script
# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    # 1. LOAD RAW DATA
    df = pd.read_csv("../data/Dataset.csv")

    # 2. SUMMARY STATS & DISTRIBUTION
    print(df.describe())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="stress_level", data=df, palette="viridis")
    plt.title("Stress Level Distribution")
    plt.show()

    # 3. CORRELATION HEATMAP
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # 4. PAIRPLOT OF SELECTED FEATURES
    selected = [
        "anxiety_level", "depression", "self_esteem",
        "sleep_quality", "academic_performance", "stress_level"
    ]
    sns.pairplot(df[selected], hue="stress_level", palette="viridis")
    plt.show()

    # 5. BOXPLOTS FOR KEY FEATURES
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(
        ["anxiety_level", "depression", "self_esteem", "sleep_quality"]
    ):
        plt.subplot(1, 4, i + 1)
        sns.boxplot(x="stress_level", y=col, data=df)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # 6. PCA FOR DIMENSIONALITY REDUCTION
    X = df.drop("stress_level", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Scree plot
    pca_full = PCA().fit(X_scaled)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_.cumsum(),
        marker="o"
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.show()

    # 2D PCA scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["stress_level"], palette="Set2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA of Stress Dataset")
    plt.legend(title="Stress Level")
    plt.show()

if __name__ == "__main__":
    main()

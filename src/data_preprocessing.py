# Data preprocessing script
# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # 1. LOAD RAW DATA
    df = pd.read_csv("../data/Dataset.csv")

    # 2. CLEANING
    df.drop_duplicates(inplace=True)
    print("Null values per column:\n", df.isnull().sum())

    # 3. SPLIT FEATURES & TARGET
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]

    # 4. SCALE FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.18, random_state=42
    )

    # 6. SAVE PROCESSED ARRAYS (optional)
    # import joblib
    # joblib.dump((X_train, X_test, y_train, y_test), "../data/processed_data.pkl")

    # For now, just return them
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()

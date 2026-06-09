
import os
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def preprocess_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    os.makedirs("iris_preprocessing", exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train

    test_df = X_test.copy()
    test_df["target"] = y_test

    train_df.to_csv(
        "iris_preprocessing/train.csv",
        index=False
    )

    test_df.to_csv(
        "iris_preprocessing/test.csv",
        index=False
    )

    print("Preprocessing selesai")

if __name__ == "__main__":
    preprocess_data()

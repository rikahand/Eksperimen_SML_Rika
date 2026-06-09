import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bmi" in df.columns and df["bmi"].isna().sum() > 0:
        df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

def encode_categorical(df: pd.DataFrame):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders

def split_and_scale(df: pd.DataFrame, target_col: str = "stroke"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocess_stroke(raw_path: str, output_path: str):
    df = load_data(raw_path)
    df = handle_missing_values(df)
    df, encoders = encode_categorical(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, target_col="stroke")

    df_preprocessed = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    df_preprocessed.to_csv(output_path, index=False)

    return {
        "df_preprocessed": df_preprocessed,
        "X_test": X_test,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler
    }

if __name__ == "__main__":
    raw_path = "stroke_raw/healthcare-dataset-stroke-data.csv"
    output_path = "stroke_preprocessing/stroke_preprocessed.csv"

    os.makedirs("stroke_preprocessing", exist_ok=True)

    result = preprocess_stroke(raw_path, output_path)
    print("Preprocessing selesai.")
    print("Dataset preprocessed disimpan di:", output_path)
    print("Shape hasil preprocessing:", result["df_preprocessed"].shape)

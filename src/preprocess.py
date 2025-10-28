import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess(path="data/heartdisease.csv", seed=66):
    data = pd.read_csv(path)

    # Encode categorical labels
    le = LabelEncoder()
    data["target"] = le.fit_transform(data["sick"])
    data["sex"] = le.fit_transform(data["sex"])
    data = data.drop(["sick"], axis=1)

    y = data["target"]
    X = data.drop(["target"], axis=1)

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=seed
    )

    # Separate feature types
    numerical_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak"
    ]
    categorical_features = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]

    num_pipeline = Pipeline([("minmax", MinMaxScaler())])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", OneHotEncoder(categories="auto"), categorical_features),
    ])

    X_train = full_pipeline.fit_transform(X_train_raw)
    X_test = full_pipeline.transform(X_test_raw)
    feature_names = full_pipeline.get_feature_names_out(list(X.columns))

    return X_train, X_test, y_train, y_test, feature_names

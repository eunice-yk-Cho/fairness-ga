import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess(path, sensitive_attr):
    df = pd.read_csv(path)

    df = df.dropna()

    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    sensitive_index = list(X.columns).index(sensitive_attr)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, sensitive_index
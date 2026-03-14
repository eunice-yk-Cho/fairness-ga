import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

_MALE_TOKENS = {"male", "m", "man"}
_FEMALE_TOKENS = {"female", "f", "woman"}

def _normalize_sensitive(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype(int)

    lower = s.astype(str).str.strip().str.lower()
    if lower.isin(_MALE_TOKENS | _FEMALE_TOKENS).all():
        return lower.isin(_MALE_TOKENS).astype(int)

    if lower.nunique() == 2:
        values = sorted(lower.unique().tolist())
        return (lower == values[1]).astype(int)

    le = LabelEncoder()
    encoded = le.fit_transform(lower)
    return (encoded > encoded.min()).astype(int)

def load_and_preprocess(path, sensitive_attr, target_col, positive_label):
    df = pd.read_csv(path)
    df = df.dropna()
    y = (df[target_col].astype(str).str.strip() == str(positive_label)).astype(int)
    X = df.drop(columns=[target_col]).copy()

    if sensitive_attr not in X.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in {path}")

    X[sensitive_attr] = _normalize_sensitive(X[sensitive_attr])

    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    sensitive_index = list(X.columns).index(sensitive_attr)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=42, stratify=y.values
    )

    return X_train, X_test, y_train, y_test, sensitive_index
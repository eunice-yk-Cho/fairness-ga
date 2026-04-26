
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

_MALE_TOKENS = {"male", "m", "man"}
_FEMALE_TOKENS = {"female", "f", "woman"}

def _normalize_sensitive(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        unique = sorted(s.unique())
        if len(unique) <= 2:
            return (s == unique[-1]).astype(int)
        raise ValueError(
            f"Numeric sensitive attribute has {len(unique)} unique values "
            f"({unique[:5]}{'...' if len(unique) > 5 else ''}). "
            f"Binary or explicit mapping required."
        )

    lower = s.astype(str).str.strip().str.lower()
    if lower.isin(_MALE_TOKENS | _FEMALE_TOKENS).all():
        return lower.isin(_MALE_TOKENS).astype(int)

    if lower.nunique() == 2:
        values = sorted(lower.unique().tolist())
        return (lower == values[1]).astype(int)

    le = LabelEncoder()
    encoded = le.fit_transform(lower)
    min_enc = int(encoded.min())
    mapping_parts = []
    for cls, enc_val in zip(le.classes_, le.transform(le.classes_)):
        mapping_parts.append(f"{cls} \u2192 {0 if enc_val == min_enc else 1}")
    warnings.warn(
        f"Sensitive attribute binarised alphabetically: {', '.join(mapping_parts)}. "
        f"Verify this reflects the intended privileged/unprivileged split.",
        UserWarning,
        stacklevel=3,
    )
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
    categorical_cols = set()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            categorical_cols.add(col)

    col_list = list(X.columns)
    sensitive_index = col_list.index(sensitive_attr)
    categorical_indices = frozenset(
        col_list.index(c) for c in categorical_cols
    ) | {sensitive_index}

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=42, stratify=y.values
    )

    return X_train, X_test, y_train, y_test, sensitive_index, categorical_indices
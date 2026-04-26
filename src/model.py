from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model
def flip_sensitive(x, sensitive_index):
    x_flipped = x.copy()
    x_flipped[sensitive_index] = 1 - x_flipped[sensitive_index]
    return x_flipped

def discrimination_score(model, x, sensitive_index):
    x_prime = flip_sensitive(x, sensitive_index)
    p1 = model.predict_proba([x])[0][1]
    p2 = model.predict_proba([x_prime])[0][1]
    return abs(p1 - p2)
def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    correct = sum(1 for y, p in zip(y_true, y_pred) if y == p)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    return [[tp, fp], [fn, tn]]

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[0][0], cm[0][1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[0][0], cm[1][0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
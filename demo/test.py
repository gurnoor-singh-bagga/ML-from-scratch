# demo/test.py
# run from repo root: python -m demo.test

import random
random.seed(42)

from models.linear_regression import linearRegression
from models.logistic_regression import logisticRegression
from models.base import TrainMethod
from optimizers.schedules import step_decay
from utils.metrics import r2_score, accuracy, precision, recall, f1_score
from maths.functions.loss import mse, binary_cross_entropy

# ── helpers ──────────────────────────────────────────────────────────────────

def make_regression_data(n=200):
    # y = 3x1 + 2x2 + 1 + noise
    X, y = [], []
    for _ in range(n):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        noise = random.gauss(0, 0.5)
        X.append([x1, x2])
        y.append(3*x1 + 2*x2 + 1 + noise)
    return X, y

def make_classification_data(n=200):
    # class 1 if x1 + x2 > 0 else 0
    X, y = [], []
    for _ in range(n):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        X.append([x1, x2])
        y.append(1 if x1 + x2 > 0 else 0)
    return X, y

def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

# ── linear regression tests ───────────────────────────────────────────────────

section("LINEAR REGRESSION")

X, y = make_regression_data(300)
X_train, y_train, X_test, y_test = train_test_split(X, y)

# SGD
model_sgd = linearRegression(method=TrainMethod.SGD, alpha=0.01, epoch=300)
model_sgd.fit(y_train, X_train)
preds = model_sgd.predict(X_test)
print(f"SGD          -> R2: {model_sgd.score(y_test, X_test):.4f}  MSE: {mse(y_test, preds):.4f}")

# BATCH with step decay schedule
model_batch = linearRegression(method=TrainMethod.BATCH, alpha=0.05,
                                batch_size=32, epoch=300,
                                schedule=step_decay(drop=0.5, every=50))
model_batch.fit(y_train, X_train)
preds = model_batch.predict(X_test)
print(f"Batch+decay  -> R2: {model_batch.score(y_test, X_test):.4f}  MSE: {mse(y_test, preds):.4f}")

# CLOSED FORM
model_cf = linearRegression(method=TrainMethod.CLOSED_FORM)
model_cf.fit(y_train, X_train)
preds = model_cf.predict(X_test)
print(f"Closed form  -> R2: {model_cf.score(y_test, X_test):.4f}  MSE: {mse(y_test, preds):.4f}")

# ── logistic regression tests ─────────────────────────────────────────────────

section("LOGISTIC REGRESSION")
X, y = make_classification_data(300)
X_train, y_train, X_test, y_test = train_test_split(X, y)

# SGD
model_sgd = logisticRegression(method=TrainMethod.SGD, alpha=0.1, epoch=300)
model_sgd.fit(y_train, X_train)
preds = model_sgd.predict(X_test)
probas = model_sgd.predict_proba(X_test)
print(f"SGD          -> Acc: {accuracy(y_test, preds):.4f}  "
      f"P: {precision(y_test, preds):.4f}  "
      f"R: {recall(y_test, preds):.4f}  "
      f"F1: {f1_score(y_test, preds):.4f}  "
      f"BCE: {binary_cross_entropy(y_test, probas):.4f}")

# BATCH
model_batch = logisticRegression(method=TrainMethod.BATCH, alpha=0.1,
                                  batch_size=32, epoch=300)
model_batch.fit(y_train, X_train)
preds = model_batch.predict(X_test)
probas = model_batch.predict_proba(X_test)
print(f"Batch        -> Acc: {accuracy(y_test, preds):.4f}  "
      f"P: {precision(y_test, preds):.4f}  "
      f"R: {recall(y_test, preds):.4f}  "
      f"F1: {f1_score(y_test, preds):.4f}  "
      f"BCE: {binary_cross_entropy(y_test, probas):.4f}")

# closed form guard
section("CLOSED FORM GUARD TEST")
try:
    bad = logisticRegression(method=TrainMethod.CLOSED_FORM)
    bad.fit(y_train, X_train)
except ValueError as e:
    print(f"Caught expected error: {e}")

# predict before fit guard
section("PREDICT BEFORE FIT GUARD TEST")
try:
    unfitted = linearRegression()
    unfitted.predict(X_test)
except RuntimeError as e:
    print(f"Caught expected error: {e}")

print("\n✓ All tests completed\n")
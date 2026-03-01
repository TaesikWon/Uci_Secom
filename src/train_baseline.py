# src/train_baseline.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score

RANDOM_STATE = 42

def main():
    df = pd.read_csv("data/uci-secom.csv")
    print("Loaded:", df.shape)

    # Target: Pass/Fail (fail often = -1)
    y_raw = df["Pass/Fail"]
    X = df.drop(columns=["Pass/Fail"])

    # Time 컬럼이 있으면 제거 (문자열/시간이라 baseline에 방해)
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])

    # numeric만 사용
    X = X.apply(pd.to_numeric, errors="coerce")

    # 실패(fail) = -1을 1로(positive), 나머지(pass)=0
    y = (y_raw == -1).astype(int).to_numpy()

    print("Class counts (0=pass, 1=fail):", np.bincount(y))

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    f1s, pras = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        proba = pipe.predict_proba(X_va)[:, 1]

        f1 = f1_score(y_va, pred)
        pr = average_precision_score(y_va, proba)

        f1s.append(f1); pras.append(pr)
        print(f"[Fold {fold}] F1={f1:.4f}  PR-AUC={pr:.4f}")

    print("\n=== CV Summary ===")
    print(f"F1 mean={np.mean(f1s):.4f} std={np.std(f1s):.4f}")
    print(f"PR-AUC mean={np.mean(pras):.4f} std={np.std(pras):.4f}")

if __name__ == "__main__":
    main()
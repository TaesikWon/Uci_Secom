# src/compare_models.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

RANDOM_STATE = 42


def row_missing_stats(X):
    """row 단위 결측 통계 2개: (결측개수, 결측비율) — numpy 입력 기준"""
    miss_cnt = np.isnan(X).sum(axis=1).reshape(-1, 1).astype(np.float32)
    miss_ratio = (miss_cnt / X.shape[1]).astype(np.float32)
    return np.hstack([miss_cnt, miss_ratio])


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray, n_grid: int = 201):
    """검증셋에서 threshold 스캔하여 F1 최대값과 그 임계값 반환"""
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_f1, best_thr


def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "uci-secom.csv"

    df = pd.read_csv(data_path)
    print("Loaded:", df.shape)

    y_raw = df["Pass/Fail"]
    X = df.drop(columns=["Pass/Fail"])

    if "Time" in X.columns:
        X = X.drop(columns=["Time"])

    # numeric 변환 (문자/이상치 -> NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # ✅ 소수 클래스를 불량(positive=1)로 자동 지정
    vc = y_raw.value_counts(dropna=False)
    fail_value = vc.idxmin()
    y = (y_raw == fail_value).astype(int).to_numpy()

    print("Pass/Fail raw counts:\n", vc)
    print(f"Using fail_value={fail_value} as positive(1)")
    print("Class counts (0=pass, 1=fail):", np.bincount(y))

    # ✅ numpy로 통일 (feature name 경고 원천 차단)
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    return X_np, y


def build_preprocess(scale_numeric: bool):
    """
    결측=신호 전처리:
    - 값: 중앙값 임퓨트 (+ 로지스틱이면 스케일링)
    - 결측 플래그: 모든 컬럼 isna
    - row 결측 통계: 2개
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    miss_flag = MissingIndicator(features="all", sparse=False)
    row_stats = FunctionTransformer(row_missing_stats, validate=False)

    preprocess = FeatureUnion(transformer_list=[
        ("num", num_pipe),
        ("miss", miss_flag),
        ("row", row_stats),
    ])
    return preprocess


def build_models(y):
    models = {}

    # Logistic (수치 스케일링 포함)
    models["Logistic"] = Pipeline(steps=[
        ("preprocess", build_preprocess(scale_numeric=True)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])

    # RandomForest (스케일링 불필요)
    models["RandomForest"] = Pipeline(steps=[
        ("preprocess", build_preprocess(scale_numeric=False)),
        ("clf", RandomForestClassifier(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
        )),
    ])

    # LightGBM (있으면)
    try:
        from lightgbm import LGBMClassifier

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        # ✅ 경고/로그 억제
        warnings.filterwarnings("ignore")  # 필요 시 전체 경고 억제(깔끔하게)

        models["LightGBM"] = Pipeline(steps=[
            ("preprocess", build_preprocess(scale_numeric=False)),
            ("clf", LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,          # ✅ LightGBM split warning 포함 로그 억제
            )),
        ])
    except ImportError:
        models["LightGBM"] = None

    return models


def evaluate_models(X, y, models):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))  # ✅ 동일 split 공유

    results = {name: {"f1_05": [], "best_f1": [], "best_thr": [], "prauc": []}
               for name in models.keys() if models[name] is not None}

    for fold, (tr, va) in enumerate(splits, start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        print(f"\n=== Fold {fold} ===")
        for name, pipe in models.items():
            if pipe is None:
                continue

            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)[:, 1]

            # F1@0.5
            pred_05 = (proba >= 0.5).astype(int)
            f1_05 = f1_score(y_va, pred_05, zero_division=0)

            # Best F1 (검증셋에서 임계값 스캔)
            best_f1, best_thr = best_f1_threshold(y_va, proba)

            # PR-AUC
            pr = average_precision_score(y_va, proba)

            results[name]["f1_05"].append(f1_05)
            results[name]["best_f1"].append(best_f1)
            results[name]["best_thr"].append(best_thr)
            results[name]["prauc"].append(pr)

            print(
                f"{name:12s} | "
                f"PR-AUC={pr:.4f}  "
                f"F1@0.5={f1_05:.4f}  "
                f"BestF1={best_f1:.4f} (thr={best_thr:.2f})"
            )

    print("\n\n==============================")
    print("=== CV Summary (5-fold) ===")
    print("==============================")
    for name in results.keys():
        f1_05 = np.array(results[name]["f1_05"])
        bf1 = np.array(results[name]["best_f1"])
        bth = np.array(results[name]["best_thr"])
        pr = np.array(results[name]["prauc"])

        print(
            f"{name:12s} | "
            f"PR-AUC mean={pr.mean():.4f} std={pr.std():.4f}  ||  "
            f"F1@0.5 mean={f1_05.mean():.4f}  ||  "
            f"BestF1 mean={bf1.mean():.4f} (avg thr={bth.mean():.2f})"
        )


def main():
    X, y = load_data()
    models = build_models(y)

    if models.get("LightGBM") is None:
        print("\n❌ LightGBM이 설치되어 있지 않습니다.")
        print("   설치: pip install lightgbm\n")

    evaluate_models(X, y, models)


if __name__ == "__main__":
    main()
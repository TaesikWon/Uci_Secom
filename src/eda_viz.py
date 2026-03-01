# src/eda_viz.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "uci-secom.csv"
OUT_DIR = ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 라벨: 1=Fail(불량), -1=Pass(정상)
y = (df["Pass/Fail"] == 1).astype(int)

X = df.drop(columns=["Pass/Fail"])
if "Time" in X.columns:
    X = X.drop(columns=["Time"])
X = X.apply(pd.to_numeric, errors="coerce")

# 1) 클래스 분포 바차트
plt.figure()
y.value_counts().sort_index().plot(kind="bar")
plt.title("Class distribution (0=Pass, 1=Fail)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=200)
plt.close()

# 2) 결측률 상위 30개 바차트
miss = X.isna().mean().sort_values(ascending=False).head(30)
plt.figure(figsize=(12, 5))
miss.plot(kind="bar")
plt.title("Top 30 missing-rate features")
plt.xlabel("Feature")
plt.ylabel("Missing rate")
plt.tight_layout()
plt.savefig(OUT_DIR / "missing_top30.png", dpi=200)
plt.close()

# 3) 결측률 분포 히스토그램
plt.figure()
X.isna().mean().hist(bins=30)
plt.title("Histogram of feature missing rates")
plt.xlabel("Missing rate")
plt.ylabel("Number of features")
plt.tight_layout()
plt.savefig(OUT_DIR / "missing_rate_hist.png", dpi=200)
plt.close()

print("Saved plots to:", OUT_DIR)
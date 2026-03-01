# src/eda_quick.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "uci-secom.csv"

df = pd.read_csv(DATA_PATH)

# 라벨 확정: 1 = Fail(불량), -1 = Pass(정상)
df["y"] = (df["Pass/Fail"] == 1).astype(int)

print("data_path:", DATA_PATH)
print("shape:", df.shape)
print("\nClass counts (0=pass, 1=fail):")
print(df["y"].value_counts())

X = df.drop(columns=["Pass/Fail", "y"])
if "Time" in X.columns:
    X = X.drop(columns=["Time"])

miss = X.isna().mean().sort_values(ascending=False)
print("\nMissing rate top10:")
print(miss.head(10))

miss_by_class = pd.DataFrame({
    "miss_rate_all": X.isna().mean(),
    "miss_rate_fail": X[df["y"] == 1].isna().mean(),
    "miss_rate_pass": X[df["y"] == 0].isna().mean(),
})
miss_by_class["diff_fail_minus_pass"] = miss_by_class["miss_rate_fail"] - miss_by_class["miss_rate_pass"]

print("\nMissing rate difference top10 (fail - pass):")
print(miss_by_class["diff_fail_minus_pass"].sort_values(ascending=False).head(10))
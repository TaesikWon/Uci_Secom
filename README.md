# 🏭 UCI SECOM – 반도체 제조 불량(Fail) 예측 & 검사량(Workload) 절감 시나리오

반도체 제조 공정 센서 데이터(SECOM)를 활용해 **불량(Fail)을 조기에 탐지**하고,  
현장에서 **검사 우선순위를 최적화하여 검사량(Workload)을 줄이는 운영 시나리오**를 도출하는 프로젝트입니다.

---

## 📋 목차

1. [프로젝트 목표](#1-프로젝트-목표)
2. [데이터 출처](#2-데이터-출처)
3. [데이터셋 요약](#3-데이터셋-요약)
4. [핵심 인사이트](#4-핵심-인사이트)
5. [프로젝트 구조](#5-프로젝트-구조)
6. [실행 준비](#6-실행-준비)
7. [데이터 배치 방법](#7-데이터-배치-방법)
8. [모델 비교 실행](#8-모델-비교-실행-logistic-vs-randomforest-vs-lightgbm-)
9. [현장 적용 – 검사량 절감 시나리오](#9-현장-적용--검사량workload-절감-시나리오-)
10. [EDA 실행](#10-eda-실행-)
11. [TODO](#11-todo-)

---

## 1. 프로젝트 목표

| # | 목표 |
|---|------|
| 1 | 센서 기반 불량(FAIL) 예측 모델 구축 |
| 2 | 불균형 데이터 환경에서 **PR-AUC 중심 평가** |
| 3 | "모델 성능 숫자"뿐 아니라, **현장에서 어떻게 쓰면 검사량을 얼마나 줄일 수 있는지**를 운영 정책으로 제시 |

---

## 2. 데이터 출처

| 구분 | 링크 |
|------|------|
| Kaggle Dataset (사용 데이터) | [paresh2047/uci-semcom](https://www.kaggle.com/datasets/paresh2047/uci-semcom) |
| Original Source (UCI ML Repository) | [SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) |

> ⚠️ 데이터 파일은 레포에 포함하지 않습니다 (`data/` 는 `.gitignore` 처리).  
> Kaggle / 원본 데이터 사용 시 각 플랫폼의 이용 규정을 따라주세요.

---

## 3. 데이터셋 요약

| 항목 | 내용 |
|------|------|
| 샘플 수 | **1,567** 개 |
| 컬럼 수 | **592** (센서 피처 + 타겟 포함) |
| 타겟 컬럼 | `Pass/Fail` |
| 정상 (Pass) | `-1` — 다수 클래스 |
| 불량 (Fail) | ` 1` — 소수 클래스 (≈ **6.6%**) |

---

## 4. 핵심 인사이트

### 4-1. 클래스 불균형

불량 비율이 매우 낮아 **Accuracy는 의미가 약합니다.**  
따라서 **PR-AUC** 로 "불량을 상위로 끌어올리는 랭킹 능력"을 평가 기준으로 사용합니다.

### 4-2. 결측치가 "신호"일 수 있음

SECOM은 결측이 매우 많으며, 일부 센서는 Fail 쪽 결측 패턴이 다를 수 있습니다.  
따라서 결측을 단순히 채우는 것이 아니라,

- **결측 플래그(`isna`)를 피처로 추가** → "결측 패턴" 자체를 학습하도록 구성

---

## 5. 프로젝트 구조

```
Uci-Secom/
├── data/                   # 원본 데이터 (커밋 제외 – .gitignore 처리)
├── outputs/                # 시각화 / 산출물 (커밋 제외)
├── src/
│   ├── eda_quick.py        # 빠른 EDA
│   ├── eda_viz.py          # 시각화 EDA
│   └── compare_models.py   # Logistic / RF / LGBM 비교
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. 실행 준비

### 6-1. 가상환경 / 패키지 설치

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 6-2. LightGBM 설치 (선택)

```bash
pip install lightgbm
```

---

## 7. 데이터 배치 방법

`compare_models.py` 는 기본적으로 아래 경로의 파일을 읽습니다.

```
data/uci-secom.csv
```

Kaggle에서 다운로드한 CSV를 아래처럼 배치하세요.

```
Uci-Secom/data/uci-secom.csv
```

> 파일명이 다를 경우  
> (1) 파일명을 `uci-secom.csv` 로 변경하거나  
> (2) `src/compare_models.py` 의 데이터 경로를 직접 수정하세요.

---

## 8. 모델 비교 실행 (Logistic vs RandomForest vs LightGBM) 🧪

프로젝트 루트에서 실행:

```bash
python src/compare_models.py
```

### 출력 지표 설명

| 지표 | 설명 | 중요도 |
|------|------|--------|
| **PR-AUC** | 불량을 상위로 랭킹하는 능력 | ⭐ 핵심 지표 |
| F1@0.5 | 임계값 0.5 기준 F1 – 불균형 시 0에 가깝게 나오는 것이 정상 | 참고 |
| BestF1 (thr=...) | 검증셋에서 threshold 스캔으로 얻은 최대 F1 | 참고 |

> 💡 불량 비율이 낮아 확률이 0.5를 넘기 어려우면 `F1@0.5 ≈ 0` 이 정상일 수 있습니다.  
> **PR-AUC** 와 "운영 임계값 기반 시나리오"를 함께 보는 것이 중요합니다.

---

## 9. 현장 적용 – 검사량(Workload) 절감 시나리오 🏭📉

### 정책 A — 목표 Recall 기반

> **"불량을 95% 잡으려면(Recall = 0.95), 전체 중 몇 %만 검사하면 되나요?"**

모델 불량 확률 **내림차순** 으로 검사했을 때,  
불량의 95%가 포함될 때까지의 검사 비율 = **Workload%**

```
Recall 0.95  →  Workload 30%
→ "불량 95%를 잡기 위해 전체의 30%만 검사"
→ 전수검사 대비 검사량 70% 절감 가능
```

### 정책 B — Top-K 검사 정책

> **"하루에 상위 10%만 검사 가능할 때, 불량을 몇 % 잡나요?"**

| Top-K | Precision | Recall | 비고 |
|-------|-----------|--------|------|
| 상위 1% | — | — | 모델 실행 후 채울 값 |
| 상위 2% | — | — | |
| 상위 5% | — | — | |
| 상위 10% | — | — | 하루 검사 가능 물량 예시 |
| 상위 20% | — | — | |

> ✅ **권장 다음 단계**: `compare_models.py` 에 Recall 목표 기반 Workload% 및 Top-K 정책표를 자동 산출하는 리포트를 추가하면 "현장 적용" 메시지가 더 강해집니다.

---

## 10. EDA 실행 📊

```bash
python src/eda_quick.py
python src/eda_viz.py
```

---

## 11. TODO 🧭

- [ ] 모델별 Recall(90 / 95 / 98%) 달성 시 **Workload% / Precision / Threshold** 표 자동 출력
- [ ] Top-K(1 / 2 / 5 / 10 / 20%) 검사 시 **Recall / Precision** 표 자동 출력
- [ ] SHAP 기반 중요 센서 해석 및 **"결측 = 신호"** 검증 강화
- [ ] 운영 시나리오 리포트 (검사량 절감 근거) 문서화
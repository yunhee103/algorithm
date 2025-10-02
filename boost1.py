# 앙상블(Ensemble) 기법 중 부스팅(Boosting)
# 여러 개의 약한 학습기를 순차적으로 학습시켜, 이전 모델이 틀린 부분을 보완하면서 점점 더 강력한 모델을 만드는 방법
# 가중치를 활용하여 약분류기를 강분류기로 만드는 방법

# brest_cancer dataset으로 분류 모델
# pip install xgboost, lightgbm - 설치

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from lightgbm import LGBMClassifier # xgboost보다 성능이 우수하나 데이터 양이 적으면 과적합 발생
import lightgbm as lgb


# 1. 데이터 불러오기

data = load_breast_cancer()   # sklearn 내장 유방암 데이터셋 로드
x = pd.DataFrame(data.data, columns=data.feature_names)  # 특징 데이터 (569행, 30열)
y = data.target               # 레이블 (0=악성, 1=양성)
print(x.shape)                # 데이터 크기 출력 → (569, 30)


# 2. 학습/테스트 데이터 분할

x_train, x_test, y_train, y_test = train_test_split(
    x, y,                         # 전체 데이터
    test_size=0.2,                 # 20%는 테스트, 80%는 학습
    random_state=12,               # 랜덤 시드 고정 (재현 가능성 보장)
    stratify=y                     # 클래스 비율 유지 (편향 방지)
)


# 3. 모델 정의

# XGBoost 모델
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',        # 학습기: 결정트리 기반 (gbtree), 선형모델 (gblinear)도 가능
    max_depth=6,             # 트리의 최대 깊이 (깊을수록 복잡, 과적합 위험 ↑)
    n_estimators=500,        # 트리 개수 (약한 학습기 개수)
    eval_metric='logloss',   # 평가 지표 (로그 손실 함수)
    random_state=42          # 랜덤 시드 고정
)

# LightGBM 모델
lgb_clf = LGBMClassifier(
    n_estimators=500,        # 트리 개수
    random_state=42,         # 랜덤 시드 고정
    verbose=2                # 학습 과정 출력 레벨
)


# 4. 모델 학습

xgb_clf.fit(x_train, y_train)   # XGBoost 학습
lgb_clf.fit(x_train, y_train)   # LightGBM 학습


# 5. 예측

pred_xgb = xgb_clf.predict(x_test)   # XGBoost 예측
pred_lgb = lgb_clf.predict(x_test)   # LightGBM 예측


# 6. 평가

print(f'XGBoost acc : {accuracy_score(y_test, pred_xgb):.4f}')   # XGBoost 정확도 출력
print(f'LightGBM acc : {accuracy_score(y_test, pred_lgb):.4f}')  # LightGBM 정확도 출력
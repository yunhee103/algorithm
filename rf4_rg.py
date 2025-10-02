# RandomForestRegressor : 정량적 예측 모델
# california_housing datasets 사용

import pandas as pd # 데이터프레임 다루는 라이브러리
import numpy as np # 수치 연산 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리 (여기서는 사용하지 않았지만 일반적으로 필요)
import seaborn as sns # 데이터 시각화 라이브러리 (matplotlib보다 더 예쁜 그래프)
from sklearn.ensemble import RandomForestClassifier # 분류(classification)를 위한 랜덤 포레스트 모델
from sklearn.model_selection import train_test_split, cross_val_score # 데이터 분할 및 교차 검증 함수
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error # 모델 성능 평가 지표
from sklearn.linear_model import LogisticRegression # 분류를 위한 로지스틱 회귀 모델
from sklearn.ensemble import RandomForestRegressor # 회귀(regression)를 위한 랜덤 포레스트 모델 (주로 사용)
from sklearn.tree import DecisionTreeClassifier # 분류를 위한 의사결정 나무 모델
from sklearn.datasets import fetch_california_housing # 캘리포니아 주택 가격 데이터셋 가져오는 함수

# 캘리포니아 주택 가격 데이터셋 로드

housing = fetch_california_housing(as_frame = True) 
# as_frame = True로 설정하면 데이터를 pandas DataFrame 형식으로 가져옴.


print(housing.data[:2]) # 데이터(독립변수, Feature)의 처음 2행 출력
print(housing.target[:2]) # 타겟(종속변수, Label)의 처음 2개 출력
print(housing.feature_names) # 독립변수(컬럼)들의 이름 출력
df = housing.frame # 전체 데이터를 하나의 DataFrame으로 만듦, as_frame = True 때문에 가능
print(df.head(3)) # 전체 데이터프레임의 처음 3행 출력

# feature / label 분리
x = df.drop('MedHouseVal', axis = 1) # 'MedHouseVal' 컬럼을 제외하고 독립변수(x)로 설정, axis = 1은 열(column)을 기준으로 삭제하라는 의미.
y = df['MedHouseVal']    # 'MedHouseVal' 컬럼을 종속변수(y)로 설정

# train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# RandomForestRegressor 모델 생성 및 학습
rfmodel = RandomForestRegressor(n_estimators = 200, random_state = 42, n_jobs = -1)
# n_estimators: 사용할 의사결정나무(Decision Tree)의 개수. 많을수록 성능이 좋아지지만 계산 시간이 늘어남.
# random_state: 모델 학습 시 랜덤성을 고정하여 재현성을 확보.
# n_jobs: 병렬 처리에 사용할 CPU 코어 수. -1은 모든 코어를 사용하라는 의미로 학습 속도를 높여줌.

rfmodel.fit(x_train, y_train)    # 학습 데이터(x_train, y_train)를 사용해 모델 학습
y_pred = rfmodel.predict(x_test)    # 학습된 모델로 테스트 데이터(x_test)에 대한 예측값 생성
print(f'MSE : {mean_squared_error(y_test, y_pred):.3f}')     # MSE(평균 제곱 오차) 계산 및 출력 MSE는 예측값과 실제값 차이의 제곱을 평균한 값. 0에 가까울수록 좋은 모델.
print(f'R^2 : {r2_score(y_test, y_pred):.3f}')      # R-squared(결정계수) 계산 및 출력,  분산을 얼마나 잘 설명하는지 나타냄. 1에 가까울수록 좋은 모델.

print('독립변수 중요도 순위 표')
importance = rfmodel.feature_importances_   # 각 독립변수의 중요도를 추출
indices = np.argsort(importance)[::-1]       # 중요도를 내림차순으로 정렬
ranking = pd.DataFrame({
    'Feature' : x.columns[indices], # 중요도 순서에 맞게 독립변수 이름 정렬
    'Importance' : importance[indices]  # 중요도 값 정렬
})
print(ranking)       # 독립변수 중요도 순위 표 출력
# 'MedInc'(중간 소득)가 가장 중요한 변수로 나타남.

### 하이퍼파라미터 튜닝 설명
# 간단한 튜닝으로 최적의 파라미터 찾기
# GridSearchCV : 정확하게 최적값 찾기에 적당. 파라미터가 많으면 계산량 증가
from sklearn.model_selection import RandomizedSearchCV  # 하이퍼파라미터 랜덤 탐색 라이브러리
# 연속적 값 처리 가능, 최적 조합 못 찾을 수 있음

param_list = {
    'n_estimators' : [200, 400, 600],                       # 의사결정나무 개수
    'max_depth' : [None, 10, 20, 30],                       # 트리 최대 깊이 . None은 제한 없음.
    'min_samples_leaf' : [1, 2, 4],                         # 리프(leaf) 노드에 포함되어야 하는 최소 샘플 수. 과적합 방지.
    'min_samples_split' : [2, 5, 10],                        # 노드를 분할하기 위한 최소 샘플 수. 과적합 방지.
    'max_features' : [None, 'sqrt', 'log2', 1.0, 0.8, 0.6]  # 최대 특성수, log2(features)
}

# RandomizedSearchCV 객체 생성
search = RandomizedSearchCV(
    RandomForestRegressor(random_state = 42), # 튜닝할 기본 모델
    param_distributions = param_list,         # 튜닝할 파라미터 목록
    n_iter = 20,                              # 랜덤으로 20번 조합을 선정해서 평가
    scoring = 'r2',                           # 모델 평가 기준으로 R^2 사용
    cv = 3,                                   # 3겹으로 교차 검증. 데이터를 3등분하여 훈련/검증 반복.
    random_state = 42
)
search.fit(x_train, y_train) # 튜닝 탐색 시작 및 학습

print('best params : ', search.best_params_)    # 20번의 시도 중 가장 좋은 성능을 낸 파라미터 조합 출력
best_model = search.best_estimator_             # 최적의 파라미터로 학습된 모델을 가져옴
print('best cv r^2(교차검증 평균 결정계수) : ', search.best_score_)
print('best model 결정계수 : ', r2_score(y_test, best_model.predict(x_test)))

"""
데이터 준비 → 모델 학습 → 성능 평가 → 모델 튜닝
앙상블 학습 > 배깅 > 랜덤 포레스트

1. 랜덤 포레스트(Random Forest) 이론 : 배깅 기법을 사용하는 구체적인 모델. 특히, 여러 개의 독립적인 의사결정나무(Decision Tree) 만들 때 배깅 기법을 활용

앙상블 학습(Ensemble Learning): 여러 개의 모델(여기서는 의사결정나무)을 만들고, 그 예측 결과를 종합하여 최종 예측값을 결정하는 방식. 
여러 모델의 단점을 보완하여 단일 모델보다 뛰어난 성능을 얻을 수 있음.
* 이것은 가장 큰 개념이자 전체 집합. 하나의 모델만 사용하는 것이 아니라, 여러 개의 모델을 만들어 그 결과들을 종합하여 더 정확한 예측을 하는 모든 기법을 통칭.

배깅(Bagging): 랜덤 포레스트는 '배깅'이라는 앙상블 기법을 사용. 
전체 훈련 데이터에서 일부를 무작위로 추출(복원 추출)하여 여러 개의 작은 데이터셋을 만듦. 
각 데이터셋으로 독립적인 의사결정나무를 만들고, 이 트리들의 예측값을 평균하여 최종 예측값을 산출.
* 앙상블 학습의 여러 가지 방법 중 하나입니다. 훈련 데이터셋에서 복원 추출을 통해 여러 개의 작은 데이터셋을 만들고, 이 각각의 데이터셋으로 여러 모델을 독립적으로 학습시키는 방식. 
이 모델들의 예측 결과를 평균(회귀)하거나 투표(분류)하여 최종 결과를 냄.

과적합(Overfitting) 방지: 랜덤 포레스트는 각 트리를 만들 때마다 무작위로 일부 특성(feature)만 선택하여 트리를 성장시킴. 
이 두 가지 랜덤성(데이터 샘플링, 특성 선택) 덕분에 과적합을 효과적으로 줄일 수 있음.


부스팅(Boosting) : 부스팅은 여러 개의 모델(약한 학습기, Weak Learner)을 순차적으로 학습시키면서 이전 모델의 오류를 보완해 나가는 방식입니다.
배깅이 여러 모델을 독립적으로 만들고 결과를 합친다면, 부스팅은 모델을 하나씩 만들 때마다 이전 모델이 틀렸던 부분(오류)을 더 잘 맞추도록 가중치를 부여하고 다음 모델을 학습시킴. 
즉, 마치 '잘못 배운 부분을 고쳐가며 더 똑똑해지는' 과정과 같음.



                                 배깅 vs 부스팅 비교
특징	       배깅 (Bagging)	                           부스팅 (Boosting)
학습 방식   	여러 모델을 독립적으로 병렬 학습        	    여러 모델을 순차적으로 학습
모델 관계	    각 모델이 서로에게 영향 X	                   각 모델이 이전 모델의 오류를 보완하며 학습
목표	       분산(Variance)감소	                        편향(Bias) 감소
대표 모델	    랜덤 포레스트	                                AdaBoost, Gradient Boosting, XGBoost, LightGBM



랜덤 포레스트 (배깅):
모델의 **분산(variance)**을 줄이는 데 효과적. (과적합 방지)
하이퍼파라미터 튜닝에 상대적으로 덜 민감하고, 병렬 처리가 가능해 학습 속도가 빠름.
데이터가 많고 과적합이 우려될 때 좋은 선택.

부스팅:
모델의 편향(bias)을 줄이는 데 효과적. (성능 향상)
일반적으로 랜덤 포레스트보다 예측 성능이 더 좋음.
순차적으로 학습하기 때문에 학습 시간이 더 오래 걸릴 수 있고, 과적합에 더 취약할 수 있어 튜닝이 중요.


2. 하이퍼파라미터 튜닝 이론
하이퍼파라미터: 모델이 학습하기 전에 사용자가 직접 설정하는 값. 예를 들어, n_estimators, max_depth 등이 하이퍼파라미터에 해당. 이 값들이 모델의 성능에 큰 영향을 미침.
RandomizedSearchCV: 모든 파라미터 조합을 탐색하는 GridSearchCV와 달리, 무작위로 특정 개수의 조합만 선택하여 탐색. 파라미터가 많을 때 GridSearchCV보다 훨씬 효율적.

교차 검증(Cross-Validation): 데이터를 여러 겹으로 나누어 번갈아 가며 학습과 검증에 사용함으로써, 특정 데이터셋에만 모델이 잘 작동하는 것을 방지하고 일반화된 성능을 평가하는 방법. 

3. 성능 평가 지표
MSE (Mean Squared Error): 예측값과 실제값의 차이를 제곱하여 평균을 낸 값. 오차에 제곱을 하기 때문에 오차가 클수록 패널티를 크게 줌.
R² (R-squared, 결정계수): 모델이 데이터의 분산을 얼마나 잘 설명하는지 나타냄. 0에서 1 사이의 값을 가지며, 1에 가까울수록 좋은 모델.

"""
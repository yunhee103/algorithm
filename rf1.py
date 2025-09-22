# # RandomForest 분류 / 예측 알고리즘
# 분류 알고리즘으로 titanic dataset 사용해 이진 분류
# Bagging 사용 : 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고 결과를 집계(Aggregating) 하는  방법 
# 참고 : 우수한 성능을 원한다면 Boosting , 오버피팅이 걱정된다면 Bagging의 방식 추천

# titanic dataset : feature (pclass, age, sex), label (survived)


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
print(df.info)          # 데이터프레임의 행, 열, 각 열의 데이터 타입, 결측치(Null) 정보를 요약해서 보여줌.
print(df.isnull().any())     # 각 열에 결측치가 있는지 True/False로 확인하는 용도임.

# 데이터 분석 전에 **데이터 정제(Data Cleaning)**를 하는 과정임.
# Pclass, Age, Sex 열에 있는 결측치가 포함된 행을 모두 삭제함.
df = df.dropna(subset=['Pclass', 'Age', 'Sex']) # 값이 null이면 삭제
print(df.shape) # 데이터 정제 후 데이터프레임의 행과 열의 크기를 확인하는 용도임.


# feature, label로 분리

# feature(특성)와 label(정답)을 나누는 과정임.
# feature: 모델을 학습시키는 데 사용되는 독립 변수임.
# label: 예측하고 싶은 종속 변수임.
df_x = df[['Pclass', 'Age', 'Sex']].copy()   # Pclass, Age, Sex 열을 feature로 사용함.


print(df_x.head(2))  #sex를 더미화 


# 머신러닝 모델은 문자열 데이터를 바로 처리할 수 없어서 숫자로 변환해야 함.
# 'LabelEncoder'는 문자열(Sex: male, female)을 숫자(0, 1)로 변환해주는 함수임.
from sklearn.preprocessing import LabelEncoder
encorder = LabelEncoder()
df_x.loc[:,'Sex'] = encorder.fit_transform(df_x['Sex'])
print(df_x.head(2))     # 인코딩이 잘 되었는지 확인하는 용도임.

df_y = df['Survived']   # 종속변수
print(df_y.head(2))
print()

# 데이터를 **훈련(training)용**과 **테스트(test)용**으로 나눔.
# 'train_test_split()' 함수는 데이터를 무작위로 섞어서 비율에 맞게 나눠주는 역할을 함.
# test_size=0.3: 전체 데이터의 30%를 테스트용으로 사용함.
# random_state=12: 코드를 여러 번 실행해도 동일한 결과를 얻기 위해 무작위성을 고정하는 용도임.
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)
# RandomForestClassifier 모델 객체를 생성하는 과정임.
# criterion='entropy': 모델이 데이터를 나눌 때 '엔트로피(entropy)'라는 기준을 사용하겠다는 의미임. 엔트로피는 데이터의 불확실성을 측정하는 지표임.
# n_estimators=500: 500개의 결정 트리(Decision Tree)를 생성해서 앙상블(Ensemble)한다는 의미임.
model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값 :' ,pred[:10])
print('실제값 :' ,np.array(test_y[:10]))
# 성능 비교 
print('맞춘 갯수 :' , sum(test_y == pred))
print('전체 대비 맞춘 비율 :' , sum(test_y == pred) / len(test_y))
print('분류 정확도 :' , accuracy_score(test_y, pred))

# k-fold 교차 검증

# 모델의 성능을 좀 더 객관적으로 평가하는 방법임.
# 데이터를 여러 개의 'fold'(여기서는 5개)로 나누고, 한 번은 테스트용으로 쓰고 나머지는 훈련용으로 쓰는 과정을 반복함.
# 이렇게 하면 특정 데이터에만 잘 맞는 **과적합(Overfitting)** 문제를 방지하고 모델의 일반적인 성능을 확인할 수 있음.
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('교차 검증 평균 정확도:', np.round(np.mean(cross_vali),5))
print()
# 중요 변수 확인
# 'feature_importances_': 모델이 어떤 특성(feature)을 중요하게 생각했는지 나타내는 값임.
# 이 값을 통해 모델이 예측에 가장 크게 기여한 변수가 무엇인지 알 수 있음.
print('특성 (변수) 중요도 :' , model.feature_importances_)
import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('feature_importancesscore')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

"""
추가로 알아두면 좋은 내용
1. 데이터 전처리(Data Preprocessing)의 중요성
이번 코드에서는 dropna()를 이용해 결측치를 삭제했지만, 실제 데이터 분석에서는 다양한 방법이 사용됨. 
결측치를 채우거나(예: 평균, 중앙값), 데이터를 표준화(Standardization)하거나, 정규화(Normalization)하는 등의 과정을 통해 모델의 성능을 높일 수 있다. 
데이터 전처리가 잘 되어야 모델이 정확한 예측을 할 수 있다.

2. 모델 파라미터 튜닝
RandomForestClassifier() 안에 있는 criterion='entropy'나 n_estimators=500 같은 값들을 하이퍼파라미터(Hyperparameter)라고 함. 
이 값들을 어떻게 설정하느냐에 따라 모델의 성능이 크게 달라질 수 있다.
n_estimators: 트리의 개수. 이 값이 너무 크면 학습 시간이 오래 걸리고, 너무 작으면 성능이 낮아질 수 있다.
max_depth: 트리의 최대 깊이. 이 값을 조절해서 모델이 과적합되는 것을 막을 수 있다.
Grid Search나 Random Search와 같은 기법을 사용하면 다양한 하이퍼파라미터 조합을 자동으로 테스트해서 최적의 값을 찾아낼 수 있다. 

3. 다른 평가 지표
이번 코드에서는 accuracy_score (정확도)를 사용. 하지만 정확도만으로 모델의 성능을 완벽하게 평가하기는 어렵다. 
예를 들어, 타이타닉 데이터에서 생존자가 10%, 사망자가 90%라면, 단순히 '모두 사망했다'고 예측해도 정확도는 90%가 나옵니다.
이런 불균형한 데이터에서는 정밀도(Precision), 재현율(Recall), F1-Score, 그리고 혼동 행렬(Confusion Matrix) 같은 다른 평가 지표를 함께 보는 것이 중요.
이 지표들은 모델이 어떤 예측을 잘못했는지(예: 생존자를 사망자로 예측했는지) 더 자세히 보여줌.

"""
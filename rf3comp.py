# titanic dataset : LogisticRegression, DecisionTreeClassifier 비교

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
# 분석에 불필요한 열을 삭제함. 'inplace=True'는 원본 데이터프레임에 바로 반영하라는 의미임.
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
print(df.head(2), df.shape)
print(df.describe())
print(df.isnull().sum())
#  Null 처리: 평균 또는 'N' 으로 변경 
# **데이터 전처리** 과정 중 하나로, 결측치를 적절한 값으로 채우는 작업임.
df['Age'].fillna(df['Age'].mean(), inplace=True) # 'Age' 열의 결측치를 평균값으로 채움.
df['Cabin'].fillna('N', inplace=True) # 'Cabin' 열의 결측치를 문자 'N'으로 채움.
df['Embarked'].fillna('N', inplace=True) # 'Embarked' 열의 결측치를 문자 'N'으로 채움.
print(df.isnull().sum()) # 결측치가 모두 채워졌는지 다시 확인하는 용도임.
print(df.info()) # 각 열의 데이터 타입과 결측치 정보를 다시 확인하는 용도임.


# object type
# 문자형 변수들의 데이터 분포를 확인하는 용도임. 
print('Sex :', df['Sex'].value_counts())
print('Cabin :', df['Cabin'].value_counts())
print('Embarked :', df['Embarked'].value_counts())
# 방 호수가 너무 방대하여 앞글자만 따기위한 가공
# 'Cabin' 열의 값(예: 'C23 C25 C27')에서 첫 글자만 추출함.
df['Cabin'] = df['Cabin'].str[:1]
print(df.head(3))
print()
# 성별이 생존 확률에 어떤 영향을 미쳤는지 확인하기 (0 사망 1 생존)
# **탐색적 데이터 분석(EDA)** 과정으로, 데이터를 시각화하기 전에 미리 경향을 파악하는 용도임.
print(df.groupby(['Sex', 'Survived'])['Survived'].count())

print('여성 생존율 :', 233/(81+233))
print('남성 생존율 :', 109/(468+109))

# 시각화
# 'barplot'은 막대 그래프를 생성하는 함수임.
# 성별('Sex')에 따른 생존율('Survived')을 시각적으로 보여줌.
sns.barplot(x = 'Sex', y='Survived', data=df, ci = 95)
plt.show()
# 성별 기준으로 Pclass별 생존 확율
# 'hue'는 추가적인 범주를 색상으로 구분해서 보여주는 역할임.
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
plt.show()

# 나이별 기준으로 생존 확률
# '나이'를 '아기', '10대', '성인' 등으로 범주화하는 함수임.
def getAgeFunc(age):
    msg = ''
    if age <= -1: msg = 'unknown'
    elif age <= 5: msg = 'baby'
    elif age <= 18: msg = 'teenager'
    elif age <= 65: msg = 'adult'
    else: msg='elder'
    return msg 

# 'Age' 열에 있는 각 값에 'getAgeFunc' 함수를 적용해서 새로운 열('Age_category')을 만듦.
df['Age_category'] = df['Age'].apply(lambda a:getAgeFunc(a))
print(df.head(2))
# 나이대, 성별에 따른 생존율을 시각화함.
# 'order'는 막대 그래프의 순서를 지정하는 용도임.
sns.barplot(x='Age_category', y='Survived', hue='Sex', data=df, order=['unknown', 'baby', 'teenager', 'adult', 'elder'])
plt.show()
# 시각화를 위해 임시로 만들었던 열('Age_category')을 삭제함
del df['Age_category']

# 문자열 자료를 숫자화
# 머신러닝 모델은 문자열을 처리할 수 없어서 숫자로 변환해야 함.
# 'LabelEncoder'는 각 고유한 문자열 값을 고유한 숫자로 매핑해주는 함수임

from sklearn import preprocessing
def labelIncoder(datas):
    cols = ['Cabin', 'Sex', 'Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique())    # [7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique())      # [1 0]
print(df['Embarked'].unique()) # [3 0 2 1]

print()
# feature / label
# 모델 학습을 위해 특성(feature)과 정답(label)을 분리하는 과정임.
feature_df = df.drop(['Survived'], axis='columns')
label_df = df['Survived']
print(feature_df.head(2))
print(label_df.head(2))

x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

logmodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
demodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logpred = logmodel.predict(x_test)
print('logmodel acc:{0:.5f}'.format(accuracy_score(y_test, logpred)))
depred = demodel.predict(x_test)
print('demodel acc:{0:.5f}'.format(accuracy_score(y_test, depred)))
rfpred = rfmodel.predict(x_test)
print('rfmodel acc:{0:.5f}'.format(accuracy_score(y_test, rfpred)))

"""
추가로 알아두면 좋은 내용
1. 모델 비교와 선택
로지스틱 회귀(Logistic Regression), 결정 트리(Decision Tree), 랜덤 포레스트(Random Forest)는 각각 다른 특성을 가진 모델.
로지스틱 회귀: 통계 기반의 선형 모델로, 예측 결과에 대한 설명력이 높습니다. '이유를 알고 싶을 때' 사용하기 좋습니다.
결정 트리: 나무 구조로 데이터를 분할하며, 직관적이고 이해하기 쉽습니다. 하지만 데이터에 따라 과적합(Overfitting)이 쉽게 발생할 수 있습니다.
랜덤 포레스트: 여러 개의 결정 트리를 만들어 결과를 종합하는 앙상블(Ensemble) 모델입니다. 결정 트리의 과적합 문제를 보완하여 일반적으로 좋은 성능을 보여줍니다.
이 코드에서는 정확도(Accuracy)만으로 모델을 비교했지만, 실제로는 모델의 특성을 고려해서 목적에 맞는 모델을 선택해야 합니다. 
예를 들어, 예측의 정확성보다 '모델이 왜 이렇게 예측했는지'를 아는 것이 더 중요하다면 로지스틱 회귀를 선택할 수 있습니다.

2. 하이퍼파라미터 튜닝
이번 코드에서는 모델을 기본 설정으로 사용했습니다. 하지만 LogisticRegression()이나 DecisionTreeClassifier() 안에 있는 하이퍼파라미터를 조절하면 모델의 성능을 더욱 향상시킬 수 있습니다. 
예를 들어, DecisionTreeClassifier에서 max_depth (트리의 최대 깊이)나 min_samples_leaf (리프 노드가 되기 위한 최소 샘플 수) 같은 파라미터를 조절하면 과적합을 방지하고 성능을 올릴 수 있습니다.
Grid Search나 Random Search와 같은 기법을 사용해서 최적의 파라미터 조합을 찾는 방법을 학습하면, 모델의 잠재력을 최대한 끌어낼 수 있습니다.

3. 데이터 불균형 문제와 평가 지표
만약 데이터가 매우 불균형하다면(예: 생존자가 극히 적은 경우), 정확도만으로는 모델 성능을 제대로 평가하기 어렵습니다. 
이럴 때는 정밀도(Precision), 재현율(Recall), F1-Score, ROC-AUC 등의 다른 평가 지표를 함께 사용해야 합니다.
정밀도: 모델이 '생존'이라고 예측한 것 중 실제로 생존한 비율.
재현율: 실제로 생존한 사람 중 모델이 '생존'이라고 맞춘 비율.
이러한 지표들은 모델의 예측이 얼마나 신뢰할 수 있는지, 중요한 샘플을 얼마나 잘 찾아냈는지 등을 더 자세히 알려줍니다.
"""
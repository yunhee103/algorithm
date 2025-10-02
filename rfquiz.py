import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# [문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
# dataset은 winequality-red.csv 
# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

"""
독립 변수: 와인 품질 예측에 사용되는 11가지 화학적 특성(quality 열을 제외한 모든 열).
종속 변수: 와인 품질 등급을 나타내는 quality_binary 열
"""

df1 = pd.read_csv('./winequality-red.csv')
print(df1.head(2), df1.shape)
print(df1.describe())

# 데이터프레임의 'alcohol' 열에 있는 값들이 각각 몇 개씩 있는지 세어서 오름차순으로 정렬하여 출력하는 코드
print('alcohol :', df1['alcohol'].value_counts().sort_index())

# 품질 점수를 이진 분류(0: 나쁨, 1: 좋음)로 변환
df1['quality_binary'] = df1['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 독립 변수(feature)와 종속 변수(label) 분리
feature_df = df1.drop(columns=['quality', 'quality_binary'])
label_df = df1['quality_binary']

# 데이터를 학습/테스트 세트로 분리
x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 로지스틱 회귀 모델
logmodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
logpred = logmodel.predict(x_test)
print('logmodel acc:{0:.5f}'.format(accuracy_score(y_test, logpred)))

# 결정 트리 모델
demodel = DecisionTreeClassifier(random_state=42).fit(x_train, y_train)
depred = demodel.predict(x_test)
print('demodel acc:{0:.5f}'.format(accuracy_score(y_test, depred)))

# 랜덤 포레스트 모델
rfmodel = RandomForestClassifier(random_state=42).fit(x_train, y_train)
rfpred = rfmodel.predict(x_test)
print('rfmodel acc:{0:.5f}'.format(accuracy_score(y_test, rfpred)))

# >> 랜덤 포레스트 모델이 와인의 화학적 특성을 바탕으로 품질을 예측하는 데 가장 높은 정확도를 보임

importances = rfmodel.feature_importances_
# 중요도 순으로 정렬
sorted_indices = np.argsort(importances)[::-1]

# 각 특성의 중요도 출력
for index in sorted_indices:
    print(f"{feature_df.columns[index]}: {importances[index]:.5f}")

# 시각화
# 품질을 제외한 모든 화학적 특성 열을 리스트로 만듬.
# 'quality'는 종속 변수이므로 시각화에서 x축으로 사용.
feature_cols = df1.columns.drop('quality')

# 그래프의 전체 크기와 서브플롯(subplot)의 개수를 설정.
# 4행 3열의 격자 모양으로 총 12개의 그래프를 만듬.
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()  # 2차원 배열을 1차원으로 변환하여 쉽게 접근.

# 반복문을 사용해 각 화학적 특성에 대한 그래프를 그립니다.
for i, col in enumerate(feature_cols):
    sns.barplot(x='quality', y=col, data=df1, ci=None, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Quality vs {col}') # 각 그래프의 제목을 설정.
    axes[i].set_xlabel('Quality') # x축 이름을 설정.
    axes[i].set_ylabel(col) # y축 이름을 설정.

# 남는 빈 공간을 삭제.
for j in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[j])

# 그래프들이 겹치지 않게 레이아웃을 조정.
plt.tight_layout()

# 그래프를 화면에 표시.
plt.show()



# 중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료/ 종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석

"""
ID (Identifier): 개체 식별자를 의미하며, 데이터셋에서 각 개인을 고유하게.
STA (Status): 상태를 의미하며, 환자의 현재 상태(예: 생존, 사망)를 나타냅니다.
AGE (Age): 나이를 의미합니다.
SEX (Sex): 성별을 의미합니다.
RACE (Race): 인종을 의미합니다.
SER (Serum): 혈청을 의미하며, 혈액 검사 결과를 나타내는 데 사용될 수 있습니다.
CAN (Cancer): 암을 의미합니다.
CRN (Clinical Record Number): 임상 기록 번호를 의미하며, 환자의 의.
INF (Infection): 감염을 의미합니다.
CPR (Cardiopulmonary Resuscitation): 심폐소생술을 의미합니다.
HRA (Heart Rate Anomaly): 심박수 이상을 의미합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/patient.csv')
# print(df.head())
print(df.isnull().sum()) # 결측치 확인

feature_df = df.drop('STA', axis=1)
label_df = df['STA']
x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('예측값 : ', pred)
print('실제값 : ', np.array(y_test))
print('정확도 : ', accuracy)

# 중요 변수 알아보기
# 참고 : 중요 변수 알아보기
print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
def plot_feature_importances(model, x):   # 특성 중요도 시각화
    n_features = x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(model, x_train)

print('중요도 상위 3개 변수 :')
for i in model.feature_importances_.argsort()[-3:][::-1]:
    print(' - {} : {:.4f}'.format(x_train.columns[i], model.feature_importances_[i]))
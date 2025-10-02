# SVM (Support Vector Machine) : 비확률적 이진 선형 분모델 작성 가능
# 직선적(선형) 분류 뿐 아니라 커널 트릭을 이용해 비선형 분류도 가능
# 커널(kernels) 선형분류가 어려운 저차원 자료를 고차원 공간으로  매핑해서 분류

# XOR(배타적 논리합) 문제를 해결하는 예제

from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn import svm, metrics # SVM(서포트 벡터 머신)과 성능 평가 지표(metrics)
import pandas as pd # 데이터프레임
import numpy as np # 수치 연산

# 데이터 준비
x_data = [
    [0,0,0], # 입력 [0,0] -> 출력 0
    [0,1,1], # 입력 [0,1] -> 출력 1
    [1,0,1], # 입력 [1,0] -> 출력 1
    [1,1,0]  # 입력 [1,1] -> 출력 0
]
x_df = pd.DataFrame(x_data)
# 독립변수(feature)와 종속변수(label) 분리
feature = np.array(x_df.iloc[:, 0:2])   # 첫 두 열(0과 1)을 독립변수로 사용. [0,0], [0,1], [1,0], [1,1]
label = np.array(x_df.iloc[:,2])        # 마지막 열(2)을 종속변수로 사용. 0, 1, 1, 0
print(feature)
print(label)

# model = LogisticRegression()  # 선형 모델이라 XOR 문제를 풀지 못함.
model = svm.SVC()               # SVC(Support Vector Classifier) 모델 생성. 커널 트릭을 이용해 비선형 문제도 해결 가능.
model.fit(feature,label)        # 독립변수(feature)와 종속변수(label)로 모델 학습
# 예측 및 성능 평가
pred = model.predict(feature)   # 학습 데이터로 예측 수행
print('예측값 :', pred)
print('실제값 :', label)
print('정확도 :', metrics.accuracy_score(label,pred))   # 예측값과 실제값 비교해 정확도 계산.


"""
이번 예제는 머신러닝 모델의 **선형(Linear)**과 비선형(Non-linear) 분류 능력을 비교코드에 대한 추가 설명
SVM의 핵심: 커널 트릭
LogisticRegression은 직선(선형)으로만 데이터를 분류가능. 
그래서 [0,0]과 [1,1]을 한 그룹으로, [0,1]과 [1,0]을 다른 그룹으로 나누는 XOR 문제는 풀지 못함.

반면 SVM은 커널 트릭(Kernel Trick)을 사용해 저차원의 데이터를 고차원의 공간으로 매핑해서 비선형 문제를 해결할 수 있음.
즉, 원래는 직선으로 나눌 수 없던 데이터를 고차원에서 보면 평면으로 나눌 수 있게 됨.

svm.SVC()는 기본적으로 RBF(Radial Basis Function) 커널을 사용하는데, 이 커널 덕분에 이 코드에서는 100%의 정확도가 나옴.

LogisticRegression과의 차이

만약 model = LogisticRegression()을 사용하고 코드를 실행하면, 정확도가 50%나 75%처럼 낮게 나올 것. 
이는 로지스틱 회귀가 XOR 문제의 비선형 관계를 제대로 학습하지 못하기 때문.

svm.SVC()를 사용하면, 내부적으로 커널 트릭을 활용하기 때문에 완벽하게 문제를 해결하는 것을 확인할 수 있음.

"""
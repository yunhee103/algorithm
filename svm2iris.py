from sklearn import datasets # 사이킷런 내장 데이터셋
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
import numpy as np # 넘파이(수치 연산)
import pandas as pd # 판다스(데이터 분석)
from sklearn.metrics import accuracy_score # 정확도 평가
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler # 데이터 표준화
import matplotlib.pyplot as plt # 데이터 시각화
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
import pickle  # 모델 저장/불러오기


# 붓꽃(iris) 데이터셋 로드
iris = datasets.load_iris()

# print(iris['data'])  # 붓꽃 데이터의 독립 변수(특성) 데이터를 출력

# 붓꽃 데이터의 세 번째 특성(petal length)과 네 번째 특성(petal width) 간의 상관계수를 계산하고 출력.
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))  # 출력 결과: 0.96286, 매우 강한 양의 상관관계


# 데이터 분리
x = iris.data[:, [2, 3]] # 독립변수(feature): 꽃잎 길이와 꽃잎 너비 / x는 행렬(matrix) 형태
y = iris.target # 종속변수(label): 붓꽃 품종 (0, 1, 2) / y는 벡터(vector) 형태


print('x: ', x[:3])
print('y: ', y[:3], set(y))  # 종속변수 데이터 3개와 고유값 출력. {0, 1, 2}

# 데이터를 학습용(train)과 테스트용(test)으로 7:3 비율로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 분할된 데이터들의 크기(shape)를 출력. (행, 열) 순서로 표시됩니다.
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

# 모델 생성 및 학습

# 로지스틱 회귀 분류 모델을 생성.
# C: 규제(regularization) 강도를 조절하는 파라미터. 값이 작을수록 규제가 강해져 과적합을 방지하는 효과.
# random_state=0: 모델 내부의 무작위성을 제어하여 실행할 때마다 동일한 결과를 얻기 위해 설정.
# model = LogisticRegression(C = 0.1, random_state=0, verbose=0)

from sklearn import svm, metrics
model = svm.LinearSVC()  # LinearSVC는 선형 SVM 모델로, 데이터의 결정 경계를 직선으로 나눔.

print('여기: ', model)  # 생성된 모델의 정보를 출력.
# 학습 데이터로 모델 훈련
model.fit(x_train, y_train)

# 훈련된 모델로 테스트 데이터 예측
y_pred = model.predict(x_test)
print('예측값: ', y_pred)  
print('실제값: ', y_test)  

# 총 예측 갯수와 틀린 갯수 계산
print('총 갯수:%d, 오류수: %d'%(len(y_test), (y_test != y_pred).sum()))
print('-'*100)  

# 모델의 분류 정확도를 확인하는 세 가지 방법

print('분류정확도 확인1: ')     # 0.97778
# accuracy_score 함수를 사용하여 실제값(y_test)과 예측값(y_pred)을 비교하여 정확도를 계산.
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류정확도 확인2: ')
# pandas의 crosstab 함수를 이용해 혼동 행렬(confusion matrix)을 만듦.
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)  # 생성된 혼동 행렬을 출력.
# 혼동 행렬의 대각선 값(정확히 맞춘 경우)을 더해 전체 데이터 수로 나눠 정확도 계산
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류정확도 확인3: ')
# 모델 객체에 내장된  model.score 메서드를 사용하여 테스트 데이터에 대한 정확도를 바로 계산.
print('test: ', model.score(x_test, y_test))
print('train: ', model.score(x_train, y_train))
# ->> 차이가 크면 과적합(overfitting)을 의심


# 모델 저장 및 불러오기
# pickle.dump를 사용하여 'model' 객체를 'logimodel.sav'라는 파일에 바이너리 쓰기('wb') 모드로 저장.
pickle.dump(model, open('logimodel.sav', 'wb'))

# 현재 메모리에 있는 model 변수를 삭제.
del model


# 'logimodel.sav' 파일을 바이너리 읽기('rb') 모드로 열어 'read_model'이라는 변수에 로드.
read_model = pickle.load(open('logimodel.sav', 'rb'))

# 새로운 데이터로 예측을 수행하기 위해 기존 테스트 데이터의 일부를 참고용으로 출력.
print(x_test[:3])

# 예측해볼 새로운 데이터를 numpy 배열 형태로 생성. (꽃잎 길이, 꽃잎 너비)
new_data = np.array([[5.1, 1.1], [1.1, 1.1], [6.1, 7.1]])

# 참고: 만약 모델을 학습시킬 때 데이터 표준화를 했다면, 새로운 데이터에도 동일한 표준화 작업을 적용.
# sc.fit(new_data); new_data = sc.transform(new_data)

# 불러온 모델(read_model)을 사용하여 새로운 데이터(new_data)의 품종을 예측.
new_pred = read_model.predict(new_data)
print('예측 결과: ', new_pred)  # 예측된 클래스(0, 1, 2)를 출력.

# predict_proba 메서드를 사용하여 새로운 데이터가 각 클래스(0, 1, 2)에 속할 확률을 출력.
# 내부적으로 softmax 함수를 거친 결과값.
# print(read_model.predict_proba(new_data))

# 시각화: 결정 경계 그리기
# plot_decisionFunc 함수는 모델의 결정 경계(Decision Boundary)를 시각화하는 역할
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # 그래프 좌표 범위 설정
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                        np.arange(x2_min, x2_max, resulution)) 
    
    # xx, yy를 1차원배열로 만든 후 전치. 이어 분류기로 클래스 예측값 Z얻기(격자 좌표를 1차원으로 펼친 후 모델로 예측)
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 예측값에 따라 배경색을 채워 결정 경계 시각화
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    # 실제 데이터 포인트 산점도 그리기
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    # 테스트 데이터 포인트를 굵은 테두리로 표시
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
# 결정 경계 시각화 함수 호출
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                test_idx = range(100, 150), title='scikit-learn 제공')

"""
 SVM 하이퍼파라미터 튜닝
LinearSVC와 SVC 모델 모두 성능에 영향을 미치는 중요한 하이퍼파라미터.
C (규제 파라미터): 코드에도 있었던 이 파라미터는 모델의 오류 허용 정도를 조절. 
C 값이 작을수록 규제가 강해져서 일반화(Generalization) 성능이 좋아지고, 클수록 모델이 훈련 데이터에 더 가깝게 맞춰져 과적합(Overfitting)될 가능성이 있음.
kernel (커널 종류): SVC 모델에만 해당하며, linear, rbf, poly 등 다양한 종류가 있음. 어떤 커널을 쓰느냐에 따라 모델의 성능이 크게 달라질 수 있음.

"""

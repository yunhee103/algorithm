from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from matplotlib.colors import ListedColormap
import pickle
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
# print(iris['data'])
print(np.corrcoef(iris.data[:,2],iris.data[:,3]))  #0.96286543
# np.corrcoef() 함수는 두 변수(배열) 간의 피어슨 상관 계수(Pearson correlation coefficient)를 계산.
# iris.data[:,2]: iris 데이터의 세 번째 열(인덱스 2), 즉 '꽃잎 길이(petal length)' 데이터를 선택.
# iris.data[:,3]: iris 데이터의 네 번째 열(인덱스 3), 즉 '꽃잎 너비(petal width)' 데이터를 선택.
#  '꽃잎 길이'와 '꽃잎 너비' 간의 상관 계수를 계산하여 행렬 형태로 출력.
x = iris.data[:, [2,3]]   # petal length, petal width만 참여 matrix
# 독립 변수(feature)를 선택합니다.
# iris.data는 붓꽃의 4가지 특징을 담고 있는 2차원 배열.
# [:, [2,3]]는 모든 행(:)에 대해 2번 인덱스('꽃잎 길이')와 3번 인덱스('꽃잎 너비')에 해당하는 열([2,3])만 선택.
# 이렇게 선택된 데이터는 2개의 열을 가진 2차원 배열(matrix).
y = iris.target #vector
print(x[:3])  # 출력 결과는 꽃잎 길이와 꽃잎 너비 쌍으로 구성된 배열입
print(y[:3], set(y))
# y[:3]: 종속 변수 y의 첫 3개 값을 출력합니다.
# set(y): y 배열의 모든 고유값(unique value)을 집합(set) 형태로 출력합니다.
# 이를 통해 붓꽃의 종류가 0, 1, 2 세 가지.

# train / test split (7:3)

# LogisticRegression 다중 클래스 지원 - > 파이썬 클래스가 아닌 종속변수를 이야기하는 것! = label = y

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (105, 2) (45, 2) (105,) (45,)
# (105, 2): 105개 행, 2개 열의 훈련용 독립 변수
# (45, 2): 45개 행, 2개 열의 테스트용 독립 변수
# (105,): 105개 값의 훈련용 종속 변수
# (45,): 45개 값의 테스트용 종속 변수

"""
# --------------------------------------
# Scaling(데이터 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우/언더플로우 방지 효과) 
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)  # 훈련 데이터(x_train)를 이용해 표준화에 필요한 평균과 표준편차를 계산
x_train = sc.transform(x_train)  # 계산된 평균과 표준편차를 이용해 훈련 데이터를 표준화
x_test = sc.transform(x_test)  #독립변수만 스케일링, 종속변수 x!
print(x_train[:3]) #표준화  값들이 0 근처로 변환된 것을 확인할 수 있
# 스케일링 원복
# 모델 학습이 끝난 후 예측 결과를 원래 스케일로 되돌리고 싶을 때 사용
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3]) 
# --------------------------------------
"""


# 분류모델 생성
# C 속성 : L2규제  -모델에 패널티 적용 (tuning parameter 중 하나) -> 릿지 엘라스틱사용  
#       : 모델에 패널티를 적용하는 L2 규제(regularization)의 역수(inverse).
# C 값이 작을수록 더 강한 규제(패널티)를 적용합니다.
# 강한 규제는 모델의 복잡성을 낮춰 과적합(overfitting)을 방지하는 효과.
# C=0.1은 기본값(C=1.0)보다 강한 규제를 적용하여 모델을 단순화하려는 의도.

# 숫자값을 조정해 가며 분류 정확도를 확인 1.0 (기본) 10.0 100.0.... 값이 작을수록 더 강한 정규화 규제를 가함
# model = LogisticRegression(C=0.1, random_state=0, verbose=0)  # solver='1bfgs' 기본  , 'lbfgs'는 다항 분류 시 내부적으로 소프트맥스(softmax) 함수를 사용 
# verbose=0: 모델 학습 과정을 출력하지 않도록 설정


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1)


print(model)
model.fit(x_train,y_train) #suervised learning 

# 분류예측 - 모델 성능 파악용
y_pred = model.predict(x_test)
print('예측 값 : ', y_pred)
print('실제 값 :' , y_test)
print('총 갯수 : %d,  오류수:%d'%(len(y_test), (y_test != y_pred).sum()))  # (y_test != y_pred).sum(): 실제값(y_test)과 예측값(y_pred)이 다른 경우(오류)의 개수를 합산
print()
print('분류정확도 확인1 : ')
print('%.3f'%accuracy_score(y_test, y_pred)) # sklearn.metrics.accuracy_score를 사용해 정확도를 계산하고 소수점 셋째 자리까지 출력

con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print('con_mat')
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

# 정확도 = (올바르게 예측한 개수) / (전체 테스트 데이터 개수) 
# con_mat[i][j]는 실제 클래스가 i, 예측 클래스가 j인 데이터의 개수

print('분류정확도 확인3 :')
# model.score() 메서드는 정확도를 직접 반환하는 편리한 함수
print('test : ', model.score(x_test, y_test))
print('train : ', model.score(x_train, y_train))  
#두개의 값 차이가 크면 과적합 의심

# 모델 저장
pickle.dump(model, open('logimodel.sav', 'wb'))
del model

read_model = pickle.load(open('logimodel.sav', 'rb'))

# 새로운 값으로 예측 : petal.length, petal.width 만 참여
print(x_test[:3])
new_data = np.array([[5.1, 1.1], [1.1, 1.1], [6.1,7.1]])
# 참고 : 만약 표준화한 데이터로 모델을 생성했다면
# sc.fit(new_data;) new_data = sc.transform(new_data) 
# sc.fit(new_data)가 잘못된 이유: sc.fit()은 데이터의 평균과 표준편차를 새로 계산
# 하지만 모델은 이미 x_train의 평균과 표준편차로 학습 됨. 따라서 새로운 데이터의 평균과 표준편차를 다시 계산하면 모델이 기대하는 데이터 형태와 달라져 오류가 발생.
#scaled_new_data = sc.transform(new_data) # <--- 이 코드가 꼭 필요


new_pred = read_model.predict(new_data)  # 내부적으로 softmax가 출력한 값을 argmax로 처리
# read_model.predict(new_data)
# 모델이 새로운 데이터(new_data)에 대한 최종 예측 클래스를 반환 -> 내부적으로 모델은 먼저 소프트맥스(softmax) 함수를 사용해 각 클래스에 속할 확률을 계산.
# -> argmax 함수를 사용해 계산된 확률 중 가장 높은 확률을 가진 클래스를 선택.
# -> 이 코드는 최종적으로 가장 가능성이 높은 클래스(0, 1, 또는 2)를 반환(predict() 최종적인 분류 결과를 알려 줌. 예를 들어, [0 1 2]와 같이 가장 높은 확률을 가진 클래스의 인덱스를 출력)

print('예측 결과 : ' , new_pred)
print(read_model.predict_proba(new_data))  # softmax가 출력한 한 원본 확률 값을 반환
# 데이터 샘플 수(행) x 클래스 개수(열) 
# 각 행은 하나의 데이터 샘플에 대한 예측이며, 각 열은 해당 클래스(0, 1, 2)에 속할 확률을 의미합니다.
# # 예를 들어, [[0.9, 0.05, 0.05], ...]와 같은 형태로 출력됩니다.

# predict_proba()는 예측 결과의 확률 분포를 보임
# 이 값을 통해 모델이 얼마나 확신을 가지고 예측했는지 파악 가능. 
# predict_proba()가 반환하는 2차원 배열에서 각 행을 분석하여 가장 큰 값을 찾고 이를 100%로 환산하면 모델의 확신도를 알 수 있음


# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스  그래프에 테스트 데이터를 표시할 때 사용
    # resulution : 등고선 오차 간격 / 격자(grid)를 만들 때의 간격. 이 값이 작을수록 더 부드러운 결정 경계가 그려짐
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')  # 색상 팔레트 정의
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    # X 데이터의 최소/최대 값에 1을 더하고 빼서 그래프의 x, y 좌표 범위를 지정
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 훈련 데이터(X)의 각 클래스(y)를 순회하며 클래스별로 마커와 색상을 다르게 하여 점을 그림
    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    # test_idx가 제공되면, 테스트 데이터를 동그라미 테두리로 둘러싸서 표시
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
# 훈련 데이터와 테스트 데이터의 독립 변수(feature)를 수직(vstack)으로 결합
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')

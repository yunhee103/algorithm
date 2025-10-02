from sklearn import datasets # 데이터셋 로드
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델 (주석 처리됨)
import numpy as np # 수치 연산
import pandas as pd # 데이터프레임
from sklearn.metrics import accuracy_score # 정확도 평가 함수
from sklearn.model_selection import train_test_split # 학습/테스트 데이터 분리
from sklearn.preprocessing import StandardScaler # 데이터 표준화 (주석 처리됨)
import matplotlib.pyplot as plt # 데이터 시각화
plt.rc('font', family = 'Malgun Gothic') # 한글 깨짐 방지 설정
from matplotlib.colors import ListedColormap # 색상 팔레트, 다중 클래스 (종속변수, label, y, class)
import pickle# 모델 저장 및 불러오기
from lightgbm import LGBMClassifier #  LightGBM 분류 모델 xgboost 보다 성능 좋아 과적합 우려
# pip install lightgbm

iris = datasets.load_iris()
# 상관계수 확인: 꽃잎 길이(petal.length)와 꽃잎 너비(petal.width)의 상관관계
print(np.corrcoef(iris.data[:,2], iris.data[:,3])) # 0.96286543
# feature(독립변수)와 label(종속변수) 분리
x = iris.data[:,[2,3]] # 꽃잎 길이, 꽃잎 너비만 사용 matrix   
y = iris.target # vetcor, 붓꽃 품종 (0: Setosa, 1: Versicolor, 2: Virginica)
print(x[:3])
print(y[:3], set(y))# 0, 1, 2 세 가지 클래스 확인

# train / test split (7:3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3 ,random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)


# 원본으로 해보고 값이 낮으면 밑에 과정을 해볼 수 있음
"""
# Scaling (데이터 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우/언더플로우 방지 효과가 있다.)늘그런건 아님
print(x_train[:3])
print()
sc = StandardScaler() # 표준화 
sc.fit(x_train); sc.fit(x_test) # -> 학습 데이터 기준으로 스케일링 준비
x_train = sc.transform(x_train)      # -> 준비된 기준으로 학습 데이터 변환
x_test = sc.transform(x_test) # 독립변수만 스케일링-> 동일한 기준으로 테스트 데이터 변환
print(x_train[:3])
print()
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
"""

# 분류 모델 생성
# C 속성 : L2규제 - 모델에 패널티 적용. 숫자값을 조정해 가며 분류 정확도를 확인 1.0 10.0 100.0 ... 값이 작을수록 더 강한 정규화를 규제함 
# model = LogisticRegression(C=0.1 , random_state = 0, verbose = 0)

from sklearn.ensemble import RandomForestClassifier 
model = LGBMClassifier(criterion='entropy',n_estimators=500, random_state=1)    # LightGBMClassifier 모델을 사용. 부스팅 계열 모델로 성능이 우수.
model.fit(x_train, y_train) # supervise learning 지도 학습

# 분류 예측 - 모델 성능 파악용
y_pred = model.predict(x_test)
print("예측값 : ", y_pred)
print("실제값 : ", y_test)

print('총 갯수:%d, 오류수:%d'%(len(y_test),(y_test != y_pred).sum()))
print()
print('분류정확도 확인1 : ')
print('%.3f'%accuracy_score(y_test, y_pred))
print('분류정확도 확인2 : ')
print()
# 혼동 행렬(Confusion Matrix) 생성: 모델이 무엇을 맞추고 틀렸는지 표로 보여줌
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)
# con_mat[0][0]: 실제 0을 0으로 예측한 개수
# con_mat[1][1]: 실제 1을 1으로 예측한 개수
# con_mat[2][2]: 실제 2를 2으로 예측한 개수
# (con_mat[0][0]+con_mat[1][1]+con_mat[2][2]) / len(y_test) 와 같이 전체 정답 수를 나눠야 정확한 정확도가 나옴.
print((con_mat[0][0]+con_mat[1][1]) / len(y_test))

print('분류정확도 확인3 : ')
print('test: ',model.score(x_test, y_test))  # 테스트 데이터에 대한 모델의 정확도
print('train: ',model.score(x_train, y_train)) # 학습 데이터에 대한 모델의 정확도
# 두 값의 차이가 크면 과적합(Overfitting)을 의심할 수 있음.
# 과적합: 학습 데이터에는 잘 맞지만, 새로운 데이터에는 성능이 떨어지는 현상.

# 모델 저장
pickle.dump(model, open('logimodel.sav', 'wb')) # 모델 객체를 logimodel.sav 파일에 저장
del model    # 메모리에서 모델 삭제 (저장 확인용)

read_model = pickle.load(open('logimodel.sav', 'rb'))    # 저장된 모델 파일 불러오기

# 새로운 값으로 예측
print(x_test[:3])
new_data = np.array([[5.1,1.1],[1.1,1.1],[6.1,7.1]])
# 참고 -> 만약 표준화한 데이터로 모델을 생성했다면 
# sc.fit(new_data) 하여 표준화 해서 new_data = sc.transform(new_data)해야함
new_pred = read_model.predict(new_data)
print('예측 결과: ', new_pred)
print(read_model.predict_proba(new_data))  # 각 클래스에 대한 예측 확률 출력 (소프트맥스 결과)

# 시각화

    # 분류기의 결정 경계(decision boundary)를 시각화하는 데 사용
    # 결정 경계: 서로 다른 클래스를 구분하는 경계선
    # 등고선 그래프(contourf)를 이용해 결정 경계 영역을 색으로 구분
    # 산점도(scatter)를 이용해 실제 데이터 포인트를 표시

def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
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

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
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
x_combined_std = np.vstack((x_train, x_test))   # feature   독립변수 합치기
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label    종속변수 합치기
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')
# test_idx = range(105, 150)는 합쳐진 데이터에서 테스트 데이터의 인덱스 범위 지정.

"""
데이터 스케일링: 주석 처리된 StandardScaler 부분은 머신러닝에서 매우 중요한 전처리 단계. 
특히 로지스틱 회귀나 SVM처럼 거리에 기반한 모델은 스케일링을 하면 성능이 크게 향상. 
하지만 LightGBM과 같은 트리 기반 모델은 스케일링에 덜 민감하지만, 다른 모델을 사용할 때는 반드시 고려해야 함.

과적합(Overfitting)과 과소적합(Underfitting): train과 test의 점수 차이를 통해 모델의 과적합 여부를 의심
과적합: train 점수는 매우 높고 test 점수는 낮은 경우. 모델이 학습 데이터에만 너무 맞춰진 상태.
과소적합: train과 test 점수 모두 낮은 경우. 모델이 너무 단순해서 데이터의 패턴을 제대로 학습하지 못한 상태.


"""
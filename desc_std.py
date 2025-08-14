# 표준편차, 분산을 중요
# 두 반의 시험 성적이 " 평균이 같다고 해서 성적분포가 동일한가?" -> NO
# 표준 편차와 분산을 알아야 함

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

np.random.seed(42)

# 목표 평균....
target_mean = 60
std_dev_small = 10
std_dev_large = 20

class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size = 100)  #100개 생성
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size = 100) 
# 평균 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
class2_adj = class2_raw - np.mean(class2_raw) + target_mean
# 정수화 및 범위 제한
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)
print(class1)
print(class2)

# 통계값 계산
mean1, mean2 = np.mean(class1), np.mean(class2)
std1, std2 = np.std(class1), np.std(class2)
var1, var2 = np.var(class1), np.var(class2)
print("1반 (성적)")
print(class1)
print(f"평균:{mean1:.2f}, 표준 편차:{std1:.2f}, 분산:{var1:.2f}")
print("2반 (성적)")
print(class2)
print(f"평균:{mean2:.2f}, 표준 편차:{std2:.2f}, 분산:{var2:.2f}")


df = pd.DataFrame({
    'Class' : ['1반'] * 100 + ['2반'] * 100,
    'Score' : np.concatenate([class1, class2])
})

print(df.head(3))
print(df.tail(3))
df.to_csv('desc_std1_1.csv', index=False , encoding='utf-8') #파일로 저장
# 시각화 : 산포도 / 박스플롯  2가지가 있음

# 시각화 : 산포도
x1 = np.random.normal(1, 0.05, size=100)
x2 = np.random.normal(2, 0.05, size=100)

plt.figure(figsize=(10, 6))
plt.scatter(x1, class1, label=f'1반(평균={mean1:.2f}, σ={std1:.2f}')
plt.scatter(x2, class2, label=f'2반(평균={mean2:.2f}, σ={std2:.2f}')
plt.hlines(target_mean, 0.5, 2.5, colors='red', linestyles='dashed', label=f'공통평균={target_mean}')
plt.title('동일 평균, 다른 성적 분포를 가진 두반 비교')
plt.xticks([1,2], ['1반', '2반'])
plt.ylabel('시험점수')
plt.legend();
plt.grid(True)
plt.tight_layout()
plt.show()

# 시각화 : 박스 플롯
plt.figure(figsize=(8,5))
plt.boxplot([class1, class2], label=['1반', '2반'])
plt.title('성적 분포를 가진 두반 비교')
plt.ylabel('시험점수')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()
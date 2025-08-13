# 기술 통계의 목적은 데이터를 수지, 요약, 정리, 시각화
# 도수분포표(Frequency Distribution Table)는 데이터를 구간별로 나눠 빈도를 정리한 표
# 이를 통해 데이터의 분포를 한 눈에 파악할 수 있다...

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

#step 1 : 데이터를 읽어 dataFrame에 저장
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')

print(df.head(2))

#step 2 : 최대값 , 최소값
min_height = df['키'].min()
max_height = df['키'].max()
print(f'최소값 : {min_height}, 최대값 :{max_height}') 

#step 3 : 계급 설정 (cut)
bins = np.arange(156, 195, 5)
print(bins)
df['계급'] = pd.cut(df['키'], bins=bins, right=True, include_lowest=True)  # 오른쪽 포함 (]/ 최솟값 포함
print(df.head(3))
print(df.tail(3))

#step 4 : 각 계급의 중앙값
df['계급값'] = df['계급'].apply(lambda x:int((x.left + x.right) / 2))
print(df.head(3))

#step 5 :  도수분포표를 만들익 위한 도수 계산
freq = df['계급'].value_counts().sort_index()

#step 6 :  상대 도수 (전체 데이터에 대한 비율) 계산
relative_freq = (freq / freq.sum()).round(2)
print(relative_freq)

#step 7 :  누적 도수 계산 - 계급별 도수를 누적
cum_freq = freq.cumsum()

#step 8 :  도수 분포표 작성  ( 기술 통계 )
dist_table = pd.DataFrame({
    # "156~ 161" 이런 모양 출력하기 
    '계급':[f"{int(interval.left)} ~ {int(interval.right)}" for interval in freq.index],
    # 계급의 중간값
    '계급값':[int((interval.left + interval.right) / 2) for interval in freq.index],
    '도수' : freq.values,
    '상대도수':relative_freq.values,
    '누적도수':cum_freq.values
})
print('도수분포표')
print(dist_table.head(3))

#step 9 : 히스토그램 그리기 
plt.figure(figsize=(8,5))
plt.bar(dist_table['계급값'], dist_table['도수'], width=5, color='cornflowerblue', edgecolor='black')
plt.title('학생 50명 키 히스토그램', fontsize=16)
plt.xlabel('키(계급값)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
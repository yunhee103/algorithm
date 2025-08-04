# 알고리즘은 문제를 해결하기 위한 일련의 단계적 절차 또는 방법을 의미합니다.
# 즉, 어떤 문제를 해결하기 위한 컴퓨터가 따라 할 수 있도록 구체적인 명령어들을 순서대로 나열한 것이라고 할 수 있다.
# 컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 합니다.

# 문제 -> 입력 -> 알고리즘으로 처리 (롤 베이스 : 전통적인 방법 프로그래밍)-> 출력

#문 ) 1~ 10(n) 까지의 정수의 합 구하기
# 방법 1 : o(n) 시간 복잡도
def totFunc(n):
    tot = 0
    for i in range(1, n + 1):
        tot += i
    return tot

print(totFunc(100))  

# 방법 2 : o(1) 시간 복잡도
def totFunc2(n):
    return n *(n + 1) // 2  
print(totFunc2(100))  # 5050

# 주어진 문제를 푸는 방법은 다양하다. 어떤 방법이 더 효과적인지 알아내는 것이 '알고리즘 분석'
# '알고리즘 분석' 평가 방법으로 계산 복잡도 표현 방식이 있다.
# 시간 복잡도 & 공간 복잡도
# 시간 복잡도 : 처리 시간을 분석
# 공간 복잡도 : 메모리의 사용량 분석
# o(빅오)표기법 : 알고리즘의 효율성을 표기해주는 방법
# o(1) : 상수 시간 복잡도, 입력의 크기에 상관없이 일정한 시간이 걸리는 알고리즘

#문2) 임의의 정수들 중 최대값 찾기  
#입력 : 숫자 n개를 가진 list
#최대값찾기
#출력 : 숫자 n개 중 최대값
def findMaxFunc(a):
    n = len(a) 
    max_V = a[0] # 첫 번째 값을 최대값으로 초기화
    for i in range(1, n):
        if a[i] > max_V:
            max_V = a[i]
    return max_V

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc(d))  


def findMaxFunc2(a):
    n = len(a) 
    max_V = 0 # 첫 번째 값을 최대값으로 초기화
    for i in range(1, n):
        if a[i] > a[max_V]:
            max_V = i
    return max_V  #인덱스 반환

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc2(d))  


#문3) 동명이인 찾기 : n명의 사람 이름 중 동일한 이름을 찾아 결과 출력
imsi = ['길동', '순신', '길동', '영희', '길동', '순신']
imsi2 = set(imsi)
imsi = list(imsi2)
print(imsi)  

def findSameFunc(a):
    n= len(a)
    result = set()  # 중복된 이름을 저장할 집합
    for i in range(0, n-1):  # 0부터 n-2까지 반복
        for j in range(i + 1, n): # j는 i+1부터 n-1까지 반복
            if a[i] == a[j]:  # 이름이 같으면
                result.add(a[i])  # 결과 집합에 추가
    return result
           

names = ['Tom', 'Jerry', 'Tom', 'Mike', 'Tom', 'Jerry']
print(findSameFunc(names))  # ['Tom'] (동명이인 이름 리스트 반환)

#문4) 팩토리얼을 구하는 알고리즘
#방법1) for 
def factFunc(n):
    imsi = 1
    for i in range(1, n + 1):
        imsi *= i
    return imsi

print(factFunc(5))  # 120

#방법2) 재귀 호출
def factFunc2(n):
    if n <= 1:  # 종료 조건 필수
        return 1
    return n * factFunc2(n - 1)  # 재귀 호출


print(factFunc2(5))  # 120


# 재귀 연습1) 1부터 n까지의 합 구하기 : 재귀 사용
def hap(n):
    if n == 1:  # 종료 조건
        return 1
    return n + hap(n - 1)  # 재귀 호출  
print(hap(10)) 

# 재귀 연습2) 숫자 n개중 최대값 구하기 : 재귀 사용
def findMax(a,n):
    if n == 1:  # 종료 조건
        return a[0]
    else:
        max_of_rest = findMax(a, n - 1)  # 나머지 숫자들 중 최대값 찾기
        return max(max_of_rest, a[n - 1])  # 현재 숫자와 나머지 중 최대값 비교
    
values = [7, 9, 15, 42, 33, 22]
print(findMax(values, len(values)))  
# 삽입정렬: 자료 배열의 모든 요소를 앞에서부터 차례대로 이미 정렬된 부분과 비교하여 적절한 위치에 삽입하는 방식의 정렬 알고리즘.
# 시간 복잡도: O(n^2) (최악의 경우)
# 알고리즘 과정:
# 1. 첫 번째 원소는 이미 정렬된 것으로 간주하고, 두 번째 원소부터 시작한다.
# 2. 현재 원소를 정렬된 부분과 비교하여 적절한 위치를 찾는다.
# 3. 빈 자리에 현재 원소를 삽입한다.
# 4. 다음 원소로 이동하여 2번과 3번 과정을 반복한다.

# 삽입 정렬(Insertion Sort)은 주어진 데이터 리스트를 순차적으로 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 첫 번째 원소는 정렬된 상태로 간주한다
# 2. 두 번째 원소부터 시작하여, 현재 원소를 정렬된 부분에 적절한 위치에 삽입한다
# 3. 이 과정을 반복하여 전체 리스트를 정렬한다

# 방법1: 원리 이해를 우선
def find_insert_func(r, value): # 이미 정렬된 r의 자료를 앞에서 부터 차례로 확인
    for i in range(0, len(r)): # r의 길이만큼 반복
        if value <= r[i]: # 현재 값이 r[i]보다 작거나 같으면
            return i # 현재 값이 삽입될 위치를 반환
    return len(r) # v가 r의 모든 요소값 보다 클 경우, r의 맨 뒤에 삽입

def ins_sort(a):
    result = []
    while a:  # 리스트가 비어있지 않을 때까지 반복
        value = a.pop(0)  # 리스트의 첫 번째 원소를 제거
        ins_index = find_insert_func(result, value)
        result.insert(ins_index, value)  # 찾은 위치에 현재 원소를 삽입. 이후 값은 오른쪽으로 이동
    return result

d = [2, 4, 5, 1, 3]
print('삽입 정렬 방법1:', ins_sort(d))

# 방법2: 일반적 정렬 알고리즘을 구사 : result X
def ins_sort2(a):
    n = len(a)
    for i in range(1, n):  # 두 번째 원소부터 시작
        value = a[i]  # 현재 원소
        j = i - 1  # 정렬된 부분의 마지막 인덱스
        while j >= 0 and a[j] > value:  # 정렬된 부분을 거꾸로 탐색
            a[j + 1] = a[j]  # 현재 원소보다 큰 값을 오른쪽으로 이동
            j -= 1  # 이전 인덱스로 이동
        a[j + 1] = value  # 현재 원소를 적절한 위치에 삽입
    return a

d = [2, 4, 5, 1, 3]
print('삽입 정렬 방법2:', ins_sort2(d))
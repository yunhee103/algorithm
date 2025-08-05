# 퀵 정렬은 다음과 같은 과정으로 진행한다.
# 정렬은 오름차순으로 진행한다고 가정
#주어진 배열에서 하나의 요소를 선택하고 이를 피벗(pivot)이라고 한다.
#피벗을 기준으로 배열을 두 부분으로 나눈다.
#피벗보다 작은 요소들은 왼쪽 부분에, 피벗보다 큰 요소들은 오른쪽 부분에 위치시킨다.
#왼쪽 부분과 오른쪽 부분에 대해 재귀적으로 퀵 정렬을 적용한다.
#이 과정을 반복하여 전체 배열이 정렬될 때까지 진행한다.    

def quick_sort(a):
    n = len(a)

    if n <= 1:  # 재귀기 때문에 종료조건 걸어놓음
        return a
    # 기준 값
    pivot = a[-1]  # 마지막 원소를 피벗으로 선택 
    g1 = []  # 피벗보다 작은 값들을 저장할 리스트
    g2 = []  # 피벗보다 큰 값들을 저장할 리스트
    for i in range(0, n - 1):  # 마지막 원소는 피벗이므로 제외
        if a[i] < pivot:  # 피벗보다 작은 값
            g1.append(a[i])
        else:  # 피벗보다 큰 값
            g2.append(a[i])

    return quick_sort(g1) + [pivot] + quick_sort(g2)

d = [6, 8, 3, 1, 2, 4, 7, 5]
print(quick_sort(d))


print('-----')
def quick_sort_sub(a, start, end):
    #종료 조건 : 정렬 대상이 한 개 이하이면 정렬할 필요  x
    if end - start <= 0:
        return
# 리스트 하나
    pivot = a[end]  # 마지막 원소를 피벗으로 선택
    i = start  # 피벗보다 작은 값의 인덱스
    for j in range(start, end):  # 피벗을 제외한 나머지 원소들에 대해 반복 (j 값으로 start부터 end-1까지)
        if a[j] <= pivot:  # 피벗보다 작거나 같은 경우
            a[i], a[j] = a[j], a[i]  # i자리에 옮겨주고 한 칸 뒤로 // 피벗보다 작은 값을 왼쪽으로 이동
            i += 1  # 다음 위치로 이동
    a[i], a[end] = a[end], a[i]  # 피벗을 i 위치로 이동 (피벗보다 작은 값들의 오른쪽에 위치)
# 재귀 호출
    quick_sort_sub(a, start, i - 1)  # 피벗 왼쪽 부분 정렬
    quick_sort_sub(a, i + 1, end)  # 피벗 오른쪽 부분 정렬

def quick_sort2(a):
    quick_sort_sub(a, 0, len(a) - 1)
    return a







d = [6, 8, 3, 1, 2, 4, 7, 5]
print(quick_sort2(d))
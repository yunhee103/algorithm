# 버블 정렬은 인접한 두개의 원소를 비교하여 자리를 교환하는 방식이다.

def bubble_sort(a):
    n = len(a)
    while True:  # 무한 루프를 사용하여 정렬이 완료될 때까지 반복
        changed = False  # 자료 변경 여부를 추적하는 변수
        for i in range(n - 1):
            if a[i] > a[i + 1]:  # 인접한 두 원소를 비교 앞이 뒤보다 크면 
                print(a)
                a[i], a[i + 1] = a[i + 1], a[i]
                changed = True  # 자리를 바꾸고 변경 여부를 True로 설정
        if changed == False:
            return  # 변경이 없으면 정렬이 완료된 것이므로 종료

d = [2, 4, 5, 1, 3 ]
bubble_sort(d)
print('---')
print(d)
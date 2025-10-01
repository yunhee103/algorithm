# 자료 구조 중 stack : LIFO 구조
# 1. 클래스 정의와 생성자 (__init__)
class MyStack:
    # __init__ 메서드는 객체가 생성될 때 가장 먼저 호출 (객체가 만들어질 때 초기 상태를 설정하는 역할)
    def __init__(self, iterable=None):
        # '_data'는 스택의 실제 데이터를 저장하는 내부 변수
        # 변수명 앞에 언더스코어(_)를 붙여서 외부에서 직접 접근하지 않도록 권장하는 '캡슐화'를 표현
        self._data = [] 
        # 객체 생성 시 초기 데이터(예: 리스트)를 전달받아 스택에 추가
        if iterable is not None:
            for x in iterable:
                self.push(x)
    # 2. 스택의 핵심 동작 (push, pop, is_empty)
    # push와 pop의 시간 복잡도가 O(1)인 것을 이해하는 것이 중요. 
    # 이는 스택의 크기에 관계없이 항상 일정한 속도로 동작한다는 의미. 
    # 또한 IndexError와 같은 예외 처리를 통해 안정적인 코드를 작성하는 방법
         
    def push(self, x):
        # 맨 위에다가 (top) 원소 추가 (0(1))
        # 스택의 '맨 위(top)'에 원소(x)를 추가.
        # 파이썬 리스트의 append()는 리스트의 끝에 원소를 추가하므로,
        # 이를 스택의 push()로 활용할 수 있음. 시간 복잡도는 O(1).
        self._data.append(x)
        return x 
    
    def pop(self):
        # 맨위 원소 제거 (0(1))
        # 스택의 '맨 위(top)'에 있는 원소를 제거하고 반환.
        # 리스트의 pop() 메서드를 사용하며, 시간 복잡도는 O(1).
        if not self._data:
            # 스택이 비어있는 상태에서 pop을 시도하면 IndexError를 발생시켜 프로그램의 잘못된 사용을 방지. 이것을 '예외 처리'
            raise IndexError('POP FROM EMPTY STACK')
        return self._data.pop()
    
    def is_empty(self):
        # 스택이 비어있는지 확인.
        # 리스트가 비어있으면 `not self._data`는 True를 반환.
        return not self._data
    # 3. 보조 메서드와 특수 메서드
    def __len__(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()
    # 객체를 문자열로 표현할 때 사용
    def __repr__(self):
        top_to_bottom = list(reversed(self._data))
        return f'Stack(top -> bottom {top_to_bottom})'
    
# 4. 동작 시연 함수 (demo_lifo) 동작 확인

# LIFO 동작 확인
# 이 함수는 MyStack 클래스 외부에 정의된 일반 함수
# 클래스를 정의한 후에는 객체(인스턴스)를 생성(s = MyStack())하고 해당 객체의 메서드(예: s.push())를 호출하여 스택의 동작을 확인하는 과정

def demo_lifo():
    s = MyStack()   #MyStack 클래스를 사용하여 새로운 스택 객체 s를 만듦. 스택은 현재 비어있는 상태입니다.
    for item in ['a', 'b', 'c', 'd']:
        s.push(item)    #  현재 반복 중인 item을 스택 s의 가장 위(top)에 추가
        print(f'push {item} -> ' , s) # __repr__이 자동 호출됨

    print('\nPop until empty (LIFO)')
    while not s.is_empty():     # 스택 s가 비어있지 않은 동안 루프를 계속 반복
        print(f'pop->', s.pop(), '|now', s) # |now', s: pop이 실행된 후, 스택 s의 남은 상태를 출력


demo_lifo()
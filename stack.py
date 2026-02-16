class Node:
    def __init__(self, data):
        self.data = data        # 이 노드가 저장할 값(데이터)
        self.next = None        # 다음 노드를 가리키는 링크(처음엔 없음)

class Stack:
    def __init__(self):
        self.top = None         # 스택의 맨 위(top)를 가리키는 포인터 (처음엔 비어있음)

    def push(self, data):
        if self.top is None:            # 스택이 비어있다면 (top이 없음)
            self.top = Node(data)       # 새 노드를 만들어 top으로 설정
        else:                            # 스택에 이미 데이터가 있다면
            node = Node(data)           # 새 노드를 하나 만들고
            node.next = self.top        # 새 노드의 next가 기존 top을 가리키게 한 뒤
            self.top = node             # top을 새 노드로 바꾼다 (맨 위에 쌓임)

    def pop(self):
        if self.top is None:            # 스택이 비어있으면 꺼낼 게 없으니
            return None                 # None 반환
        node = self.top                 # 현재 top 노드를 임시로 저장해두고
        self.top = self.top.next        # top을 "그 다음 노드"로 내린다 (맨 위 제거 효과)
        return node.data                # 제거된 노드의 data를 반환

    def peek(self):
        if self.top is None:            # 스택이 비어있으면
            return None                 # 맨 위가 없으니 None 반환
        return self.top.data            # 맨 위(top)의 데이터만 확인(제거는 안 함)

    def is_empty(self):
        return self.top is None         # top이 None이면 비어있는 스택(True)


if __name__ == "__main__":              # 이 파일을 직접 실행할 때만 아래 코드 실행
    s = Stack()                          # Stack 객체(스택) 하나 생성

    for i in range(3):                   # i = 0, 1, 2 반복
        s.push(chr(ord("A") + i))        # 'A'+i 문자를 push: A, B, C 순서로 스택에 넣음
        print(f"Push data = {s.peek()}") # 방금 넣은 값이 top이므로 peek로 확인해서 출력
    print()                               # 줄바꿈 출력

    while not s.is_empty():              # 스택이 빌 때까지 반복
        print(f"Pop data = {s.pop()}")   # pop 해서 top 값을 꺼내 출력 (C, B, A 순서)
    print()                               # 줄바꿈 출력

    print(f"Peek data = {s.peek()}")     # 다 꺼낸 뒤 peek -> 비었으니 None 출력


def reverse_word(word: str) -> str:
    answer: str = ""
    s = Stack()
    for w in word:
        s.push(w)
    while not s.is_empty():
        answer += s.pop()
    return answer


#테스트 코드



from collections import Counter
import itertools
def removeElement(nums: List[int], val: int) -> int:
    a = 0
    b = len(nums) - 1
    mismatch = 0  # count of elements that are NOT val and are placed/confirmed in the front
    while a <= b:
        # 1) a and b both equal to val -> shrink b
        if nums[a] == val and nums[b] == val:
            b -= 1
        # 2) a is val, b is not val -> swap, confirm front is good
        elif nums[a] == val and nums[b] != val:
            nums[a], nums[b] = nums[b], nums[a]
            mismatch += 1
            a += 1
            b -= 1

        # 3) a is not val, b is val -> confirm a is good, shrink b
        elif nums[a] != val and nums[b] == val:
            mismatch += 1
            a += 1
            b -= 1

        # 4) both are not val -> confirm a is good, move a
        else:
            mismatch += 1
            a += 1

    return mismatch
    
'''
1/26
'''
def maxArea(self, height: List[int]) -> int:
        a = 0
        b = len(height) - 1
        max_area = 0

        while a < b:
            # 넓이를 먼저 계산해야한다
            area = (b - a) * min(height[a], height[b])
            if area > max_area:
                max_area = area

            # 계산한 후에는 포인터를 움직인다
            if height[a] < height[b]:
                a += 1
            elif height[a] > height[b]:
                b -= 1
            else:
                a += 1
        return max_area

def containsNearbyDuplicate(nums: List[int], k: int) -> bool:
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):  # i 다음부터만 찾아서 같은 원소 재사용 방지
            if nums[j] == nums[i] and abs(j-i) <= k:
                return True
    return False
                

    
def maxProfit(prices: List[int]) -> int:
        buy = 0
        sell = 1
        max_profit = 0

        while sell < len(prices):
            profit = prices[sell] - prices[buy]
            if profit > max_profit:
                max_profit = profit

            if prices[buy] > prices[sell]:
                # 손해면: 더 싼 날을 buy로 리셋
                buy = sell
                sell += 1
            elif prices[sell] > prices[buy]:
                # 이득이면: sell만 이동
                sell += 1
            else:
                # 같으면: sell만 이동
                sell += 1

        return max_profit

'''
2/6

'''
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    d = {}
    for i in range(len(nums)):
        #이미 딕셔너리에 들어있는 경우에는 검증하고
        if nums[i] in d and i - d[nums[i]] <= k:
            return True
        #업데이트 해야하는 경우에는 딕셔너리에 추가한다
        d[nums[i]] = i
    return False

def canWinNim(n: int) -> bool:
        #애초에 4개 이하인 수가 뜨면 (1,2,3) 내가 바로 먹어버리고 끝난다
        if n < 4:
            return True
        elif n % 4 == 0:
            return False
        else:
            return True

def isPowerOfFour(self, n: int) -> bool:
        #something of an edge case I guess?
        if n <= 0:
            return False
        elif n == 1:
            return True
        else:
            result = math.log(n,4)
            return result == int(result)

def isUgly(n: int) -> bool:
        #edge cases
        if n <= 0:
            return False
        elif n == 1:
            return True

        while n > 1: #do note for this
            if n %2 == 0:
                n = n//2
            elif n % 3 == 0:
                n = n // 3
            elif n % 5 == 0:
                n = n // 5
            else: #아무거로도 못 나누는 상황이니까 루프에서 얼렁 나와야함
                break #this is needed to break free
        return n == 1

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p1 = list1
        p2 = list2

        dummy = ListNode(0)
        result = dummy  # tail pointer

        while p1 is not None and p2 is not None:
            if p1.val >= p2.val:
                result.next = p2      # attach the node (not the value)
                p2 = p2.next          # advance pointer in list2
            else:
                result.next = p1      # attach the node (not the value)
                p1 = p1.next          # advance pointer in list1

            result = result.next      # advance tail pointer

        # attach the remaining nodes (only one of these is non-None)
        if p1 is not None:
            result.next = p1
        else:
            result.next = p2

        return dummy.next

def is_palindrome_recursive(s: str) -> bool:
    s = s.lower()

    # base case: empty or 1 char is a palindrome
    if len(s) <= 1:
        return True

    # if ends don't match, not a palindrome
    if s[0] != s[-1]:
        return False

    # recursive step: check the inside
    return is_palindrome_recursive(s[1:-1])


def subsets(nums: List[int]) -> List[List[int]]:
    result = []
    for i in range(len(nums)+1):
        shall_add = list(itertools.combinations(nums, i))
        for item in shall_add:
            result.append(list(item))
    return result



from collections import Counter


def canConstruct(ransomNote: str, magazine: str) -> bool:
    ransom_counts = Counter(ransomNote)
    magazine_counts = Counter(magazine)
        
    return ransom_counts <= magazine_counts

def calPoints(self, operations: List[str]) -> int:
        result = []
        
        for item in operations:
            match item:
                case 'C':
                    # Removes the last element (same as del result[-1])
                    result.pop()
                case 'D':
                    # Doubles the last score
                    result.append(result[-1] * 2)
                case '+':
                    # Adds the last two scores
                    result.append(result[-1] + result[-2])
                case _:
                    # The wildcard '_' handles anything else (the numbers)
                    result.append(int(item))
                    
        return sum(result)

def firstUniqChar(s: str) -> int:

    where = {}
    for i in range(len(s)):
        if s[i] in where:
                #이미 본적 있으면 위치를 업데이트 해야하는거임
            where[s[i]] = i
        else:
            where[s[i]] = i
    return where

def addDigits(self, num: int) -> int:
        #Base case
        if num < 10:
            return num
            
        #Main Logic
        s_num = str(num)
        result = 0
        for item in s_num:
            result += int(item)
            
        #Recursive Call
        return self.addDigits(result)

digits = [1,2,3,4]
r = 3



class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]

        items = list(zip(*strs)) #인덱스를 매칭시키는거임
        exact_location = None
        
        for idx, item in enumerate(items):
            if len(set(item)) > 1:
                exact_location = idx
                break
        

        if exact_location is None:
            return strs[0][:len(items)]
        else:
            return strs[0][:exact_location]

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        #일단 매핑을 만들고 시작한다
        mapping_dict = {}
        mapping_dict['2'] = 'abc'
        mapping_dict['3'] = 'def'
        mapping_dict['4'] = 'ghi'
        mapping_dict['5'] = 'jkl'
        mapping_dict['6'] = 'mno'
        mapping_dict['7'] = 'pqrs'
        mapping_dict['8'] = 'tuv'
        mapping_dict['9'] = 'wxyz'

        #베이스 케이스: 길이가 1이다
        if len(digits) == 1:
            lst1 = []
            only = digits[0]              
            for item in mapping_dict[only]:  
                lst1.append(item)
            return lst1

        #recursive case: Take a recursive leap of faith
        else:
            result = []
            first = digits[0]
            rest = digits[1:]
            letters = mapping_dict[first]
            suffixes = self.letterCombinations(rest)
            for c in letters:
                for s in suffixes:
                    result.append(c+s)
            return result

def detectCapitalUse(self, word: str) -> bool:
        #all(['Boo', 'is', 'happy', 'today'])
    
        if all(w.isupper() for w in word):
            return True
            
        if all(w.islower() for w in word):
            return True
            
  
        if word[0].isupper() and all(w.islower() for w in word[1:]):
            return True
            
        return False


from typing import List
from collections import Counter

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        
        # 1. 행(Row) 검사
        for row in board:
            if not self.is_valid_unit(row):
                return False
        
        # 2. 열(Column) 검사 (zip 활용)
        for col in zip(*board):
            if not self.is_valid_unit(col):
                return False
        
        # 3. 3x3 박스 검사
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = []
                for i in range(3):
                    for j in range(3):
                        block.append(board[r + i][c + j])
                
                if not self.is_valid_unit(block):
                    return False
                    
        return True

    # 공통 검증 로직 (행, 열, 박스 모두 이거 하나로 검사)
    def is_valid_unit(self, unit) -> bool:
        counts = Counter(unit)
        for key, count in counts.items():
            if key == '.':
                continue
            if count > 1:
                return False
        return True

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        permutes = itertools.permutations(nums)

        result = []

        for item in permutes:
            result.append(list(item))
        return result


class Solution:
    def isValid(self, s: str) -> bool:
        matching_dict = {')': '(', '}': '{', ']':'['}
        opened = ['(', '{', '[']
        closed = [')', ']', '}']
     
        stack = []
        for word in s:
            if word in opened:
                stack.append(word)

            elif word in closed:
                if len(stack) == 0:
                    return False
                if matching_dict[word] == stack[-1]:
                    stack.pop()
                elif matching_dict[word] != stack[-1]:
                    return False
        return len(stack) == 0


class Solution3:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        idx = 0
        #1. when the stack is empty
        while idx < len(s):
            #스택에 아무것도 안 들어 있는 상황이다
            if len(stack) == 0:
                stack.append(s[idx])
                idx += 1

            #중복 요소를 발견한 경우이다
            elif stack[-1] == s[idx]:
                stack.pop()
                idx += 1

            #중복이 아니다
            elif stack[-1] != s[idx]:
                stack.append(s[idx])
                idx += 1

        #그리고 나서 이제 스택에 들어있는 요소들을 문자열로 만들어서 반환
        return ''.join(stack)

class Solution4:
    def helper(self, word: str) -> str:
        stack = []
        idx = 0
        while idx < len(word):
            c = word[idx]
            if c == '#':
                if stack: #only pop if stack is not empty 
                    #분기문에서 엣지 케이스 생각해보는것이 중요한 이유이다
                    stack.pop()
            else:
                stack.append(c)
            idx += 1
        return ''.join(stack)
    def backspaceCompare(self, s: str, t: str) -> bool:
        return self.helper(s) == self.helper(t)

            
'''
Feb 15th Daily Challenge
151. Reverse Words in a String
'''
class feb15:
    def reverseWords(self, s: str) -> str:
        s = s.strip() #should not contain leading or tailing spaces
        s_list = s.split(' ')
        ans = [] 
        for item in s_list:
            if item:
                ans.append(item)
        ans.reverse()
        return ' '.join(ans)


'''
Feb 16 Daily Challenge
'''
class MinStack:
    def __init__(self):
        self.stack = [] 
        
    def push(self, val: int) -> None:
        self.stack.append(val)
    
    def pop(self) -> None:
        if self.stack:
            self.stack.pop()
        
    def top(self) -> int:
        return self.stack[-1]
        
    def getMin(self) -> int:
        return min(self.stack) #could probably optimize this part

class MyStack:
    def __init__(self):
        self.stack = []
    def push(self, x: int) -> None:
        self.stack.append(x)
    def pop(self) -> int:
        if not self.stack:
            return None 
        return self.stack.pop()
    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        return None
    def empty(self) -> bool:
        return not self.stack

class MyQueue:
    def __init__(self):
        self.queue = []
    def push(self, x: int) -> None:
        self.queue.append(x)
    def pop(self) -> int:
        if not self.queue:
            return None
        else:
            first = self.queue[0]
            del self.queue[0]
            return first
    def peek(self) -> int:
        if not self.queue:
            return None
        else:
            return self.queue[0]
'''
Feb 17 Daily Challenge
'''
class Bits:
    def countBits(self, n: int) -> List[int]:
        lst = []
        for i in range(n+1):
            i_str = str(bin(i)[2:])
            count = 0
            for item in i_str:
                if '1' in item:
                    count += 1
            lst.append(count)
        return lst

class Sttairs:
    def climbStairs(self, n: int) -> int:
        lst = [1,2]
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            while len(lst) < n:
                lst.append(lst[-1] + lst[-2]) #이것도 다이나믹 프로그래밍임
        return lst[-1]

# nums = [2,3,1,1,4]
# # 시작: 0번 인덱스에서 시작한다
# # you can add range of numbers from 1 to idx
# # 그렇게 이동한 위치에서 또 1 to idx 만큼 더한다
# # 이렇게 나온 숫자의 조합으로 len(nums)-1에 도달 할 수 있는지 본다
# def jump_game(nums):
#     start = 0
#     for idx, i in enumerate(nums):
#         start += 
#         #start에 어떤 식으로 더해서 len(nums)-1이 나오게 할 수 있는지 보고 싶다
#         #이 로직만 다이나믹 프로그래밍으로 해결 할 수 있으면 정답이 나올것 같은데
#         #더하는 로직이 이게 아닌거 같은데
#         if start == len(nums)-1:
#             return True
#     return False 

def jump_game_greedy(nums):
    'Failed to solve'
    max_reach = 0 
    
    for idx, jump_len in enumerate(nums):
        if idx > max_reach:
            return False
        
        current_jump = idx + jump_len
        if current_jump > max_reach:
            max_reach = current_jump
            
        if max_reach >= len(nums) - 1:
            return True
            
    return False

'''
FEB 18 Daily Challenge

힌트를 보고 max depth를 구현한것은 아쉬웠음. stack 로직으로 그걸 계산하는 과정은 괜찮았음
'''

class Solution:
    def maxDepth(self, s: str) -> int:
        #open bracket이면 푸쉬
        #close bracket 이면 팝
        #push pop을 트랙하는 스택이라는 리스트를 만들어야 하나
        #리스트의 최대 길이를 기억하는 변수를 가지고 있는다
        #리스트의 길이가 최대일때 그 값을 기억해두고 있다가
        #문자열을 끝꺼지 돌고 나서 다 끝나면 리턴한다
        stack = []
        max_length = 0
        for word in s:
            #지금 당장 스택의 최대 길이가 얼마일지 저장하는 변수
            
            if not max_length or len(stack) > max_length:
                max_length = len(stack)

            #open bracket을 만났기 때문에 stack에 추가한다
            if word == '(':
                stack.append(word)

            #close bracket so pop
            elif word == ')':
                stack.pop()
        return max_length

'''Feb 19 Leetcode challenge'''
class SolutionFeb19:
    def finalPrices(self, prices: List[int]) -> List[int]:
        idx = 0 
        final_output = [] 
        while idx < len(prices):
            current = prices[idx] 
            length = len(prices)
            conditions_met = False
            for i in range(idx+1,length):
                if prices[i] <= current:
                    final_output.append(current - prices[i])
                    conditions_met = True
                    break
            if not conditions_met:
                final_output.append(current)
            idx += 1
        return final_output
#구조 자체는 괜찮았지만 로직에 빈틈이 많이 있었음
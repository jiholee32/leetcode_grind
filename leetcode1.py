

def twoSum1(nums: List[int], target: int) -> List[int]:
    for i in range(len(nums)):
        need = target - nums[i]
        for j in range(i + 1, len(nums)):  # i 다음부터만 찾아서 같은 원소 재사용 방지
            if nums[j] == need:
                return [i, j]

#TODO: use one for loop


'''
1/4 
오늘 배운것
range 값 조심하기, set 활용하기 (특히 - 하는데)
ch.isalnum()
필요한 정보만 골라내기 => 요약값 하나만 골라낼 수 있는가?
파이썬에서 딕셔너리의 비교. 키와 밸류가 완전히 같으면 둘이 같은 딕셔너리이다
'''


def missingNumber(nums: List[int]) -> int:
    length = len(nums) #length of nums
    all_values = [] #list that has all the values within the correct range
    for i in range(0,length+1): #여기서 위험했다
        all_values.append(i)
    missing_no = set(all_values) - set(nums)
    missing_no = list(missing_no)
    for item in missing_no:
        return item #O(2n)

def isPalindrome2(s: str) -> bool:
    #1 remove white spaces
    no_space = s.replace(' ', '')
    if not no_space: 
        return True
    #2 convert to lower cases (since it has characters)
    lower = no_space.lower()
    #3 removing non-alphanumeric character
    alpha_num = ''
    for item in lower:
        if item.isalnum():
            alpha_num += item
    #4 check if it is actually a palindrome
    check = alpha_num[::-1] #flipped the order of strings
    return (check == alpha_num)

def jihoProfit(prices: List[int]) -> int:
    'not so efficient version made by Jiho'
    prof = []
    length = len(prices)
    x = 0
    while x < length-1:
        buy = prices[:x+1] #had to watch out for this
        sell = prices[x+1:]
        for b in buy:
            for s in sell:
                prof.append(s-b)
        x += 1
    if max(prof) < 0:
        return 0
    else:
        return max(prof)

def maxprofit(prices: List[int]) -> int:
    'return max profit. This is the efficient version'
    min_price = prices[0]
    max_profit = 0
    for i in range(1, len(prices)):
        profit = prices[i] - min_price
        max_profit = max(max_profit, profit)
        min_price = min(min_price, prices[i])
    return max_profit

# “모든 경우를 다 계산하지 말고, 문제에서 정말 필요한 정보만 유지하라.”
# 지금까지의 최저가, 지금까지의 최대 이익 => 이 두가지만 알면 된다

def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    counter_s = {}
    for char in s:
        if char not in counter_s:
            counter_s[char] = 1
        else:
            counter_s[char] += 1

    counter_t = {}
    for char in t:
        if char not in counter_t:
            counter_t[char] = 1
        else:
            counter_t[char] += 1

    return (counter_s == counter_t) 
    # 파이썬에서 딕셔너리 비교 ==는 순서가 아니라 내용을 비교해.
    # 키 집합이 완전히 같고
    # 각 키에 대한 값도 완전히 같으면 → True

'''
1/5
LIFO
Stack
bin(n) => 이거에 대해서도 새로 배움
'''
def isValid(word: str) -> bool:
    'I have to use stack for this '
    open_sign = ['[', '(', '{']
    close_sign = [']', ')', '}']
    matching = {')':'(', '}':'{', ']':'['}
    stack = []
    for char in word:
        if char in open_sign:
            stack.append(char)
        elif char in close_sign: #두개가 동시에 참일수는 없으니까 이렇게 하는게 좋음
            if len(stack) == 0: #if close sign comes first
                return False
            elif stack[-1] == matching[char]: #if close and open sign matches
                stack.pop()
            else:
                return False

    return (len(stack) == 0)

#can we use it like a number
# student id average would not mean anything => categorical
# do not think like a computer science. Think like a statistician


def hammingWeight(n: int) -> int:
    'I learned how to use bin today'
    a = bin(n)
    binary = a[2:]
    binary_list = list(binary)
    one_counter = 0
    for item in binary_list:
        if item == '1':
            one_counter += 1
    return one_counter
    #one_counter = binary.count('1')

'''
1/6
binary.zfill(32)
len(n) != len(set(n))
default argument for int
'''

def reverseBits(n: int) -> int:
    'reverse 32 bits'
    binary = bin(n)[2:] #in the binary form             
    pad = 32 - len(binary) #how many zeros in front of the binary
    padded = "0" * pad + binary     
    #binary.zfill(32)
    reversed_bit = padded[::-1]      
    return int(reversed_bit, 2) #문자열 2진수로 읽어서 정수로 바꾸기
    #the concept of defualt argument comes in here as well

def has_duplicate(a):
    x = 0
    while x < len(a):
        checking_element = a[x]              # 매번 현재 기준 원소 갱신
        for i in range(x + 1, len(a)):       # only after x
            if checking_element == a[i]:
                return True                  
        x += 1
    return False
    # len(n) != len(set(n))

'''
1/7
'''

def sqrt(x:int) -> int:
    'return the square root of x rounded down to the nearest integer'
    item = []
    new_item = []
    
    #first: if x is 0 or 1
    if x < 2:
        return x
    #next: if number is greater than 1
    else:
        for i in range(x):
            item.append(i)
        for n in item:
            if n*n <= x:
                new_item.append(n)
    return max(new_item)

def romanToInt(s: str) -> int:
    'convert roman numeral to int'
    pass

# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
        
'''
1/9
안되는 조건을 먼저 만들어두는게 좋은 습관인거 같다
'''

from typing import List

def triangleType(nums: List[int]) -> str:
    a, b, c = nums

    # 먼저 삼각형 성립 여부 확인 (세 조건 모두 만족해야 함)
    if not ((a + b) > c and (a + c) > b and (b + c) > a):
        return "none"

    # 정삼각형
    if a == b and b == c:
        return "equilateral"

    # 이등변삼각형 (정삼각형은 위에서 걸렀으니 '정확히 두 변 같음' 만족)
    if a == b or a == c or b == c:
        return "isosceles"

    # 나머지는 세 변 모두 다름
    return "scalene"

def longestCommonPrefix(strs: List[str]) -> str:
    if not strs:
        return ""

    # 가장 짧은 문자열 길이까지만 비교 가능
    possible_max = min(len(s) for s in strs)
    standard = strs[0]

    answer = []
    for idx in range(possible_max):
        ch = standard[idx]
        # 같은 idx의 글자가 모든 문자열에서 동일한지 확인
        for s in strs[1:]:
            if s[idx] != ch:
                return "".join(answer)
        answer.append(ch)

    return "".join(answer)


def lengthOfLastWord(s: str) -> int:
    s = s.split(' ')
    no_space = []
    for item in s:
        if item:
            no_space.append(item)
    return len(no_space[-1])

def searchInsert(nums: List[int], target: int) -> int:
    'this is binary search'
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1 #타겟보다 작으면 시작 범위를 오른쪽으로 가는거지
        else:
            right = mid - 1

    return left #earch Insert Position은 “실패했을 때도 답이 있다(삽입 위치)”가 포인트야
    # 정석적인 binary search라면 return None을 해야한다


def threeConsecutiveOdds(lst: List[int]) -> bool:
    idx = 0
    length = len(lst)
    while idx + 2 < length: #this is better
        if lst[idx] % 2 != 0 and lst[idx+1] % 2 != 0 and lst[idx+2] % 2 != 0:
            #시간복잡도도 O(n)이라 요구에 딱 맞아
            return True
        idx += 1
    return False
'''
1/10
find 
join
'''

def strStr(haystack: str, needle: str) -> int:
    return haystack.find(needle)
    #사실상 이게 핵심 아이디어 였음
    
def plusOne(digits: List[int]) -> List[int]:

    str_digits = []
    for digit in digits:
        str_digits.append(str(digit))
    str_num = ''
    for s in str_digits:
        str_num += s

    plus_one = int(str_num) + 1
    str_plus_one = str(plus_one)

    answer = []
    for item in str_plus_one:
        answer.append(int(item))
    return answer


from typing import List
def plusOne_correct(digits: List[int]) -> List[int]:
    # digits -> "123"
    s = ''.join(str(d) for d in digits)

    # +1 -> "124"
    s = str(int(s) + 1)

    # "124" -> [1, 2, 4]
    return [int(ch) for ch in s]

def jihoVowels(msg: str) -> str:
    'this works but it is long and messy'
    vowels = 'aeiouAEIOU'
    vowels_idx = []
    vowels_itself = []
    length = len(msg)
    for i in range(length):
        if msg[i] in vowels:
            vowels_idx.append(i)

    for word in msg:
        if word in vowels:
            vowels_itself.append(word)
    vowels_itself = vowels_itself[::-1] #리스트를 뒤집는다

    mapping_dict = {} #{0: 'a', 2: 'e', 6: 'e', 7: 'i'}
    for i in range(len(vowels_idx)): #well they have to be equal in length so
        mapping_dict[vowels_idx[i]] = vowels_itself[i]

    translate_table = str.maketrans('', '', 'aeiouAEIOU')
    msg = msg.translate(translate_table)

    lst_msg = list(msg)
    for key, value in mapping_dict.items():
        lst_msg.insert(key, value)
    answer = ''.join(lst_msg)

    return answer
'''
1/12 
swapping
two_pointers
'''
    
def reverseVowels(s: str) -> str:
    s_list = list(s)
    left, right = 0, len(s_list) - 1
    vowels = set("aeiouAEIOU")

    while left < right:
        if s_list[left] not in vowels:
            left += 1
        elif s_list[right] not in vowels:
            right -= 1
        else:
            s_list[left], s_list[right] = s_list[right], s_list[left]
            left += 1
            right -= 1

    return "".join(s_list)

def moveZeroes(nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        non_zero = 0
        zero = 0
        length = len(nums)
        while non_zero < length:
            if nums[non_zero] != 0:
                nums[non_zero], nums[zero] = nums[zero], nums[non_zero]
                zero += 1
            non_zero += 1
        
def isSubsequence(s: str, t: str) -> bool:
    #빈 문자열이면 항상 true
    if not s:
        return True
    i = 0 #pointer for s
    j = 0 #pointer for t
    length = len(t)
    while j < length and i < len(s): #엔딩조건
        if s[i] == t[j]: #두개가 매칭하면
            i += 1 #i 하나 증가
        j += 1 #j는 항상 증가함
    return i == len(s)


def reverse_string(word: str) -> str:
    'reverse a string'
    word_list = list(word)
    right = len(word) - 1
    left = 0
    while left < right:
        # Swap the characters
        word_list[right], word_list[left] = word_list[left], word_list[right]
        
        # Update the pointers (CRITICAL STEP)
        left += 1
        right -= 1

    final_output = ''.join(word_list)
    return final_output




def strStr(haystack: str, needle: str) -> int:
    if needle == "":
        return 0

    n, m = len(haystack), len(needle)

    for start in range(n - m + 1):
        if haystack[start:start + m] == needle:
            return start
    return -1

'''
1/18 
포인터 이동은 루프 안에서
루프 제어는 필요할때만
문자열 직접 못 바꿈
'''
           
def reverseOnlyLetters(s: str) -> str:
    start = 0
    end = len(s) - 1
    list_s = list(s)

    while start < end:
        if list_s[start].isalpha() and list_s[end].isalpha():
            list_s[start], list_s[end] = list_s[end], list_s[start]
            start += 1
            end -= 1

        elif list_s[start].isalpha() and not list_s[end].isalpha():
            end -= 1

        elif not list_s[start].isalpha() and list_s[end].isalpha():
            start += 1

        else: 
            start += 1
            end -= 1

    return ''.join(list_s)


def isPalindrome(s: str) -> bool:
    s  = s.lower()
    lst = []
    for item in s:
        if item.isalnum():
            lst.append(item)
    modified = ''.join(lst)
    return modified == modified[::-1]






def is_prime(num):
    for i in range(2,num):
        if num%i == 0:
            return False
    return True

def countPrimes(n: int) -> int:
    counter = []
    if n in (0,1):
        return 0
    else:
        for i in range(2,n):
            #to include n
            if is_prime(i) == True:
                counter.append(i)
    return len(counter)

def isSubsequence(s: str, t: str) -> bool:
    s_index = 0
    t_index = 0

    while s_index < len(s) and t_index < len(t):
        if s[s_index] == t[t_index]:
            s_index += 1
            t_index += 1
        else:
            t_index += 1

    return s_index == len(s)

def findTheDifference(s: str, t: str) -> str:
    # s = abc
    # t = abcd
    # 대충 이런식이다
    s_index = 0
    t_index = 0
    if s == '':
        return t[-1]
    #루프에서 정답을 찾는 경우
    while s_index < len(s):
        if s[s_index] == t[t_index]:
            s_index += 1
            t_index += 1
        else:
            return t[t_index]
    #루프에서 정답을 못 찾았으면 마지막 요소가 정답이라는 뜻이다
    return t[-1]


def thirdMax(nums: List[int]) -> int:
        if len(nums) < 3:
            return max(nums)
    
        nums_sorted = sorted(nums)
        nums_sorted = list(set(nums_sorted))[::-1]
        return nums_sorted[2]

def canPlaceFlowers(flowerbed: List[int], n: int) -> bool:
    pass

def guessNumber(n: int) -> int:
    low, high = 1, n #setting the boundaries
    while low <= high:
        mid = (high+low) // 2
        res = guess(mid)

        if res == 0:
            return mid
        elif res == -1:
            high = mid - 1
        else:  # res == 1
            low = mid + 1

def search(nums: List[int], target: int) -> int:
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            end = mid - 1
        elif nums[mid] < target:
            start = mid + 1
    return -1


def nextGreatestLetter(self, letters: List[str], target: str) -> str:

    end = len(letters) - 1 #end index
    begin = 0 #begin index
    while begin <= end: 
        mid = (end+begin) // 2
        if letters[mid] == target:
            begin = mid + 1
        elif letters[mid] < target:
            begin = mid + 1
        elif letters[mid] > target:
            end = mid - 1
        else:
            return letters[0]
    return letters[mid]

def moveZeroes(nums: List[int]) -> 'List':
    """
    Do not return anything, modify nums in-place instead.
    """
    zero = 0
    non_zero = 0
    n = len(nums)

    while non_zero < n and zero < n:
        # zero는 0을 가리키도록(또는 찾도록) 맞춘다
        if nums[zero] != 0:
            zero += 1
            continue

        # non_zero는 zero보다 뒤에서 non-zero를 찾는다
        if non_zero <= zero:
            non_zero = zero

        # case 1: 둘 다 0
        if nums[zero] == 0 and nums[non_zero] == 0:
            non_zero += 1

        # case 2: zero는 0, non_zero는 non-zero -> swap
        elif nums[zero] == 0 and nums[non_zero] != 0:
            nums[zero], nums[non_zero] = nums[non_zero], nums[zero]
            zero += 1
            non_zero += 1

        # case 3: zero는 non-zero, non_zero는 0 (사실 위에서 zero를 0으로 맞춰서 거의 안 나옴)
        elif nums[zero] != 0 and nums[non_zero] == 0:
            zero += 1

        # case 4: 둘 다 non-zero
        else:
            zero += 1
        
def firstPalindrome(words: List[str]) -> str:
    for word in words:
        if word == word[::-1]:
            return word #concept of returning right away
def getCommon(nums1: List[int], nums2: List[int]) -> int:
    set1 = set(nums1)
    set2 = set(nums2)
    set3 = set1.intersection(set2)
    if len(set3) == 0:
        return -1
    else:
        return min(set3)


def reversePrefix(s: str, k: int) -> str:
    list_s = list(s)
    list_s[:k] = list_s[:k][::-1]
    return ''.join(list_s)

'''
1/25
'''

def twoSum(numbers: List[int], target: int) -> List[int]:
    left = 0
    right = len(numbers)-1
    ans = []
    while left <= right:
        #1번조건: 정답이 나오는 경우
        if numbers[left] + numbers[right] == target:
            ans.append(left+1)
            ans.append(right+1)
            return ans #I can just return right away no?
        #2번조건: 정답보다 계산값이 큰 경우
        elif numbers[left] + numbers[right] > target:
            right -= 1
        
        #3번조건: 정답보다 계산값이 작은 경우
        else:
            left += 1
def sortedSquares(nums: List[int]) -> List[int]:
    pass

# a = [-1,-4,0,3,10]
# #how to find first negative number in a list?
# c = 0
# for item in range(len(a)):
#     if item < 0:
#         c += a[item]
#         break
# print(c)
#this means I add all the fucking negative numbeers, which is not what I was aiming for


# def find_indexes(nums):
#     first_neg = None
#     first_nonneg = None
#     for i, x in enumerate(nums):
#         if first_neg is None and x < 0:
#             first_neg = i
#         if first_nonneg is None and x >= 0:
#             first_nonneg = i
#         if first_neg is not None and first_nonneg is not None:
#             break
#     return first_neg, first_nonneg

def sortedSquares(nums: List[int]) -> List[int]:
        res = []

        l, r = 0, len(nums) - 1

        while l <= r:
            if nums[l] * nums[l] > nums[r] * nums[r]:
                res.append(nums[l] * nums[l])
                l += 1
            else:
                res.append(nums[r] * nums[r])
                r -= 1

        return res[::-1]  # reverse

#this might be the best I wrote lol
def sortArrayByParity(self, nums: List[int]) -> List[int]:
    a = 0
    b = len(nums)-1
    while a < b: #so this is safer approach apparently

        #1. a and b is both odd number
        # in this case, only move the b pointer
        if nums[a] % 2 != 0 and nums[b] %2 != 0:
            b -= 1
        #2. a is odd and b is even:
        # in this case, flip the numbers and move both pointer
        elif nums[a] % 2 != 0 and nums[b] % 2 == 0:
            nums[a], nums[b] = nums[b], nums[a]
            a += 1
            b -= 1 

        #3. a and b is both even:
        #in this case, only move the a pointer
        elif nums[a] % 2 == 0 and nums[b] % 2 == 0:
            a += 1

        #4. a is even and b is odd:
        # in this case, move both pointers. Howevever, the flip does not happen
        else:
            a += 1
            b -= 1
    return nums
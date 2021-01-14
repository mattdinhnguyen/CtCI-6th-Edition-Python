from collections import deque

class Solution:
    def __init__(self):
        self.INT_MAX = 2**31 - 1
        self.INT_MIN = -2**31 + 1
        self.MOD = 1000000007
    # based 26 string to decimal
    def excelTitleToNumber(self, s: str) -> int:
        res, j = 0, len(s)-1
        for c in s:
            i = ord(c) - ord('A') + 1
            res += i*(26**j)
            j -= 1
        return res
    # decimal to based 26 number
    def convertToTitle(self, n) -> str:
        s = ""
        while n:
            rem = n%26
            if rem:
                s = chr(ord('A')+ rem-1) + s
                n //= 26
            else:
                s = 'Z' + s
                n = n//26 -1
        return s
    def convertToTitle(self, n):
        s, ordA = "", ord('A')-1
        while n > 0:
            bit_val = n % 26 or 26
            n = (n - bit_val) // 26
            s = chr(bit_val + ordA) +s
        return s
    def generate_primes(self, n):
        if not n or isinstance(n, float): raise TypeError("invalid number")
        primes = 2*[False] + (n-1)*[True]
        for i in range(2, int(n**0.5+1.5)):
            if primes[i]:
                for j in range(i*i, n+1, i):
                    primes[j] = False
        return primes
    def primesum(self, e):
        primes = self.generate_primes(e)
        for p in range(2,e//2+1):
            if primes[p] and primes[e-p]:
                return [p,e-p]
        return [0,0]
    def primeFactors(n): 
        ret = []
        while n % 2 == 0: # number of two's that divide n 
            ret.append(2)
            n = n // 2
        for i in range(3,int(n**0.5)+1,2): # n must be odd at this point, kip of 2 ( i = i + 2)
            while n % i== 0:
                ret.append(i)
                n = n // i
        if n > 2: ret.append(n) # if n is a prime, n > 2 
        return ret
    def gcd(self, m, n):
        dividend, divisor = (m,n) if m > n else (n,m)
        while divisor:
            divisor, dividend = dividend%divisor, divisor
        return dividend
    def nextSimilarNum(self, ns):
        if len(ns) < 2 or len(set(ns)) < 2: return -1
        if len(ns) == 2: return -1 if ns[1]>ns[0] else ns[::-1]
        stax = [ns[-1]]
        newS, minIdx = "", len(ns)-1
        for i in range(minIdx-1,-1,-1): # starting at ns[-2], look for smaller digit
            if ns[i] >= ns[i+1]:
                stax.append(ns[i])
                if ns[minIdx] > ns[i]: minIdx = i # min digit up to now
                continue
            if ns[minIdx] > ns[i]: # found digit smaller than ns[minIdx], re-arrange for "251678"
                newS = ns[:i] + ns[minIdx] + ns[i]
                stax.remove(ns[minIdx])
                newS += "".join(stax)
            else:
                for j in range(len(stax)): # found digit == ns[minIdx] or ns[i] < ns[i+1], re-arrange for "32256" and "892957"
                    if ns[i] < stax[j]: # 
                        newS = ns[:i] + stax[j] + "".join(stax[:j]) + ns[i] + "".join(stax[j+1:])
                        break
            break
        return newS or -1
    def nextSimilarNum(self, s):
        for i in range(len(s) - 2, -1, -1):
            tail = sorted(s[i:]) # brute force sort tail and check if > int(s)
            for j in range(len(tail)):
                m = int(s[:i] + tail[j] + ''.join(tail[:j] + tail[j+1:]))
                if m > int(s): # found m > int(s)
                    return m
        return -1
    def nextSimilarNum(self, A): # fastest
        l = len(A)
        idx = l-2
        while idx >= 0:
            if A[idx] < A[idx+1]: # start from A[-2], found idx smaller digit
                j = l -1
                while A[j] <= A[idx]: j -= 1 # start from A[-1], look for A[j] bigger A[idx]
                A = A[:idx] + A[j] + A[l-1:j:-1] + A[idx] + A[j-1:idx:-1] # re-arrange A[idx:]
                return A
            idx -= 1
        return -1
    def prettyPrint(self, A):
        mat = [[1]]
        for i in range(2,A+1):
            mat = [ [i] + r + [i] for r in mat]
            r = [i]*len(mat[0])
            mat = [r] + mat + [r]
        return mat
    def reverse_bit(self, num):
        ans = 0
        for _ in range(32):
            ans <<= 1
            ans += num&1
            num >>= 1
        return ans
    # 
    def leftmostSetBit(self, x):
        count = 0
        while (x):
            count += 1
            x = x >> 1
        return count
    def isKthBitSet(self, x, k):
        return 1 if x & (1 << (k - 1)) else 0
    def isBinPalindrome(self, x):
        l = self.leftmostSetBit(x)
        r = 1
        while (l > r): # Compare left and right bits and converge
            if (self.isKthBitSet(x, l) != self.isKthBitSet(x, r)):
                return 0
            l -= 1
            r += 1
        return 1
    def findNthBinPalindrome(self, n):
        pal_count = 0; i = 0
        while pal_count < n:
            i += 1
            if (self.isBinPalindrome(i)):
                pal_count += 1
        return i
    # https://www.interviewbit.com/problems/palindromic-binary-representation/
    def findNthBinPalindrome(self, A):
        if(A == 1): return 1
            
        pallin = ['1']
        
        for i in range(1,A):
            plen = len(pallin)
            
            
            if(plen == pallin.count('1')):
                pallin = ['1']+['0']*(plen-1)+['1']
            else:
                if(plen&1):
                    if(pallin[plen//2] == '0'):
                        pallin[plen//2] = '1'
                    else:
                        pallin[plen//2] = '0'
                        i = 1
                        while(pallin[plen//2-i] == '1'):
                            pallin[plen//2-i] = '0'
                            pallin[plen//2+i] = '0'
                            i+=1
                        pallin[plen//2-i] = '1'
                        pallin[plen//2+i] = '1'
                else:
                    if(pallin[plen//2]=='0'):
                        pallin[plen//2] = '1'
                        pallin[plen//2-1] = '1'
                    else:
                        pallin[plen//2] = '0'
                        pallin[plen//2-1] = '0'
                        i = 2
                        while(pallin[plen//2-i] == '1'):
                            pallin[plen//2-i] = '0'
                            pallin[plen//2+i-1] = '0'
                            i+=1
                        pallin[plen//2-i] = '1'
                        pallin[plen//2+i-1] = '1'
        
        return int("".join(pallin),2)
    def findNthBinPalindrome(self, A):
        if A==1: return 1
        if A==2: return 3
        count=2;n=3;lcount=0
        while A>count:
            lcount = count
            count += pow(2,(n-2)%2+(n-2)//2)
            n += 1
        ans=0
        ans |= 1<<(n-2) # first and last bit cannot be zero as specified therefore they must be 1
        ans |= 1 
        value = A-lcount # if we use n-1 bit to represent palindrome then A-lcount will specify its number in those combination.
        j=(n-1)//2 # used to specify the bit place
        key=2 # used to determine the value of the bit in the palindrome whether 0 or 1
        while j<n-2:
            if value%key==0 or value%key-key//2>0:
                ans |= 1<<j
                ans |= 1<<(n-2-j)
            j += 1
            key *=2
        return ans
    # BFS https://www.geeksforgeeks.org/find-n-th-number-whose-binary-representation-palindrome/
    def binS2Decimal(self, s): # convert binary string into integer
        num  = 0
        for ch in s:
            num = num*2 + int(ch)
        return num
    def findNthBinPalindrome(self, A):
        if A == 1: return A
        A -= 1
        q = deque(["11"]) # start with 2nd binary palindrome string
        while q: # iterate q elements till found nth binary palindrome (A==0)
            s = q.popleft()
            A -= 1
            if not A: # found the n'th binary palindrome
                return self.binS2Decimal(s)
            mid = len(s)//2
            if len(s)%2: # odd length: clone s[mid]
                q.append(s[:mid]+s[mid]+s[mid:])
            else: # even length: insert 0|1
                q.append(s[:mid]+"0"+s[mid:])
                q.append(s[:mid]+"1"+s[mid:])
        return 0
    def diffBitSumPairwise(self, A):
        n = len(A); sum = 0
        for i in range(31): # in +ve numbers 31st bit will always be 0 so no need to check that
            count = 0 # count number of elements with i'th bit set
            for j in range(n):
                if A[j] & (1<<i):
                    count += 1 # count is the number of set(1) bits, so (n - count) will give number of unset(0) bits
            sum += count*(n-count)*2 % self.MOD # we are calculating differences for (x,y) so for (y,x) just multiply it with 2
                                               # So FORMULA = COUNT * (N - COUNT ) *2
            if sum >= self.MOD:
                sum -= self.MOD
        return sum
    def divide(self, dividend, divisor):
        result, sign = 0, 1
        if dividend < 0:
            dividend = -dividend
            sign = -sign
        if divisor < 0:
            divisor = -divisor
            sign = -sign
        if divisor == 1:
            # if sign < 0 and -dividend < self.INT_MIN:
            #     dividend = -self.INT_MIN
            if sign > 0 and dividend > self.INT_MAX:
                dividend = self.INT_MAX
            return -dividend if sign < 0 else dividend
        while dividend >= divisor:
            result += 1
            dividend -= divisor
        if result > self.INT_MAX:
            result = self.INT_MAX
        return -result if sign < 0 else result
    def divide(self, A: int, B: int) -> int:
        if abs(B) > abs(A): return 0
        if A >= self.INT_MAX and B == 1 or A <= -self.INT_MAX and B == -1:
            return self.INT_MAX
        if A == -(self.INT_MAX+1) and B == 1 or A == self.INT_MAX+1 and B == -1:
            return -(self.INT_MAX+1)
        if abs(A) > self.INT_MAX and abs(B) == 1: return B*self.INT_MAX
        if B == 0: return self.INT_MAX
        neg = -1 if A>0 and B<0 or A<0 and B>0 else 1
        A, B = abs(A), abs(B)
        div = -1
        while A >= 0:
            A -= B
            div += 1
            if div > self.INT_MAX and neg+1: return self.INT_MAX
            elif div > self.INT_MAX and not neg+1: return self.INT_MAX + 1
        return -div if neg < 0 else div
if __name__ == '__main__':
    sol = Solution()
    print(sol.diffBitSumPairwise([1,3,5]))
    # print(sol.prettyPrint(4))
    # print(sol.excelTitleToNumber('AZZ'))
    # print(sol.convertToTitle(1378))
    # print(sol.primesum(4))
    # print(sol.primesum(16777214))
    # print(sol.gcd(1024,16))
    # assert sol.nextSimilarNum("903885770893074783710083450145620356667677191627276513995926532") == "903885770893074783710083450145620356667677191627276513995932256"
    # assert sol.nextSimilarNum("892795") == "892957"
    # assert sol.nextSimilarNum("218765") == "251678"
    # r = sol.reverse_bit(3)
    # print(f'{3:b}', f'{r:b}', r)
    # assert r == 3221225472
    # r = sol.reverse_bit(1)
    # print(f'{1:b}', f'{r:b}', r)
    # assert r == 2147483648
    # print(sol.divide(2147483648,1))
    # print(sol.divide(-2147483648,-1))
    # print(sol.divide(-2,-2))
    # for i in range(1,29):
    #     print(f'{i:b}', sol.isBinPalindrome(i))
    # assert sol.findNthBinPalindrome(1) ==  1
    # assert sol.findNthBinPalindrome(9) ==  27 # 10101
    # assert sol.findNthBinPalindrome(14) ==  63 # 10101
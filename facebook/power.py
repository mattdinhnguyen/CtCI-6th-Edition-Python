import unittest

class Solution:
    # def myPow(self, base: float, expo: int) -> float:
    def myPow(self, base, expo):
        inverse = 0; ans = sign = 1
        if base < 0:
            y = -base; sign = -1 if expo%2 else 1
        else: y = base
        if expo < 0:
            expo = -expo; inverse = 1
        while expo > 0:
            if expo&1:
                ans *= y
            y *= y
            expo >>= 1
        return ans*sign if not inverse else 1/ans*sign
    def power(self, base, expo, m):
        if m == 1: return 0
        ans = 1; y = base%m; inverse = 0
        if expo < 0:
            expo = -expo
            inverse = 1
        while expo > 0:
            if expo&1:
                ans = (ans*y)%m
            y = (y*y)%m
            if y < 0: y+=m
            expo >>= 1
        if ans < 0:
            ans = m - abs(ans)%m
            return ans if not inverse else 1/ans
        return ans%m if not inverse else 1/ans
    def power(self, x, n, d):
        res = 1 % d  # Cover case d == 1
        while n > 0:
            if n & 1:   # Odd?
               res = (res * x) % d
            x = (x*x)%d
            n >>= 1
        return res
class Test(unittest.TestCase):
    def test_power(self):
        sol = Solution()
        assert sol.power(2.0,-2,100000) == 0.25000
        assert round(sol.power(-2.0,2,100000),5) == 4
        assert round(sol.power(2.1,3,100000000),5) == 9.26100
        assert sol.power(-2,3,3) == 1
        r = sol.power(3,9,100000000)
        print(r,3**9)
        assert r == 3**9
        assert sol.power(2,3,3) == 2

if __name__ == "__main__":
    unittest.main()

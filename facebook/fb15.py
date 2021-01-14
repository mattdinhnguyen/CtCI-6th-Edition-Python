# O(N)
import unittest
from typing import List
# https://leetcode.com/problems/move-zeroes/submissions/
def moveZeroes(self, nums: List[int]) -> None:
    for i in range(len(nums))[::-1]:
        if nums[i] == 0:
            nums.pop(i)
            nums.append(0)
def move_zeros_to_left(A):
    j = 0
    for i in range(len(A)-1,-1,-1):
        if A[i]:
            if j:
                A[j] = A[i]
                j -= 1
        else:
            if not j: j = i
    while j > -1:
        A[j] = 0
        j -= 1
    return A

class Pair:
  def __init__(self, first, second):
    self.first = first
    self.second = second

def merge_intervals(v):
    result = [v[0]]
    for i in range(1,len(v)):
        if v[i].first <= result[-1].second:
            if v[i].second > result[-1].second:
                result[-1].second = v[i].second
        else:
            result.append(v[i])
    return result
def power(x, n):
    if 0 <= n < 3: return [1, x, x*x][n]
    r, m = n//2, n%2
    e = x
    while r:
       e **= 2
       r //= 2
    return e*x if m else e
# https://www.interviewbit.com/problems/power-of-two-integers/ n = A**P, A>0, P>1
def isPower(n: int):
    for p in range(2,33):
        a = round(n**(1/p))
        if a**p == n :
            return 1
    return 0
# A[i] becomes A[A[i]], 0 <= A[i] < len(A), with O(1) extra space
def arrange(A):
    n = len(A)  
    for i in range(n):
        A[i] += (A[A[i]]%n)*n
    for i in range(n):
        A[i] = A[i]//n

def arrange(A):
    for i,n in enumerate(A):
        j = i
        while 0 <= A[j] and A[j] != i and A[A[j]] >= 0:
            m = A[j]
            A[j], j = -A[m], m
        if A[j] == i:
            A[j] = -n
    for i,n in enumerate(A):
        A[i] = -n

def find_low_index(arr, key):
  r = len(arr)
  l = 0
  while l < r:
    m = (l+r)//2
    if key > arr[m]: l = m+1
    else: r = m
  return l if l < len(arr) and key == arr[l] else -1
def find_high_index(arr, key):
  r = len(arr)
  l = 0
  while l < r:
    m = (l+r)//2
    if key >= arr[m]: l = m+1
    else: r = m
  return r-1 if r > 0 and key == arr[r-1] else -1
def binaryConversion(n,m):
    xor = n^m
    cnt = 0
    while xor:
        if xor & 1 == 1: cnt += 1
        xor >>= 1
    return cnt
class Test(unittest.TestCase):
    test_cases = [
        ([1,10,20,0,59,63,0,88,0], [0,0,0,1,10,20,59,63,88]),
    ]
    test_functions = [move_zeros_to_left]
    def test_find_hilo_index(self):
        testcases = [([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6],1,[0,2])]
        for a,k,expected in testcases:
            print(expected)
            assert [find_low_index(a,-2), find_high_index(a,-2)] == [-1,-1]
            assert [find_low_index(a,7), find_high_index(a,7)] == [-1,-1]
            assert [find_low_index(a,k), find_high_index(a,k)] == expected
            assert [find_low_index(a,2), find_high_index(a,2)] == [3,7]
            assert [find_low_index(a,3), find_high_index(a,3)] == [8,10]
            assert [find_low_index(a,4), find_high_index(a,4)] == [11,14]
            assert [find_low_index(a,5), find_high_index(a,5)] == [15,17]
            assert [find_low_index(a,6), find_high_index(a,6)] == [18,23]
    def test_move_zeros_to_left(self):
        for text, expected in self.test_cases:
            assert move_zeros_to_left(text) == expected
    def test_merge_intervals(self):
        pairs = [Pair(1, 5), Pair(3, 1), Pair(4, 6), Pair(6, 8), Pair(10, 12), Pair(11, 15)]
        result = merge_intervals(pairs)
        for i in range(len(result)):
            print("[" + str(result[i].first) + ", " + str(result[i].second) + "]", end =" ")
    def test_power(self):
        r = power(3,9)
        print(r,3**9)
        assert r == 3**9
    def test_binaryConcersion(self):
        r = binaryConversion(29,15)
        print(r,29,15)
        assert r == 2
    def test_isPower(self):
        print(isPower(16808))
    def test_arrange(self):
        A = [1,2,3,0]
        A = [2, 1, 3, 0]
        arrange(A)
        assert A == [3, 1, 0, 2]
        print(A)

if __name__ == "__main__":
    unittest.main()

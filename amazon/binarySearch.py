import sys
from bisect import bisect_right as upper_bound 
  
class Solution:
    def __init__(self):
        self.MAX = 100
    def sqrt(self, A):
        if A < 2: return A
        ans = start = 0; end = A
        while start <= end:
            mid = start + (end - start)//2
            if mid <= A//mid:
                start = mid + 1
                ans = mid
            else:
                end = mid - 1
        return ans
    def findMedianSortedArrays(self, A, B):
        ABlen = len(A)+len(B)
        mid = ABlen//2
        prev = med = i = j = 0
        while i+j < mid and i < len(A) and j < len(B):
            if A[i] < B[j]:
                prev = A[i]
                i += 1
            elif A[i] > B[j]:
                prev = B[j]
                j += 1
            else:
                prev = A[i]
                i += 1; j += 1
        while i+j < mid and i < len(A):
            prev = A[i]; i += 1
        while i+j < mid and j < len(B):
            prev = B[j]; j += 1
        if ABlen%2:
            if len(A) and len(B):
                if i == len(A): return B[j]
                if j == len(B): return A[i]
                return min(A[i],B[j])
            if len(A): return A[i]
            if len(B): return B[j]
        elif i == len(A) and j == len(B): return (A[i-1]+B[j-1])/2
        elif i == len(A): return (prev+B[j])/2
        elif j == len(B): return (prev+A[i])/2
        else:
            return (prev+min(A[i],B[j]))/2
    def findMedianSortedArrays(self, A, B):
        if len(A) > len(B): return self.findMedianSortedArrays(B, A)
        x, y = len(A), len(B); lo, hi, mid = 0, x, (x+y+1)//2 # len(left side) >= len(right side)
        while lo <= hi:
            partX = (lo+hi)//2
            partY = mid - partX
            maxLeftX = A[partX-1] if partX else -sys.maxsize
            minRightX = A[partX] if partX < x else sys.maxsize
            maxLeftY = B[partY-1] if partY else -sys.maxsize
            minRightY = B[partY] if partY < y else sys.maxsize
            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                return max(maxLeftX,maxLeftY) if (x+y)%2 else (max(maxLeftX,maxLeftY) + min(minRightX,minRightY))/2.0
            elif maxLeftX > minRightY: hi = partX-1
            else: lo = partX+1
        raise ValueError("Arrays not sorted")
    def findMedianSortedArrays(self, nums1, nums2):
        a, b = sorted((nums1, nums2), key=len)
        m, n = len(a), len(b)
        after = (m + n - 1) // 2
    # find a pair (i, j) s.t. a[0:i] + b[0:j] forms the list of numbers 
    # smaller than or equal to the median
    # There could be multiple such pairs if there are duplicated numbers.
    # Each of such pairs satisfies the following criteria (conjunctively):
    # 1) i + j == after
    # 2) (j>=1 and a[i] >= b[j-1]) or j==0
    # 3) (i>=1 and b[j] >= a[i-1]) or i==0
    # note that at least one of cond1 and cond2 holds!
        lo, hi = 0, m
        while lo < hi:
            i = (lo + hi) // 2
            j = after - i
            cond1 = (j>=1 and a[i] >= b[j-1]) or j==0
            cond2 = (i>=1 and b[j] >= a[i-1]) or i==0
            if(cond1 and cond2):
                lo = i
                break
            elif(not cond1):
                lo = i + 1
            else:
                hi = i
        i = lo
        j = after - i
    # consider the 4 different cases
    # a = [1,2,5,6], b = [3,4,7,8]
    # a = [1,2,4,5], b = [3,6,7,8]
    # a = [1,2,3,6], b = [4,5,7,8]
    # a = [1,4,5], b = [2,3,6,7]
    # a = [1,5,6], b = [2,3,4,7]
        nextfew = sorted(a[i:i+2] + b[j:j+2])
        return (nextfew[0] + nextfew[1 - (m+n)%2]) / 2.0
    def matrixMedian(self, m): # return median value O(|max-min| * r * log(c))
        r, d = len(m), len(m[0])
        mi, mx = m[0][0], 0 # min/max values
        for _r in m: 
            if _r[0] < mi: mi = _r[0] # min of col 0
            if _r[d-1] > mx : mx =  _r[-1] # max of last col
        desired = (r * d + 1) // 2 # median position
        while (mi < mx): 
            mid = mi + (mx - mi) // 2 # median value
            mididx = 0 # mid position
            # Find count of elements smaller than mid 
            for _r in m: 
                 j = upper_bound(_r, mid) # count values on each row < mid
                 mididx = mididx + j 
            if mididx < desired: mi = mid + 1 # narrowing the mi/mx range
            else: mx = mid 
        # print("Median is", mi) 
        return mi
if __name__ == '__main__':
    sol = Solution()
    assert sol.matrixMedian([ [1, 3, 5], [2, 6, 9], [3, 6, 9]] ) == 5
    # assert sol.sqrt(11) == 3
    # assert sol.findMedianSortedArrays([0,23], []) == 11.5
    # assert sol.findMedianSortedArrays([1], [2,3]) == 2
    # assert sol.findMedianSortedArrays([2], [2]) == 2
    # assert sol.findMedianSortedArrays([1], [1]) == 1
    # assert sol.findMedianSortedArrays([], [1]) == 1
    # assert sol.findMedianSortedArrays([2], []) == 2
    # assert sol.findMedianSortedArrays([0, 0], [0, 0]) == 0
    # assert sol.findMedianSortedArrays([1, 3], [2]) == 2
    # assert sol.findMedianSortedArrays([1, 2], [3, 4]) == 2.5
    # assert sol.findMedianSortedArrays([1, 4, 5], [2, 3]) == 3
    # assert sol.findMedianSortedArrays([1, 4, 5], [2, 3, 6]) == 3.5
    # assert sol.findMedianSortedArrays([1], [2, 3, 4, 5, 6]) == 3.5

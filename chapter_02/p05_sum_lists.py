from chapter_02.linked_list import LinkedList, LinkedListNode as ListNode


def sum_lists(ll_a, ll_b):
    n1, n2 = ll_a.head, ll_b.head
    ll = NumericLinkedList()
    carry = 0
    while n1 or n2:
        result = carry
        if n1:
            result += n1.value
            n1 = n1.next
        if n2:
            result += n2.value
            n2 = n2.next

        ll.add(result % 10)
        carry = result // 10

    if carry:
        ll.add(carry)

    return ll

class Solution:
    def addTwoNumbers(self, l1, l2):
        len1, len2 = self.getLength(l1), self.getLength(l2)
        if not len1: return len2
        elif not len2: return len1
        elif len1 > len2: l2 = self.addLeadingZeros(len1-len2, l2)
        else: l1 = self.addLeadingZeros(len2-len1, l1)
        c, ans = self.combineList(l1, l2)
        if c>0:
            l3 = ListNode(c)
            l3.next = ans
            ans = l3
        return ans
    def getLength(self, node):
        l = 0
        while node:
            l += 1
            node = node.next
        return l
    def addLeadingZeros(self, n, node):
        for i in range(n):
            new = ListNode(0)
            new.next = node
            node = new
        return node
    def combineList(self, l1, l2):
        if not l1: return (0, None) # l1 and l2 have same length
        c, new = self.combineList(l1.next, l2.next)
        c, m = divmod(l1.value + l2.value + c, 10)
        return (c, ListNode(m, new))
    # https://leetcode.com/problems/insertion-sort-list/
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head: return head
        cur = head; nex = head.next
        while nex:
            if cur.value > nex.value:
                pp = p = head
                if p.value >= nex.value: cur.next = nex.next; nex.next = p; head = nex
                else:
                    p = pp.next
                    while p.value < nex.value: pp = p; p = p.next
                    pp.next = nex; cur.next = nex.next; nex.next = p
            else: cur = nex
            nex = cur.next
        return head
    # https://leetcode.com/problems/merge-two-sorted-lists/
    def merge(self, h1, h2):
        dummy = tail = ListNode(None)
        while h1 and h2:
            if h1.value < h2.value: # move lesser node to dummy list
                tail.next, h1 = h1, h1.next
            else:
                tail.next, h2 = h2, h2.next
            tail = tail.next
        tail.next = h1 or h2 # point to remaining of longer list
        return dummy.next # head of merged list
    # https://leetcode.com/problems/sort-list/discuss/46710/Clean-python-code
    def sortList(self, head): # recursive merge, Time logN, Space N due to call stack
        if not head or not head.next:
            return head
        pre, slow, fast = None, head, head
        while fast and fast.next: # split
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None # break current linked list into two equal lengths linked lists when we run a merge
        return self.merge(*map(self.sortList, (head, slow))) # recursively sort on head/slow halves
    # https://leetcode.com/problems/sort-list/discuss/46712/Bottom-to-up(not-recurring)-with-o(1)-space-complextity-and-o(nlgn)-time-complextity
    def sortList(self, head: ListNode) -> ListNode:
        def merge(tail: ListNode, l1: ListNode, l2: ListNode) -> ListNode:
            while l1 or l2:
                if not l2 or l1 and l1.value < l2.value:
                    tail.next = l1; l1 = l1.next
                else:
                    tail.next = l2; l2 = l2.next
                tail = tail.next
            return tail
        def split(node: ListNode, size: int) -> ListNode: # node starts a new list of size
            prev, curr = None, node
            while size > 0 and curr:
                prev, curr = curr, curr.next
                size -= 1
            if prev: prev.next = None # cut at list end: prev
            return curr # left-over list

        if not head or not head.next: return head
        curr, size, interval = head, 0, 1
        while curr:
            size += 1
            curr = curr.next
        dummy = ListNode(0, head)
        while interval < size: # repeat for intervals 2**0.. == 1, 2, 4, 8, 16
            tail, curr = dummy, dummy.next
            while curr: # walk cur (remaining unsorted list) to split it into sublists of size 1..
                left = curr
                right = split(left, interval)
                curr = split(right, interval)
                tail = merge(tail, left, right) # merge left/right to dummy list, return tail for subsequent merges
            interval *= 2
        return dummy.next
    # https://leetcode.com/problems/reverse-linked-list-ii/submissions/
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head or not head.next or m == n: return head
        pre = dummy = ListNode(None,next_node=head)
        for _ in range(m-1): pre = pre.next
        cur = mp = pre.next
        for _ in range(n-m):
            nex = cur.next
            cur.next = pre
            pre = cur
            cur = nex
        mp.next.next = cur
        mp.next = cur.next
        cur.next = pre
        return dummy.next
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            nex = cur.next
            cur.next = pre
            pre = cur
            cur = nex
        return pre
        # last = head # recursive, last becomes new head
        # if head:
        #     if head.next:
        #         last = self.reverseList(head.next)
        #         head.next.next = head
        #         head.next = None
        # return last
        # if not head or not head.next: return head
        # node = self.reverseList(head.next)
        # head.next.next = head
        # head.next = None
        # return node
    def reverseAltKnodes(self, head: ListNode, k):
        if not head or not head.next or k<2: return head
        def skipKNodes(cur, k):
            pre = None
            while k and cur: k = k - 1; pre, cur = cur, cur.next
            return pre,cur
        def reverseKNodes(pre, cur, k):
            mp = cur
            while k and cur:
                k = k - 1
                nex = cur.next
                cur.next = pre
                pre = cur
                cur = nex
            if k:
                return pre,cur
            mp.next.next = pre
            mp.next = cur
            return pre,cur
        pre = dummy = ListNode(None,next_node=head); cur = pre.next
        while cur:
            pre, cur = reverseKNodes(pre, cur, k)
            if cur: pre, cur = skipKNodes(cur, k)
        return dummy.next
    # https://leetcode.com/problems/reverse-nodes-in-k-group/
    def reverseListKgroup(self, A, k):
        def reverseKNodes(pre, cur, k):
            mp = cur
            while k and cur:
                k = k - 1
                nex = cur.next
                cur.next = pre
                pre = cur
                cur = nex
            if k:
                return pre,cur
            mp.next.next = pre
            mp.next = cur
            return mp,cur
        if not A or not A.next or k<2: return A
        cur = A; count = 0
        while cur: cur = cur.next; count += 1
        if count < k: return A
        cur = A; kN = count//k
        pre = dummy = ListNode(None,next_node=A); cur = pre.next
        for _ in range(kN):
            pre, cur = reverseKNodes(pre, cur, k)
        return dummy.next
    # https://www.techiedelight.com/reverse-alternate-group-k-nodes-linked-list/
    def reverseAltKnodes(self, head, k):
        def reverse(curr, k):
            prev = None
            while curr and k > 0:
                k = k - 1
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            return prev, curr
        def skipKNodes(curr, k):
            prev = None
            while curr and k > 0:
                k = k - 1
                prev = curr
                curr = curr.next
            return prev, curr
        prev = None; curr = head
        while curr:
            last = curr
            front, curr = reverse(curr, k)
            if prev is None:
                head = front
            else:
                prev.next = front
            last.next = curr
            prev, curr = skipKNodes(curr, k)
        return head
    def reorderList(self, A):
        dummy = ListNode(None,A); half = fast = A; count = 1
        while fast:
            if count%2: half = half.next
            fast = fast.next
        return dummy.next
    def rotateRight(self, A, k):
        if not A or not A.next: return A
        tail = cur = A; count = 0
        while cur: tail, cur = cur, cur.next; count += 1
        if count <= k: k = k%count
        if not k: return A
        skip = count - k
        pre = cur = A
        while skip:
            pre, cur = cur, cur.next
            skip -= 1
        pre.next = None; tail.next = A; A = cur
        return A
    def removeNthFromEnd(self, A, k):
        if not A or not A.next:
            return A.next if A and k == 1 else A
        tail = cur = A; count = 0
        while cur: tail, cur = cur, cur.next; count += 1
        skip = count - k
        pre = dummy = ListNode(None,A); cur = A
        while skip:
            pre, cur = cur, cur.next
            skip -= 1
        pre.next = cur.next
        return dummy.next
    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for i in range(n):
            fast = fast.next
            if not fast: return head.next
        while fast.next:
            fast = fast.next; slow = slow.next
        slow.next = slow.next.next
        return head
    # https://leetcode.com/problems/linked-list-cycle-ii/discuss/44793/O(n)-solution-by-using-two-pointers-without-change-anything
    # https://leetcode.com/problems/linked-list-cycle-ii/discuss/44783/Share-my-python-solution-with-detailed-explanation
    # x1 distance from head to the start of the loop
    # x2 distance from the start of the loop to the point fast and slow meet
    # x3 distance from the point fast and slow meet to the start of the loop
    # x1 + x2 + x3 + x2 distance fast moved
    # x1 + x2 distance slow moved
    # x1 + x2 + x3 + x2 = 2 (x1 + x2) => x1 = x3
    def detectCycle(self, head):
        try:
            fast = head.next
            slow = head
            while fast is not slow:
                slow = slow.next; fast = fast.next.next
        except: # reached the end, no cycle
            return None 
        slow = slow.next # since fast starts at head.next, we need to move slow one step forward
        while head is not slow:
            head = head.next
            slow = slow.next
        return head
    # https://medium.com/@tuvo1106/the-tortoise-and-the-hare-floyds-algorithm-87badf5f7d41
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next: return None
        slow = fast = head
        while fast and fast.next:
            slow = slow.next; fast = fast.next.next
            if slow == fast:
                slow = head # x1 = x3, distance from head to the start of the loop = distance from the point fast and slow meet to the start of the loop
                while slow != fast:
                    slow = slow.next; fast = fast.next
                return fast
        return None
    # https://leetcode.com/problems/middle-of-the-linked-list/submissions/
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next; fast = fast.next.next
        return slow
    def middleNodeKleft(self, head: ListNode, k: int) -> ListNode:
        slow = fast = head; mid = 1
        while fast and fast.next:
            slow = slow.next; fast = fast.next.next
            mid += 1
        if k >= mid: return -1
        fast = head
        for _ in range(mid-k-1): fast = fast.next
        return fast.value
    # https://leetcode.com/problems/remove-duplicates-from-sorted-list/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        cur = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else: cur = cur.next
        return head
    # https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/discuss/28336/Python-in-place-solution-with-dummy-head-node.
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        pre = dummy = ListNode(None, head); cur = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                while cur and cur.next and cur.val == cur.next.val:
                    cur = cur.next
                cur = cur.next
                pre.next = cur
            else:
                pre, cur = pre.next, cur.next
        return dummy.next
    # https://leetcode.com/problems/swap-nodes-in-pairs/submissions/
    def swapPairs(self, head: ListNode) -> ListNode:
        # pre = dummy = ListNode(None, head); cur = head
        # while cur and cur.next:
        #     nex = cur.next.next
        #     cur.next.next = cur
        #     pre.next = cur.next
        #     cur.next = nex
        #     pre, cur = cur, nex
        # return dummy.next
        pre, pre.next = self, head
        while pre.next and pre.next.next: # a/b exist
            a = pre.next; b = a.next # a/b cur/next nodes
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return self.next
    # https://leetcode.com/problems/swapping-nodes-in-a-linked-list/submissions/
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        self.next = ak = zk = head        
        for _ in range(k-1): ak = ak.next # kth node from head
        fast = ak
        while fast and fast.next: fast = fast.next; zk = zk.next # kth node from end
        ak.val, zk.val = zk.val, ak.val
        return self.next
    # https://www.geeksforgeeks.org/reverse-even-elements-in-a-linked-list/
    def evenReverse(self, head, pre = None):
        if not head:
            return head
        cur = head
        while cur and cur.value%2 == 0:
            nex = cur.next; cur.next = pre
            pre = cur; cur = nex
        if cur != head:
            head.next = cur; cur = self.evenReverse(cur)
            return pre
        else:
            head.next = self.evenReverse(head.next, head)
            return head
    def mix(self, h1, h2):
        a = h1; b = h2
        while a and b:
            nex = a.next; a.next = b; a = nex; a, b = b, a
        return h1
    # https://www.interviewbit.com/problems/even-reverse/
    def evenReverse(self, head):
        if not head or not head.next or not head.next.next or not head.next.next.next: return head 
        ohead = op = head; ehead = ep = head.next; cur = ehead.next
        while cur:
            op.next = cur; op, cur = cur, cur.next
            if cur:
                ep.next = cur; ep, cur = cur, cur.next
        op.next = ep.next = None
        ehead = self.reverseList(ehead)
        return self.mix(ohead, ehead)
    # https://leetcode.com/problems/copy-list-with-random-pointer/
    def copyRandomList(self, head: 'Node') -> 'Node':
        cur = head; cphead = cpcur = ListNode(cur.value); i = 0
        nodeIdx = {cur:0}; i2cp = {0:cpcur}
        while cur.next:
            cur = cur.next; i += 1; nodeIdx[cur] = i;
            i2cp[i] = cpcur.next = ListNode(cur.value); cpcur = cpcur.next
        cur = head; cpcur = cphead
        while cur:
            if cur.random:
                cpcur.random = i2cp[nodeIdx[cur.random]]
            cur = cur.next; cpcur = cpcur.next
        return cphead
class NumericLinkedList(LinkedList):
    @classmethod
    def generate_from_integer(cls, integer):
        integer_parts = [int(c) for c in str(integer)]
        integer_parts.reverse()
        return cls(integer_parts)

    def numeric_value(self):
        number = 0
        for place, node in enumerate(self):
            number += node.value * 10 ** place
        return number

test_cases = (
    # all values can either be list of integer or integers
    # a, b, expected_sum
    ([7, 1, 6], [5, 9, 2], [2, 1, 9]),
    (0, 0, 0),
    ([], [], 0),
    ([3, 2, 1], [3, 2, 1], [6, 4, 2]),
    (123, 123, 246),
    (123, 1, 124),
    (1, 123, 124),
)

testable_functions = (
    sum_lists,
    # sum_lists_followup
)

def test_numeric_linked_list():
    ll = NumericLinkedList.generate_from_integer(321)
    assert ll.numeric_value() == 321


def test_linked_list_addition():
    for f in testable_functions:
        for a, b, expected in test_cases:
            print(f"{f.__name__}: {a}, {b}, {expected}")
            if isinstance(a, int):
                ll_a = NumericLinkedList.generate_from_integer(a)
            else:
                ll_a = NumericLinkedList(a.copy())

            if isinstance(b, int):
                ll_b = NumericLinkedList.generate_from_integer(b)
            else:
                ll_b = NumericLinkedList(b.copy())
            result = f(ll_a, ll_b)
            if isinstance(expected, int):
                assert result.numeric_value() == expected
            else:
                assert result.values() == expected

if __name__ == "__main__":
    ll_a = LinkedList.generate(1, 0, 9)
    ll_b = LinkedList.generate(3, 0, 2)
    # print(ll_a)
    # print(ll_b)
    # print(sum_lists(ll_a, ll_b))
    sol = Solution()
    # ll = LinkedList([1, 2, 3, 3, 8, 6, 4, 5])
    # ll = LinkedList([1, 2, 3, 4, 5, 6, 7, 8, 9,10])
    # ll = LinkedList([7, 13, 11, 10, 1])
    ll = LinkedList([3,3,3])
    ll.add_randoms([[3,None],[3,0],[3,None]])
    # ll.head = sol.reverseBetween(ll_a.head, 2, 4)
    # ll.head = sol.reverseList(ll_a.head)
    # print(ll_a.head)
    print(ll)
    print(sol.copyRandomList(ll.head))
    # print(sol.middleNodeKleft(ll_a.head, 1))
    # ll.head = sol.evenReverse(ll.head)
    # ll.head = sol.removeNthFromEnd(ll_a.head, 2)
    # ll.head = sol.rotateRight(ll_a.head, 1)
    # ll.head = sol.reverseList(ll_a.head, 2)
    # ll.head = sol.reorderList(ll_a.head)
    # ll.head = sol.reverseAltKnodes(ll_a.head, 2)
    # print(ll)
    # ll.head = sol.sortList(ll_a.head)
    # ll.head = sol.addTwoNumbers(ll_a.head, ll_b.head)
    # print(ll)
    # ll.head = sol.insertionSortList(ll_b.head)
    # print(ll)
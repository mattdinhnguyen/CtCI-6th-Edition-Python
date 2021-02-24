from typing import List, Optional, Deque, Set
import sys
from collections import defaultdict, deque
from heapq import heappush, heappop
from itertools import groupby
class Node:
    def __init__(self, val):
        self.val = val
        self.parent = None
        self.left = None
        self.right = None

class WordDictionary:
    def __init__(self):
        self.words = dict(chrCnt = defaultdict(int))
    def insert(self, word: str) -> None:
        node = self.words
        for chr in word:
            node["chrCnt"][chr] += 1
            node = node.setdefault(chr,dict(chrCnt = defaultdict(int)))
        node['$'] = word
    def findPrefix(self, word: str) -> str:
        nodes = [self.words]; pre = ""
        for char in word:
            # nodes = [kid for node in nodes for kid in ([node[char]] if char in node else [])]
            next= []
            for node in nodes:
                if char in node:
                    pre += char
                    if node["chrCnt"][char] > 1:
                        next.extend([kid for kid in [node[char]]])
                    else: break
            nodes = next
        return pre

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        new = Node(val)
        if self.root is None:
            self.root = new
            return

        current = self.root
        while current:
            if current.val > val:
                if current.left is None:
                    current.left = new
                    new.parent = current
                    return
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = new
                    new.parent = current
                    return
                else:
                    current = current.right

    def get_node(self, val):
        current = self.root
        while current:
            if current.val == val:
                return current
            elif current.val > val:
                current = current.left
            else:
                current = current.right
        raise Exception("No such value in the tree")
class Solution:
    # Traversal BST to find pair for each node
    def findTarget(self, root, k):
        # Skip the current node calling this function in case its value is counted (e.g., current value = 2, try to find 2).
        def findNode(root, current, value):
            if not root or root == current: return False
            # if root.val == value: return True
            # if value < root.val:
            #     return findNode(root.left, current, value)
            # else:
            #     return findNode(root.right, current, value)
            return root.val == value or \
                (findNode(root.left, current, value) if value < root.val else findNode(root.right, current, value))
        def _findTarget(root, current, target):
            if not root or not current: return False
            return findNode(root, current, target - current.val) or \
                _findTarget(root, current.left, target) or \
                _findTarget(root, current.right, target)
        return _findTarget(root, root, k)
    # set
    def findTarget(self, root, k):
        def _findTarget(root, target, visited):
            if not root: return False
            if (target - root.val) in visited: return True
            visited.add(root.val) # mark node 'visited' after checking: target = 4, root.val = 2. If we mark 2 as visited, then we will find 4 - 2 = 2 immediately.
            return _findTarget(root.left, target, visited) or \
                _findTarget(root.right, target, visited)
        return _findTarget(root, k, set()) # Preorder traversal all nodes and search the complement visited before
    # https://leetcode.com/problems/validate-binary-search-tree/submissions/
    def isValidBST(self, root, min = -sys.maxsize, max = sys.maxsize):
        ans = True
        if root:
            if max <= root.val or root.val <= min: ans = False
            elif root.left:
                ans = self.isValidBST(root.left, min, root.val) if root.left.val < root.val else False
            if ans and root.right:
                ans = self.isValidBST(root.right, root.val, max) if root.right.val > root.val else False
        return ans
    def isValidBST(self, A):
        if not A: return 0
        node = A; stack = []; minimum = -float('inf')
        while node or stack:
            while node:
                stack.append(node)
                node = node.left # inorder traversal
            node = stack.pop()
            if node.val <= minimum: return False
            minimum = node.val # set to val of inorder discovered nodes
            node = node.right
        return True
    #
    def minDiffInBST(self, root):
        minDist = sys.maxsize
        if root:
            node = root; st = []; minimum = sys.maxsize
            while node or st:
                while node:
                    st.append(node)
                    node = node.left # inorder traversal
                node = st.pop()
                minDist = min(minDist, abs(node.val-minimum))
                minimum = node.val # set to val of inorder discovered nodes
                node = node.right
        return minDist
    # https://leetcode.com/problems/recover-binary-search-tree/discuss/32535/No-Fancy-Algorithm-just-Simple-and-Powerful-In-Order-Traversal
    def recoverTree(self, root): # inorder, recursive
        first = second = None; prev = Node(-sys.maxsize)
        def inorder(root):
            nonlocal first, second, prev
            if not root: return
            inorder(root.left)
            if prev.val >= root.val:
                if not first: first = prev # first element has not been found
                second = root
            prev = root
            inorder(root.right)
        inorder(root) # In order traversal to find the two elements
        if first and second:
            first.val, second.val = second.val, first.val
        return root
   # https://github.com/royalpranjal/Interview-Bit/blob/master/Trees/RecoverBinarySearchTree.cpp
    def recoverTree(self, root): # inorder iterative
        first = second = None; prev = Node(-sys.maxsize); curr = root; st = []; done = False
        while not done:
            if curr: st.append(curr); curr = curr.left # go left till reaching leaf.left == None
            else: # node has no left child
                curr = st.pop()
                if prev.val > curr.val: # set prev, first, second at the supposedly ordered values, but not
                    if not first: first = prev
                    second = curr
                prev = curr
                curr = curr.right
            if not st and not curr: done = True
        return [first.val,second.val] if first.val < second.val else [second.val,first.val]
    # https://leetcode.com/problems/recover-binary-search-tree/discuss/32559/Detail-Explain-about-How-Morris-Traversal-Finds-two-Incorrect-Pointer
    def recoverTree(self, root): # Morris Traversal
        cur = root; pre = temp = first = second = None
        while cur:
            if cur.left:
                temp = cur.left
                while temp.right and temp.right != cur:
                    temp = temp.right # find the right-most leaf of cur.left
                if temp.right == cur: # threading already exists
                    # res.append(cur.val)
                    if pre and pre.val > cur.val:
                        if not first: first = pre
                        second = cur
                    pre = cur
                    cur = cur.right
                    temp.right = None
                else: # right-most leaf points to cur
                    temp.right = cur
                    cur = cur.left
            else:
                # res.append(cur.val)
                if pre and pre.val > cur.val:
                    if not first: first = pre
                    second = cur
                pre = cur
                cur = cur.right
        # if first and second:
        #     first.val, second.val = second.val, first.val
        # return root
        return sorted([first.val,second.val])
    # https://leetcode.com/problems/binary-tree-postorder-traversal/submissions/
    def postorderTraversal(self, root: Node) -> List[int]:
        ans = []
        if root:
            ans += self.postorderTraversal(root.left)
            ans += self.postorderTraversal(root.right)
            ans.append(root.val)
        return ans
    # https://leetcode.com/problems/binary-tree-postorder-traversal/discuss/45551/Preorder-Inorder-and-Postorder-Iteratively-Summarization ofLucas
    def postorderTraversal(self, root: Node) -> List[int]:
        if not root: return []
        ans = []; st = [root]; p = root
        while st:
            prev = p; p = st[-1]
            if p.left and p.left != prev and p.right != prev: st.append(p.left)
            else:
                if p.right and p.right != prev: st.append(p.right)
                else: ans.append(st.pop().val)
        return ans
    def postorderTraversal(self, root: Node) -> List[int]:
        ans = []; st = []; cur = root
        while cur or st:
            while cur and (cur.left or cur.right): # not leaf
                st.append(cur); cur = cur.left
            if cur: ans.append(cur.val) # leaf
            while st and cur == st[-1].right: # pop st till top.right is not cur
                cur = st.pop(); ans.add(cur.val)
            cur = st[-1].right if st else None
        return ans
    # FrankChu_0229
    def postorderTraversal(self, root: Node) -> List[int]:
        if not root: return []
        stack = []; res = deque()
        while stack or root:
            while root:
                stack.append(root)
                res.appendleft(root.val) # build res from tail to head: root then right
                root = root.right
            root = stack.pop(); root = root.left # left nodes will be pushed to head of res
        return res
    # xiaoyuz666
    def postorderTraversalT(self, root: Node) -> List[int]:
        if not root: return []
        list = deque(); stack = [root]
        while stack: 
            curr = stack.pop(); list.appendleft(curr.val) # O(N)
            if curr.left: stack.append(curr.left)
            if curr.right: stack.append(curr.right)
        return list
    # sevenhe716
    def postorderTraversal(self, root: Node) -> List[int]:
        if not root: return []
        stack, res = [root], []
        last = None # mark whether left or right child has been visited
        while stack:
            root = stack[-1]
            if root.left == root.right == None or last and (root.left == last or root.right == last):
                res.append(root.val) # if leaf, or left or right child has been visited, then process and pop it
                stack.pop() # pop root
                last = root
            else: # push right and left child in stack
                if root.right: stack.append(root.right) # push right first because of FILO
                if root.left: stack.append(root.left)
        return res
    # jocelynayoga
    def postorderTraversal(self, root: Node) -> List[int]:
        if not root: return []
        list = []; stack = []; cur = root
        while cur or stack:
            if cur: stack.append(cur); cur = cur.left # visit left first
            else:
                node = stack[-1].right # top has unvisited right child?
                if node: cur = node # yes, push into stack, then visit left descendants
                else: # no right child, pop leaf
                    node = stack.pop()
                    list.append(node.val)
                    while stack and stack[-1].right == node:
                        node = stack.pop(); list.append(node.val) # pop all tops having right descendants
        return list
    # root_010
    def postorderTraversal(self, root: Node) -> List[int]:
        stack = []; list = []; pre = None; current = root
        while current or stack:
            while current:
                stack.append(current); current = current.left
            current = stack[-1]
            if current.right and pre != current.right: current = current.right
            else:
                stack.pop()
                list.add(current.val)
                pre = current; current = None
            # current = stack.pop()
            # if current.right and pre != current.right:
            #     stack.append(current); current = current.right; continue
            # list.add(current.val)
            # pre = current; current = None
        return list
    # mmao3
    def postorderTraversal(self, root: Node) -> List[int]:
        stack = []; res = []
        while stack or root:
            while root: stack.append(root); root = root.left or root.right # 
            node = stack.pop(); res.add(node.val) # pop leaf
            if stack and stack[-1].left == node: root = stack[-1].right
        return res
    #
    def verticalTraversal(self, root: Node) -> List[List[int]]:
        nodeMat = [[0,0,root.val,root]] # node matrix has root at x,y coordinates (0,0)
        curLevel = nodeMat; ans = []
        while curLevel:
            nextLevel = []
            for x,y,v,n in curLevel:
                if n.left: heappush(nextLevel,[x-1,y+1,n.left.val,n.left])
                if n.right: heappush(nextLevel,[x+1,y+1,n.right.val,n.right])
            for t in nextLevel: heappush(nodeMat,t)
            curLevel = nextLevel
        g = 0; l = [] # vertical group
        for x,y,v,n in nodeMat:
            if t[0] > g:
                g = t[0]
                if l: ans.append(l)
                l = [v]
            else:
                l.append(v)
        return ans
        return res
    # https://leetcode.com/problems/binary-tree-inorder-traversal/
    def inorderTraversal(self, root: Node) -> List[int]: # Morris Traversal
        res = []; cur = root; prev = None
        while cur:
            if cur.left:
                prev = cur.left
                while prev.right and prev.right != cur:
                    prev = prev.right # find the right-most leaf of cur.left
                if prev.right == cur: # threading already exists
                    res.append(cur.val)
                    cur = cur.right
                    prev.right = None
                else: # right-most leaf points to cur
                    prev.right = cur
                    cur = cur.left
            else:
                res.append(cur.val)
                cur = cur.right
        return res
    # https://www.cnblogs.com/AnnieKim/archive/2013/06/15/morristraversal.html
    def preorderTraversal(self, root: Node):
        res = []; st = []
        def findLeaf(node) -> None:
            while node:
                st.append(node)
                node = node.left if node.left else node.right
        def getNext():
            node = st.pop()
            if len(st) > 0 and st[-1].left == node:
                findLeaf(st[-1].right)
            return node.val
        findLeaf(root)
        while len(st) > 0: res.append(getNext())
        return res
    def preorderTraversal(self, root: Node) -> List[int]: # Morris Traversal Time O(n), Space O(1)
        res = []; cur = root; prev = None
        while cur:
            if cur.left:
                prev = cur.left
                while prev.right and prev.right != cur: #  Each of n-1 edges in BST is walked up to 2 times: 1 is to locate a node, and 2 is to find the predecessor node
                    prev = prev.right # find the right-most leaf of cur.left
                if prev.right == cur: # threading already exists
                    cur = cur.right
                    prev.right = None
                else: # right-most leaf points to cur
                    res.append(cur.val)
                    prev.right = cur
                    cur = cur.left
            else:
                res.append(cur.val)
                cur = cur.right
        return res
    # preorder traversal to fill the map with diagonal elements https://www.techiedelight.com/print-diagonal-traversal-binary-tree/
    def diagonalTraversal(self, root: Node) -> List[List[int]]:
        def traversal(node, diagonal, diagMap):
            if not node: return
            diagMap[diagonal].append(node.val)
            traversal(node.left, diagonal+1, diagMap)
            traversal(node.right, diagonal, diagMap)
 
        diagonalMap = defaultdict(list) # store diagonal elements in every slope
        traversal(root, 0, diagonalMap)
        res = []
        for i in range(len(diagonalMap)):
            res.append(diagonalMap.get(i))
        return res
    def diagonalTraversal(self, root: Node) -> List[List[int]]:
        q = deque(); res = []; diag = []
        sentinel = Node(-1) # dummy node to denote the end of a diagonal
        while root: # # enqueue all nodes on the first diagonal
            q.append(root)
            root = root.right
        q.append(sentinel) # enqueue sentinel node at the end of each diagonal
        while len(q) != 1: # run till only sentinel is left
            front = q.popleft() # dequeue front node
            if front != sentinel:
                diag.append(front.val) # build 1 diag at a time
                node = front.left # for each front on this diag, enqueue the next diag
                while node:
                    q.append(node)
                    node = node.right
            else: # current diagonal end is reached, enqueue the sentinel node for next diag
                q.append(sentinel)
                # res.append(diag); diag = []
                res.extend(diag); diag = []
        # res.append(diag)
        res.extend(diag)
        return res
    # https://www.geeksforgeeks.org/maximum-edge-removal-tree-make-even-forest/
    # Return max edge to be removed for even Connected Components (evenCC)
    def maxEdgeRemovalToMakeBTEven(self, root: Node) -> int:
        evenCC = 0
        def dfs(n): # node, count, return subtree node count
            nonlocal evenCC
            c = 0
            for kid in (n.left,n.right):
                if kid: c += dfs(kid)
            c = c+1
            if c%2 == 0: evenCC += 1; c = 0
            return c
        dfs(root)
        return evenCC
    def maxEdgeRemovalToMakeForestEven(self, A: int, B: List[List[int]]) -> int: # https://www.interviewbit.com/problems/maximum-edge-removal/
        g = defaultdict(list); res = 0
        def addEdge(u: int, v: int) -> None:
            nonlocal g
            g[u].append(v)
            g[v].append(u)
        def dfs(u, visited):
            nonlocal res, g
            visited.add(u)
            currComponentNode = 0
            for v in g[u]: # traverse u's neighbors
                if v not in visited:
                    subtreeNodeCount = dfs(v, visited) # Count the number of nodes in a subtree
                    if subtreeNodeCount % 2 == 0: res += 1
                    else: currComponentNode += subtreeNodeCount
            return currComponentNode + 1 # Number of nodes in current component and one for current node
        def maxEdgeRemovalToMakeForestEven(N):
            visited = set()
            dfs(1, visited) # starting with root node 1
            return res
        for e in B: addEdge(*e)
        return maxEdgeRemovalToMakeForestEven(len(B))
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
    def flatten(self, root: Node) -> None:
        cur = root; prev = None
        while cur:
            if cur.left:
                prev = cur.left
                while prev.right: prev = prev.right # find the right-most leaf of cur.left
                prev.right = cur.right # right-most leaf points to cur.right
                cur.right = cur.left; cur.left = None
                if prev.right: self.flatten(prev.right)
            else: cur = cur.right
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/36977/My-short-post-order-traversal-Java-solution-for-share
    def flatten(self, root: Node) -> None:
        def flatten(root, pre):
            if not root: return pre
            pre = flatten(root.right, pre)
            pre = flatten(root.left, pre)
            root.right = pre
            root.left= None
            pre = root
            return pre
        flatten(root, None)
    # https://leetcode.com/problems/path-sum/submissions/
    def hasPathSum(self, root, sum): # recursive N
        if not root: return False # this path has not sum
        if not root.left and not root.right and root.val == sum: return True  # found leaf of path with sum
        sum -= root.val
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum) # branch to all leaves
    def hasPathSum(self, root, sum): # iterative, N
        if not root: return False
        stack = [(root, root.val)]
        while stack:
            curr, val = stack.pop()
            if not curr.left and not curr.right and val == sum: return True
            if curr.right:
                stack.append((curr.right, val+curr.right.val))
            if curr.left:
                stack.append((curr.left, val+curr.left.val))
        return False
    # https://leetcode.com/problems/maximum-depth-of-binary-tree/submissions/
    def maxDepth(self, root: Node) -> int:
        if not root: return 0
        def h(node,level):
            # print(f"node {node} level {level}")
            cs = [c for c in [node.left,node.right] if c]
            return max([h(c, level+1) for c in cs]) if cs else level
        return h(root,1)
    def maxDepth(self, root: Node) -> int:
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0
    def maxDepth(self, root: Node) -> int:
        depth = 0; level = [root] if root else []
        while level:
            depth += 1; queue = []
            for el in level:
                if el.left:
                    queue.append(el.left)
                if el.right:
                    queue.append(el.right)
            level = queue
        return depth
    # https://leetcode.com/problems/minimum-depth-of-binary-tree/submissions/
    def minDepth(self, root) -> int:
        if not root: return 0
        h = [self.minDepth(root.left), self.minDepth(root.right)]
        return 1 + (min(h) or max(h)) # 644 ms
    def minDepth(self, root) -> int:
        if not root: return 0
        d, D = sorted(map(self.minDepth, (root.left, root.right)))
        return 1 + (d or D)
    def minDepth(self, root: Node) -> int:
        l, i = [root], 1
        while l and root and all(n.left or n.right for n in l):
            l, i = [kid for n in l for kid in [n.left, n.right] if kid], i+1
        return i if root else 0
    def minDepth(self, root: Node) -> int:
        def h(node,level):
            kids = [c for c in [node.left,node.right] if c]
            return min([h(c, level+1) for c in kids]) if kids else level
        return h(root,1) if root else 0
    # https://leetcode.com/problems/balanced-binary-tree/submissions/
    def isBalanced(self, root: Node) -> bool:
        def check(node): # recursive
            if not node: return 0
            l, r = check(node.left), check(node.right)
            if l == -1 or r == -1 or abs(l-r) > 1: return -1
            return 1 + max(l,r)
        return check(root) != -1
    def isBalanced(self, root: Node) -> bool:
        stack, node, last, depths = [], root, None, {} # postorder traversal
        while stack or node:
            if node: stack.append(node); node = node.left # go to leftmost leaf
            else:
                node = stack[-1]
                if not node.right or last == node.right: # top is a leaf or last was its right
                    node = stack.pop()
                    left, right  = depths.get(node.left, 0), depths.get(node.right, 0)
                    if abs(left - right) > 1: return False
                    depths[node] = 1 + max(left, right)
                    last = node
                    node = None
                else: node = node.right # go right, then left
        return True
    #
    def isSymmetric(self, root: Node) -> bool:
        def helper(left, right):
            if not left or not right: return left == right
            if left.val != right.val: return 0
            return helper(left.left,right.right) and helper(left.right,right.left)
        return not root or helper(root.left, root.right)
    def isSymmetric(self, root: Node) -> bool:
        if root:
            st = [[root.left,root.right]]
            while st:
                l, r = st.pop()
                if not l and not r: continue
                if not l or not r: return l == r
                if l.val != r.val: return False
                st.extend([[l.left,r.right],[l.right,r.left]])
        return True
    def isSameTree(self, A, B):
        if not A or not B: return int(A == B)
        if A.val != B.val: return 0
        return self.isSameTree(A.left, B.left) and self.isSameTree(A.right, B.right)
    def isSameTree(self, A, B):
        st = [[A,B]]
        while st:
            a, b = st.pop()
            if not a and not b: continue
            if not a or not b: return int(a == b)
            if a.val != b.val: return 0
            st.extend([[a.left,b.left],[a.right,b.right]])
        return 1
    # https://leetcode.com/problems/merge-two-binary-trees/submissions/
    def mergeTrees(self, A: Node, B: Node) -> Node:
        if not A or not B: return A or B
        A.val += B.val
        A.left = self.mergeTrees(A.left, B.left)
        A.right = self.mergeTrees(A.right, B.right)           
        return A
    def mergeTrees(self, t1: Node, t2: Node) -> Node: # https://www.interviewbit.com/problems/merge-two-binary-tree/
        if not t1 or not t2: return t1 or t2
        s = [(t1, t2)]
        while s: 
            n1, n2 = s.pop()
            if not n2: continue #nothing to add on
            n1.val += n2.val
            if not n1.right: n1.right = n2.right
            elif n2.right: s.append((n1.right, n2.right))
            if not n1.left: n1.left = n2.left
            elif n2.left: s.append((n1.left, n2.left))
        return t1
    # https://leetcode.com/problems/queue-reconstruction-by-height/
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        ans = [[] for _ in range(len(people))]
        for h,f in sorted(people):
            ef = -1 # empty fronts
            for l in ans:
                if not l or l[0] >= h:
                    ef += 1
                if ef == f:
                    l.extend([h,f])
                    break
        return ans
    # https://leetcode.com/problems/queue-reconstruction-by-height/discuss/89345/Easy-concept-with-PythonC%2B%2BJava-Solution
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        res = []
        for k, g in groupby(sorted(people, reverse=True), key=lambda x: x[0]):
            for person in sorted(g):
                res.insert(person[1], person)
        return res
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        res = []
        for p in sorted(people, key=lambda x: (-x[0], x[1])):
            res.insert(p[1],p)
        return res
    # https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
    def connect(self, root: 'Node') -> 'Node':
        if not root: return root
        row = [root]
        for cur in iter(row):
            if cur.left and cur.right:
                cur.left.next = cur.right
                row.extend([cur.left,cur.right])
            if cur.next and cur.next.left:
                cur.right.next = cur.next.left
        return root
    def connect(self, root: 'Node') -> 'Node': # https://leetcode.com/problems/populating-next-right-pointers-in-each-node/discuss/37484/7-lines-iterative-real-O(1)-space
        pre = root # real root
        while root and root.left:
            next = root.left # next level pointer
            while root:
                root.left.next = root.right # right always exists in perfect BT
                root.right.next = root.next and root.next.left # in case root has sibling
                root = root.next # next sibling
            root = next # root points to next level
        return pre
    # https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/
    def zigzagLevelOrder(self, root: Node) -> List[List[int]]:
        if not root: return []
        res = []; level = [root]; dir = 1
        while level:
            res.append([n.val for n in level][::dir])
            dir *= -1
            level = [c for n in level for c in [n.left,n.right] if c]
        return res
    # lee215 https://leetcode.com/problems/binary-tree-level-order-traversal-ii/discuss/34978/Python-solutions-(dfs-recursively-dfs%2Bstack-bfs%2Bqueue).
    def levelOrderBottom(self, root): # poweric
        if not root: return []
        cur = [root]; res = deque()
        while cur:
            nxt = []
            res.appendleft([])
            for node in cur:
                res[0].append(node.val)
                if node.left: nxt.append(node.left)
                if node.right: nxt.append(node.right)
            cur = nxt
        return list(res)
    def levelOrderBottom(self, root):
        res, queue = [], [root]
        while queue:
            res.append([node.val for node in queue if node])
            queue = [child for node in queue if node for child in (node.left, node.right)]
        return res[-2::-1] # to avoid root = [] situation
    def levelOrderBottom(self, root):
        if not root: return []
        res, queue = deque(), [root]
        while queue:
            res.appendleft([node.val for node in queue])
            queue = [child for node in queue for child in (node.left, node.right) if child]
        return list(res)
    # https://leetcode.com/problems/cousins-in-binary-tree/ https://www.interviewbit.com/problems/cousins-in-binary-tree/
    def cousins(self, A, B):
        cousins, st = [], [c for c in (A.left,A.right) if c]
        while len(st) >= 2:
            next = []; found = False
            for node in st:
                if B in [chld.val for chld in (node.left,node.right) if chld]: found = True
                else: next.extend([node.left,node.right])
            if found: cousins = [n.val for n in next]; break
            else: st = next
        return cousins
    #
    def prefix(self, A: List[str]) -> List[str]:
        wordDict = WordDictionary()
        for w in A: wordDict.insert(w)
        return [wordDict.findPrefix(w) for w in A]
    def prefix(self, words):
        prefixes=defaultdict(int)
        for word in words:
            for i in range(1,len(word)):
                prefixes[word[:i]]+=1 # count all prefix candidates
        out=[]
        for word in words:
            result=word
            for i in range(1,len(word)):
                pref=word[:i]
                if prefixes[pref]==1:
                    result=pref # for each word, search its prefix candidates for first single shortest instance, else use full word
                    break
            out.append(result)       
        return out
    def serialize(self, root: Node) -> List[int]:
        if not root: return ""
        serializedVals: List[int] = []
        queue: Deque[Node] = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                serializedVals.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                serializedVals.append(None)
        while serializedVals[-1] == None: serializedVals.pop()  # Strip trailing None nodes.
        return serializedVals
    def deserialize(self, vals: List[int]) -> Optional[Node]:
        if not vals: return None
        valsIter = iter(vals)
        root = Node(next(valsIter))
        queue = deque([root])
        while queue:
            node = queue.popleft()
            val = next(valsIter, None)
            if val != None:
                node.left = Node(val)
                queue.append(node.left)
            val = next(valsIter, None)
            if val:
                node.right = Node(val)
                queue.append(node.right)
        return root
    # https://leetcode.com/problems/serialize-and-deserialize-bst/
    def serializeBST(self, root: Node) -> str:
        return ','.join(map(str,self.inorderTraversal(root)))
    def deserializeBST(self, data: str) -> Node:
        root = None
        if data:
            def buildTree(vals, l, r):
                node = Node(vals[l]) if l == r else None
                if l < r:
                    mid = (l+r)//2
                    node = Node(vals[mid])
                    node.left = buildTree(vals, l, mid-1) if l < mid else None
                    node.right = buildTree(vals, mid+1, r) if mid < r else None
                return node
            vals = list(map(int, data.split(',')))
            root = buildTree(vals, 0, len(vals)-1)
        return root
    def sortedArrayToBST(self, A):
        root = None
        if A:
            def buildTree(vals, l, r):
                node = Node(vals[l]) if l == r else None
                if l < r:
                    mid = (l+r)//2
                    node = Node(vals[mid])
                    node.left = buildTree(vals, l, mid-1) if l < mid else None
                    node.right = buildTree(vals, mid+1, r) if mid < r else None
                return node
            root = buildTree(A, 0, len(A)-1)
        return root
    # https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/submissions/
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Node:
        valIdxMap = {num:i for i,num in enumerate(inorder)}
        rootIdx = len(postorder) -1
        def dfs(left, right):
            nonlocal rootIdx, valIdxMap
            if rootIdx < 0: return
            val = postorder[rootIdx]; i = valIdxMap[val]
            if left > i or i > right: # i,val go outside in its subtree range left,right
                return 
            root = Node(val)
            rootIdx -= 1
            root.right = dfs(i+1, right) # index i of root, whose right subtree starts at i+1
            root.left  = dfs(left, i)    # root's left subtree ends at i
            return root
        return dfs(0, len(inorder)) if inorder else None
    # https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    def buildTreePreIn(self, preorder: List[int], inorder: List[int]) -> Node:
        valIdxMap = {num:i for i,num in enumerate(inorder)}
        def buildTree(ps, ins, ine): # preorder start, inorder start/end indices
            root = None
            if ps < len(preorder) and ins <= ine:
                root = Node(preorder[ps])
                i = valIdxMap[preorder[ps]]
                root.left = buildTree(ps+1, ins, i-1) # left subtree root, start/end inorder indices
                root.right = buildTree(ps+1+i-ins, i+1, ine) # right substree root = ps+1 + numsOnLeft (i-is), start/end inorder indices
            return root
        return buildTree(0, 0, len(inorder)-1)
    # https://github.com/SrGrace/InterviewBit/blob/master/Trees/InorderTraversalOfCartesianTree.cpp
    def buildTreeCartesian(self, A: List[int]) -> Node:
        def buildTree(st, ed):
            if st > ed: return None
            max = A[st]; idx = st
            for i in range(st, ed+1):
                if A[i] > max: max = A[i]; idx = i
            root = Node(max)
            root.left = buildTree(st, idx-1)
            root.right = buildTree(idx+1, ed)
            return root
        return buildTree(0, len(A)-1)
    # https://leetcode.com/problems/maximum-binary-tree-ii/discuss/242936/JavaC%2B%2BPython-Recursion-and-Iteration
    # val is appended to list, hence it is the most-right node, with only left subtree
    def insertIntoMaxTree(self, root, val): # time O(N), space O(N)
        if root and root.val > val:
            root.right = self.insertIntoMaxTree(root.right, val)
            return root
        node = Node(val)
        node.left = root
        return node
    def insertIntoMaxTree(self, root, val):
        pre,cur = None, root # time O(N), space O(1)
        while cur and cur.val > val: pre, cur = cur, cur.right
        node = Node(val) # Find on the right a node cur: cur.val > val > cur.right.val
        node.left = cur
        if pre: pre.right = node
        return root if root.val > val else node
class BSTIterator:

    def __init__(self, root):
        self.root = root
        self.path = []
        self.pushall(root)
        
    def pushall(self, root):
        while root:
            self.path.append(root)
            root = root.left
    def next(self):
        top = self.path.pop()
        self.pushall(top.right)
        return top.val

    def hasNext(self):
        return len(self.path) > 0
if __name__ == "__main__":
    bst = BinarySearchTree()
    A = [20,9,25,5,12,11,14]
    for v in A: bst.insert(v)
    # for v in [20,9,25,5,12]: bst.insert(v)
    bstIter = BSTIterator(bst.root)
    while bstIter.hasNext(): print(bstIter.next()),
    sol = Solution()
    root = sol.buildTreeCartesian([1,2,3])
    assert root.val == 3
    assert sol.maxDepth(bst.root) == 4
    assert sol.minDepth(bst.root) == 2
    assert sol.isBalanced(bst.root) == False
    root = sol.sortedArrayToBST(range(1,16))
    assert root.val == 8
    sstr = ','.join(map(str,range(1,16)))
    root = sol.deserializeBST(sstr)
    assert root.val == 8
    assert sol.serializeBST(root) == sstr
    root = sol.buildTree([9,3,15,20,7],[9,15,7,20,3]) # inorder, postorder
    assert root.val == 3
    root = sol.buildTreePreIn([3,9,20,15,7],[9,15,7,20,3]) # preorder, inorder
    assert root.val == 3
    # assert sol.findTarget(bst.root, 20) == True
    assert sol.isValidBST(bst.root) == True
    # assert sol.inorderTraversal(bst.root) == [5,9,11,12,14,20,25]
    # assert sol.preorderTraversal(bst.root) == [20,9,5,12,11,14,25]
    bst.root.val = 26
    assert sol.isValidBST(bst.root) == False
    assert sol.recoverTree(bst.root) == [25,26]
    # assert sol.isValidBST(sol.recoverTree(bst.root)) == True
    a = sol.deserialize([4,2,6,1,3])
    assert sol.minDiffInBST(a) == 1
    a = sol.deserialize([27,None,34,None,58,50,None,44])
    assert sol.minDiffInBST(a) == 6
    a = sol.deserialize([1,0,48,None,None,12,49])
    assert sol.minDiffInBST(a) == 1
    a = sol.deserialize([1,2,5,3,4,None,6])
    expectedVal = a.val
    sol.flatten(a)
    assert a.val == expectedVal
    a = sol.deserialize([2,3,3,4,5,5])
    assert sol.isSymmetric(a) == False
    a = sol.deserialize([1,2,2,3,4,4,3])
    assert sol.isSymmetric(a) == 1
    a = sol.deserialize([1,2,2,None,3,None,3])
    assert sol.isSymmetric(a) == 0
    a = sol.deserialize([9,-42,-42,None,76,76,None,None,13,None,13])
    assert sol.isSymmetric(a) == False
    assert sol.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]) == [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    assert sol.reconstructQueue([[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]) == [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
    assert [h for h,f in sol.reconstructQueue(list(zip([5,3,2,6,1,4],[0,1,2,0,3,2])))] == [5,3,2,1,6,4]
    a = sol.deserialize([1,2,3,4,None,5,6,None,None,7,8])
    # assert sol.diagonalTraversal(a) == [[1,3,6],[2,5,8],[4,7]]
    assert sol.diagonalTraversal(a) == [1,3,6,2,5,8,4,7]
    assert sol.maxEdgeRemovalToMakeBTEven(sol.deserialize([0,2,1,3])) == 2
    assert sol.maxEdgeRemovalToMakeBTEven(sol.deserialize([0,2,4,3,None,5,None,6,7])) == 2
    assert sol.levelOrderBottom(sol.deserialize([])) == []
    assert sol.levelOrderBottom(sol.deserialize([3,9,20,None,None,15,7])) == [[15,7],[9,20],[3]]
    assert sol.maxEdgeRemovalToMakeForestEven(2, [[1, 2]]) == 0
    assert sol.maxEdgeRemovalToMakeForestEven(6, [[1, 2],[1, 3],[1, 4],[3, 5],[4, 6]]) == 2
    assert sol.maxEdgeRemovalToMakeForestEven(7, [[0,2],[0,1],[0,4],[2,3],[4,5],[5,6],[5,7]]) == 2
    assert sol.cousins(sol.deserialize([1,2,3,4,5,6,7]), 4) == [6,7]
    assert sol.prefix(["zebra", "dog", "duck", "dove"]) == ["z","dog","du","dov"]

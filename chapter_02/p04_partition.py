from chapter_02.linked_list import LinkedList, LinkedListNode


def partition(ll, x):
    current = ll.tail = ll.head
    while current:
        nextNode = current.next # current scans ll
        current.next = None # isolate current node
        if current.value < x: # smaller than x, current becomes new ll.head
            current.next = ll.head
            ll.head = current
        else:
            ll.tail.next = current # current.value >= x, current becomes new ll.tail
            ll.tail = current
        current = nextNode
    if ll.tail.next is not None: ll.tail.next = None # Error check in case all nodes are less than x
    return ll.head
def partition(ll, x):
    current = slow = ll.head # next large node to be swapped with next small node
    while current:
        while slow and slow.value < x: current = slow = slow.next
        while current and current.value >= x:
            current = current.next
        if current and slow:
            slow.value, current.value = current.value, slow.value
    return ll.head
# Preserve node order
def partition(head, x): # separate the list into 2 distinct lists and link them afterwards.
    hd1 = p1 = LinkedListNode(0) # p1, p2 traverses the list and hd1 and hd2 are the heads of two lists
    hd2 = p2 = LinkedListNode(0)
    while head:
        if head.value < x: p1.next = head; p1 = p1.next
        else: p2.next = head; p2 = p2.next
        head = head.next
    p2.next = None #join the lists
    p1.next = hd2.next
    return hd1.next
if __name__ == "__main__":
    # ll = LinkedList.generate(10, 0, 99)
    ll = LinkedList([1,4,3,2,5,2])
    print(ll)
    partition(ll.head, 3)
    print(ll)
    
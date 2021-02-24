import random

class LinkedListNode:
    def __init__(self, value, next_node=None, prev_node=None, random=None):
        self.value = value
        self.next = next_node
        self.prev = prev_node
        self.random = random

    def __str__(self):
        # values = []; cur = self
        # while cur: values.append(str(cur.value)); cur = cur.next
        # return ' -> '.join(values)
        return str(self.value)
class LinkedList:
    def __init__(self, values=None):
        self.head = None
        self.tail = None
        self.val2node = dict()
        if values is not None:
            self.add_multiple(values)

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next

    def __str__(self):
        values = [str(x) for x in self]
        return " -> ".join(values)

    def __len__(self):
        result = 0
        node = self.head
        while node:
            result += 1
            node = node.next
        return result

    def values(self):
        return [x.value for x in self]

    def add(self, value):
        if self.head is None:
            self.val2node[value] = self.tail = self.head = LinkedListNode(value)
        else:
            self.val2node[value] = self.tail.next = LinkedListNode(value)
            self.tail = self.tail.next
        return self.tail

    def add_to_beginning(self, value):
        if self.head is None:
            self.tail = self.head = LinkedListNode(value)
        else:
            self.head = LinkedListNode(value, self.head)
        return self.head

    def add_multiple(self, values):
        for v in values:
            self.add(v)
    def add_randoms(self, randoms):
        cur = self.head
        for v,ri in randoms:
            if ri != None:
                r = self.head
                while ri: r = r.next; ri -= 1
                cur.random = r
            cur = cur.next

    @classmethod
    def generate(cls, k, min_value, max_value):
        return cls(random.choices(range(min_value, max_value), k=k))


class DoublyLinkedList(LinkedList):
    def add(self, value):
        if self.head is None:
            self.tail = self.head = LinkedListNode(value)
        else:
            self.tail.next = LinkedListNode(value, None, self.tail)
            self.tail = self.tail.next
        return self

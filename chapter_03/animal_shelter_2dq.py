"""
An animal shelter holds only dogs and cats, and operates on a strictly "first in, first out" basis. 
People must adopt either the "oldest" (based on arrival time) of all animals at the shelter, or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of that type). 
They cannot select which specific animal they would like. 
Create the data structures to maintain this system and implement operations such as enqueue, dequeueAny, dequeueDog and dequeueCat.

Example
int CAT = 0
int DOG = 1

enqueue("james", DOG);
enqueue("tom", DOG);
enqueue("mimi", CAT);
dequeueAny();  // should return "james"
dequeueCat();  // should return "mimi"
dequeueDog();  // should return "tom"
Challenge
Can you do it with single Queue?
"""

# use two queues
from collections import deque
import unittest

class Animal:
    def __init__(self, name, order):
        self.order = order
        self.name = name


class Cat(Animal):
    def __init__(self, name, order = 0):
        super(Cat, self).__init__(name, order)

class Dog(Animal):
    def __init__(self, name, order = 0):
        super(Dog, self).__init__(name, order)

class AnimalShelter(object):
    def __init__(self):
        self.dogs = deque([])
        self.cats = deque([])
        self.count = 0

    def __len__(self):
        return len(self.dogs) + len(self.cats)

    def enqueue(self, animal):
        animal.order = self.count
        if isinstance(animal, Dog):
            self.dogs.append(animal)
        elif isinstance(animal, Cat):
            self.cats.append(animal)
        self.count += 1

    # return a string
    def dequeueAny(self):
        if not self.dogs and not self.cats:
            return None
        if not self.dogs:
            cat = self.cats.popleft()
            return cat.name
        if not self.cats:
            dog = self.dogs.popleft()
            return dog.name
        dog = self.dogs[0]
        cat = self.cats[0]
        if dog.order > cat.order:
            self.cats.popleft()
            return cat.name
        else:
            self.dogs.popleft()
            return dog.name

    # return a string
    def dequeueDog(self):
        if self.dogs:
            dog = self.dogs.popleft()
            return dog.name
        else:
            return None

    # return a string
    def dequeueCat(self):
        if self.cats:
            cat = self.cats.popleft()
            return cat.name
        else:
            return None

class Tests(unittest.TestCase):
    def test_enqueue(self):
        animal_shelter = AnimalShelter()
        animal_shelter.enqueue(Cat("Fluffy"))
        animal_shelter.enqueue(Dog("Sparky"))
        animal_shelter.enqueue(Cat("Sneezy"))
        self.assertEquals(
            len(animal_shelter), 3, "Amount of animals in queue should be 3"
        )
        print(animal_shelter.dequeueAny())
        print(animal_shelter.dequeueCat())
        print(animal_shelter.dequeueAny())

if __name__ == '__main__':
    unittest.main()

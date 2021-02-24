from typing import List
from itertools import groupby
from collections import Counter

def GroupAnagrams():
    strings = initialise_anagrams()
    anagrams = {}
    for i in range(len(strings)):
        word = "".join(sorted(strings[i].lower()))
        if word not in anagrams:
            anagrams.setdefault(word, [])
        anagrams[word].append(strings[i])
    keys = list(anagrams.keys())
    index = 0
    for i in range(len(keys)):
        values = anagrams.get(keys[i])
        for j in range(len(values)):
            strings[index] = values[j]
            index += 1
    print(strings)


def initialise_anagrams():
    strings = [0] * 8
    strings[0] = "abed"
    strings[1] = "later"
    strings[2] = "bead"
    strings[3] = "alert"
    strings[4] = "altered"
    strings[5] = "bade"
    strings[6] = "alter"
    strings[7] = "alerted"
    return strings

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = []; a2i = dict()
        i = 0
        for s in strs: 
            key = frozenset(Counter(s).items())
            if key in a2i:
                groups[a2i[key]].append(s)
            else:
                a2i[key] = i; i += 1; groups.append([s])
        return groups
    def groupAnagramIndices(self, strs: List[str]) -> List[List[str]]:
        groups = []; a2i = dict()
        i = 0
        for j,s in enumerate(strs,1): 
            key = frozenset(Counter(s).items())
            if key in a2i:
                groups[a2i[key]].append(j)
            else:
                a2i[key] = i; i += 1; groups.append([j])
        return groups

if __name__ == "__main__":
    GroupAnagrams()
    sol = Solution()
    # strs = ['abed', 'bead', 'bade', 'later', 'alert', 'alter', 'altered', 'alerted']
    strs = ["eat","tea","tan","ate","nat","bat"]
    assert sol.groupAnagrams(strs) == [["eat","tea","ate"],["tan","nat"],["bat"]]
    strs = ["cat", "dog", "god", "tca"]
    assert sol.groupAnagramIndices(strs) == [[1,4],[2,3]]

from typing import List

class Solution:
    def __init__(self, V: int, color: List[int]):
        self.V = V
        self.color = color
    def isSafeToColor(self, v: int, graphMatrix: List[List[int]], color: List[int], c: int) -> bool:
        for i in range(self.V): # check for each edge
            if graphMatrix[v][i] == 1 and c == color[i]: return False # vertice i matches v color
        return True
    def graphColorUtil(self, graphMatrix: List[List[int]], m: int, color: List[int], v: int) -> bool:
        if v == self.V: return True # All vertices are assigned a color 
        for i in range(1,m+1): # color 1..m
            if self.isSafeToColor(v, graphMatrix, color, i): # Assign color i to vertice v, safe?
                color[v] = i
                if self.graphColorUtil(graphMatrix, m, color, v + 1):  return True # repeat for vertice v+1
                color[v] = 0 # not safe, backtrack
        return False # If no color can be assigned to  vertex
    # Returns False if the m colors cannot be assigned else True
    def graphColoring(self, graphMatrix: List[List[int]], m: int) -> bool:
        color = [0]*self.V
        if not self.graphColorUtil(graphMatrix, m, color, 0): # vertex 0
            print("Color schema not possible")
            return False
        print("Color schema for vertices are: ", color)
        return True

if __name__ == '__main__':
    m = 3 # Number of colors
    sol = Solution(4, list[range(1,m+1)])
    graphMatrix = [[ 0, 1, 1, 1 ],
                   [ 1, 0, 1, 0 ],
                   [ 1, 1, 0, 1 ],
                   [ 1, 0, 1, 0 ]]
    assert sol.graphColoring(graphMatrix, m) == True
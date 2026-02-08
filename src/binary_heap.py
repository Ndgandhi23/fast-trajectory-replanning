# binary_heap.py
from typing import Tuple

class BinaryHeap:
    def __init__(self):
        self.heap = []  # List of (priority, position) tuples
        self.position_map = {}  # Maps position -> index in heap (for O(1) lookup)
    
    def __len__(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def _parent(self, i):
        return (i - 1) // 2
    
    def _left_child(self, i):
        return 2 * i + 1
    
    def _right_child(self, i):
        return 2 * i + 2
    
    def _swap(self, i, j):
        pos_i = self.heap[i][1]
        pos_j = self.heap[j][1]
        self.position_map[pos_i] = j
        self.position_map[pos_j] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _heapify_up(self, i):
        while i > 0:
            parent = self._parent(i)
            if self.heap[i][0] < self.heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break
    
    def _heapify_down(self, i):
        while True:
            smallest = i
            left = self._left_child(i)
            right = self._right_child(i)
            
            if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            
            if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break
    
    def insert(self, priority, position):
        self.heap.append((priority, position))
        index = len(self.heap) - 1
        self.position_map[position] = index
        self._heapify_up(index)
    
    def extract_min(self):
        """Remove and return element with minimum priority."""
        if self.is_empty():
            return None
        
        min_elem = self.heap[0]
        min_pos = min_elem[1]
        
        # Remove from position map
        del self.position_map[min_pos]
        
        # Move last element to root
        last_elem = self.heap.pop()
        
        if len(self.heap) > 0:
            self.heap[0] = last_elem
            self.position_map[last_elem[1]] = 0
            self._heapify_down(0)
        
        return min_elem
    
    def contains(self, position):
        return position in self.position_map
    
    def peek(self):
        if self.is_empty():
            return None
        return self.heap[0]
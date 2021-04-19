# Trees Traversal 

### Level-Order Binary Tree traversal (Breadth-first searh, BFS)
```python 
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

        
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        result = []
        
        if root is None: 
            return result 
        
        q = []
        
        q.append(root)
        
        while len(q)>0: 
            size = len(q)
            current_level = []
            
            for i in range(0, len(q)): 
                current_node = q.pop(0)
                current_level.append(current_node.val)
                
                if current_node.left is not None: 
                    q.append(current_node.left)

                if current_node.right is not None:
                    q.append(current_node.right)
            result.append(current_level)
            
        return result
  ```
  
  

### Pre Order Traversal 

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None: 
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

```

### In Order Traversal 
```python 
    
        
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None: 
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

### Post Order Traversal
```python 
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None: 
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) +[root.val]
```        


<hr>
# Advanced-Python
OOP, Inheritance, decorators, static methods


## Code Review Cheat Sheet
* Use `pylint` to standardize code format.

**Questions to Ask Yourself When Conducting a Code Review**

**Is the code clean and modular?**
* Can I understand the code easily?
* Does it use meaningful names and whitespace?
* Is there duplicated code?
* Can you provide another layer of abstraction?
* Is each function and module necessary?
* Is each function or module too long?

**Is the code efficient?**
* Are there loops or other steps we can vectorize?
* Can we use better data structures to optimize any steps?
* Can we shorten the number of calculations needed for any steps?
* Can we use generators or multiprocessing to optimize any steps?

**Is documentation effective?**
* Are in-line comments concise and meaningful?
* Is there complex code that's missing documentation?
* Do functions use effective docstrings?
* Is the necessary project documentation provided?

**Is the code well tested?**
* Does the code have high test coverage?
* Do tests check for interesting cases?
* Are the tests readable?
* Can the tests be made more efficient?

**Is the logging effective?**
* Are log messages clear, concise, and professional?
* Do they include all relevant and useful information?
* Do they use the appropriate logging level?


<hr> 


# Linked Lists 
* A linked list is a chain of values connected with pointers.   
  * An array is a sequence of fixed size. Alinked list can have its elements to be dynamically allocated. 
* The starting node of a linked list is referred to as the header. 

### Python Implementation of a Linked List 
```python 
class Node:
    def __init__(self,val):
        self.val = val
        self.next = None # the pointer initially points to nothing
```

Once you have the Node class, you can impleent any linked list as follows: 

```python 
node1 = Node(12) 
node2 = Node(99) 
node3 = Node(37) 
node1.next = node2 # 12->99
node2.next = node3 # 99->37
# the entire linked list now looks like: 12->99->37
```


### Useful Functions 

**Truth Table**
```
import itertools

def truth_table(n): 
  return list(itertools.product([0, 1], repeat=n))
```
### Combinations and Permutations 
* **Combination**: Sampling without replacement where order does not matter 
* **Permutation**: Sampling without replacement where order does matter

```python
def factorial(n): 
  if n==0 or n==1:
    return 1 
  else:
    return n*factorial(n-1)

#n is total number of items
#r is how many you're choosing
def combination(n, r):
  return factorial(n)/(factorial(n-r)*factorial(r))

#n is total number of items
#r is how many you're choosing
def permutation(n, r):
  return factorial(n)/(factorial(n-r))
```

<hr>



**Named tuple - kind of a relational database**
```
from collections import namedtuple

Student = namedtuple('Student', 'fname, lname, age')
s1 = Student('John', 'Clarke', '13')
print(s1.fname)
```

Output: 
```
Student(fname='John', lname='Clarke', age='13')
```

<hr>



**Stack and queue**

Stack - LIFO
```
from collections import deque 

from collections import deque

# 
list = ["a","b","c"]
deq = deque(list)
deq.append('d') # deque(['a', 'b', 'c', 'd'])
deq.pop() # 'd'
```



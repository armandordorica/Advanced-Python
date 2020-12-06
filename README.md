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



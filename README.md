# Training a Classifier 
### Approach
1. Define a model 
2. Define a cost function
3. Minimize the cost using gradient descent 

Classification error and squared error are problematic cost functions for classification. 

# Gaussian Processes 
For a Gaussian radom process, WSS implies SSS since the process is completely specified by its mean and autocorrelation functions. 

### Autocorrelation 
The autocorrelation function of a Gaussian process is the delta function. 
A process is Wide Sense Stationary if its mean and autocorrelation functions are time inveriant. 
   * $E(X(t)) = \mu$
   * $R_X(t_1, t_2)$ is a function only of the time difference $t_2 - t_1$
   * 


# Dimensionality Reduction with PCA 

1. Standardize your data (0 mean, unit variance)
2. Compute the covariance matrix of the dataframe. Hint: The dimensions of your Cov matrix should be mxm where m represents the number of features.
3. Compute eigenvalues and eigenvectors using np.linalg.eigh.
    * As we can see, eigenvalues are just 149 but eigenvectors are 149x149 matrices (because num_features = 149 (days)).
Eigenvectors here are the columns not the rows. For example, first column corresponds to first eigenvector.
5. Show the effectiveness of your principal components in covering the variance of the dataset with a scree plot. How many PCs do you need to cover 99\% of the dataset's variance?
    * An eigenvalue is a number, telling you how much variance there is in the data in that direction, in the example above the eigenvalue is a number telling us how spread out the data is on the line. The eigenvector with the highest eigenvalue is therefore the principal component.
 The first three eigenvectors are able to explain 99.21% of the variance.
```python
args = (-eigenValues).argsort()
eigenValues = eigenValues[args]
eigenVectors = eigenVectors[:, args]


eigValSum = sum(eigenValues)
expVar = [eigV/eigValSum*100 for eigV in eigenValues]
cumExpVar = np.cumsum(expVar)
cumExpVar
```

7. Show the first 10 principal components (Eigenvectors) plotted as a time series.
8. Based on your knowledge of the dataset contents, can you explain what any of the principal components might represent?
https://github.com/armandordorica/APS1070_Project3_PCA/blob/master/Project_3_(1).ipynb

# Useful Plot Templates 

### Probability of Fraud given the Fraud Score based on historical data
```python
import seaborn as sns
sns.set_theme()

plt.figure(figsize=(20,10))
plt.plot(scores_list, fraud_rates1_list, label='Confirmed Fraud')
plt.plot(possible_scores_list_neg, fraud_rates0_list, label='Non Confirmed Fraud')
plt.ylabel("Conditional Probability of Fraud given the Fraud Score")
plt.xlabel("Predicted Fraud Risk Score")
plt.legend()
plt.title("Probability of Fraud given the Fraud Score based on historical data", fontsize=20)
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/1_proba_fraud.png?raw=true)

### Optimal threshold for classification
```pyton 
from sklearn.metrics import f1_score

possible_thresholds = np.arange(0, 1, 0.01)

f1_scores = []
for i in range(0, len(possible_thresholds)): 
    df_pre_20191120['y_pred_binary'] = np.where(df_pre_20191120['y_pred_pre_20191120']>possible_thresholds[i], 1, 0)
    
    f1_scores.append(f1_score(df_pre_20191120['y_pred_binary'] , df_pre_20191120['fraud_acct']))

plt.figure(figsize=(20,10))
plt.scatter(possible_thresholds, f1_scores)
plt.title("Finding Optimal Threshold", fontsize=20)
plt.xlabel("Possible Threshold")
plt.ylabel("F1 score")
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/2_optimal_threshold.png?raw=true)

### Dist Plot 

```python 
import seaborn as sns
from scipy.stats import norm

sns.set_theme();

temp_df = df[df['fraud_category']!='other']
fraud_categories = temp_df['fraud_category'].unique()

i=0

plt.figure(figsize=(20,10))
for i in range(0, len(fraud_categories)): 
    temp_df2 = temp_df[temp_df['fraud_category']==fraud_categories[i]]
    sns.distplot(temp_df2['systemcreditlimit'], label=fraud_categories[i])

plt.legend(title='Status Reason')
plt.title("Probability Density by Status Reason for 'Fraud' accounts by Credit Limit Assigned", fontsize=20)
```

![](https://user-images.githubusercontent.com/14205978/115934477-153b7a80-a45f-11eb-907f-793a1c79a661.png)

### ROC curve 
```python 
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# generate 2 class dataset
X_ex, y_ex = make_classification(n_samples=1000, n_classes=2, random_state=1)
# # split into train/test sets
trainX, testX, trainy, testy = train_test_split(X_ex, y_ex, test_size=0.9, random_state=2)
# # generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)

# calculate AUC comparing historical output with binary output
auc = roc_auc_score(df_pre_20191120['fraud_acct'].values,df_pre_20191120['y_pred_binary'].values)
print('AUC: %.3f' % auc)

# calculate AUC comparing historical output with regression output
auc = roc_auc_score(df_pre_20191120['fraud_acct'].values,df_pre_20191120['y_pred_pre_20191120'].values)
print('AUC: %.3f' % auc)

rf_fpr, rf_tpr, _ = roc_curve(df_pre_20191120['fraud_acct'].values, df_pre_20191120['y_pred_pre_20191120'].values)
rf_fpr, rf_tpr, _
pyplot.figure(figsize=(20,10))
pyplot.title("ROC curve", fontsize=20)

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Coin Toss (No Skill)')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest ROC Curve')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/3_roc_curve.png?raw=true)

### Classic bar plot 
```python 
plt.figure(figsize=(20,10))
plt.bar(used_features['feature_names'], used_features['feature_importances'])
plt.xticks(rotation=90)
plt.title("Feature Importances", fontsize=20)
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/5_classic_bar_plot.png?raw=true)

### Classic scatter plot 
```python 
import seaborn as sns
sns.set_theme()

plt.figure(figsize=(20,10))
plt.scatter( df_pre_20191120['FinalApplicationRiskScore'],df_pre_20191120['y_pred_pre_20191120'])
plt.xlabel("Application TU Credit Score")
plt.ylabel("Predicted Fraud Score")
plt.title("Predicted Fraud Score vs TU Credit Score", fontsize=20)
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/6_classic_scatter.png?raw=true)

### Whisker plot (distribution plot) over time 
```python
import seaborn as sns
sns.set_theme(style="whitegrid")

credit_score_dist_df = query_to_df("""select substring(cast(applicationdate as varchar(12)),1,7) as yr_mth, 
                    systemcreditscore
                    from flxdw_rpt.dbo.applications_tu_details_deduped
where decisionsystem = 'TU'
""")

credit_score_dist_df['systemcreditscore'] = credit_score_dist_df['systemcreditscore'].astype(float)

plt.figure(figsize=(20, 6))

ax = sns.boxplot(x="yr_mth", y="systemcreditscore", data=credit_score_dist_df,
                 order=credit_score_dist_df['yr_mth'].unique())

ax.set_title('Distribution of Application Credit Scores over Time ', 
             fontsize = 20) 
```

![](https://github.com/armandordorica/Advanced-Python/blob/master/images/7_whisker_plot.png?raw=true)

### Plot Correlation
```python
 
def plot_fraud_rates(df, var_name): 
    plt.figure(figsize=(20,6))
    plt.scatter(df.index, df['fraud_rate']*100)
    plt.title("Historical fraud accounts rates by {}".format(var_name), fontsize=20)
    plt.xlabel(var_name)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=90)

def plot_correlation(df, var_name):
    target_var = 'fraud_acct'
    df2 = pd.get_dummies(df[[var_name, target_var]], columns=[var_name])
    x_labels = []
    labels = list(df2.columns[1:])
    for i in range(0,len(labels)):
        x_labels.append(labels[i].split('_')[-1])

    df3 = df2.corrwith(df2.fraud_acct)[1:]
    df3.index = x_labels

    df3.plot.bar(
            figsize = (20, 10), title = "Correlation with {}".format(var_name), fontsize = 15, grid = True)
    plt.title("Correlation between {} and Fraud".format(var_name), fontsize=20)
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel("Correlation with {}".format(target_var), fontsize=20)
    
plot_correlation(df, 'num_apps_same_phone')
```
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/corr_num_apps.png?raw=true)

<hr>


# Binary Trees
### Binary Tree List to List of Nodes

Given a list of integers to represent a binary tree, return a list of TreeNode objects 
```python 
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
def int_list_to_node_list(a):
    if len(a) <=2: 
        num_nulls = 3-len(a)
        nulls_list = [None]*num_nulls
        a = a + nulls_list
    
    nodes = []
    i = 0
    while 2*i+2 < len(a):    
        nodes.append(TreeNode(a[i], a[2*i+1], a[2*i+2]))
        i+=1
        
    node_vals = [x.val for x in nodes]
    missing_vals = [x for x in a if x not in node_vals and x is not None]

    for x in missing_vals: 
        nodes.append(TreeNode(x, None, None))
        
    return nodes

```
<hr>

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



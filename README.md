# Anatomy of a README 
* Description
* Installation
* Usage


* What steps are need to be taken to get the code up and running? 
* What should the user have installed or configured? 
* What might they have a hard time understanding right away?


Tool to preview a markdown file: 
`https://dillinger.io/`.


### Google Style Python DocStrings 
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

<hr>

### Making a Python Package
A *Python module* is just a Python file containing code. 
A package is essentially a collection of modules placed into a directory. 
A Python package also needs an `__init__.py` file. This file is telling Python that this folder contains a package. 
* A package always needs to have an `init` file even if the file is completely empty. The code inside the `init` file gets run whenever you import a package inside a Python program. 
* There should be a `setup.py` file at the same level as the core folder, i.e. `distributions`. This file is necessary for `pip` installing. `pip` will automatically look for this file.
    * This file will contain information of metadata about the package like the package name, version, description, etcetera. 
 
It will look something like this: 

`setup.py`

```python 
from setuptools import setup

setup(name='distributions', 
      versions='0.1',
      description='Gaussian distributions', 
      packages=['distributions'], 
      zip_safe=False)
```



### Magic Methods
* Magic methods let you overwrite and customize default Python behaviour. For example, the init method lets you customize how Python instantiates an object. 
* The `_add_` method overwrites the behaviour of the `+` sign.  
 * Example use case: Imagine you create a `Gauss` class where you are able to define a Gaussian function based on the mean and variance. You want to be able to change the behaviour of the `+` sign such that when you add two Gaussians you get a Gaussian back. 
* The `__repr__` method is what says what is printed when a single variable is on a cell on a Jupyter Notebook. 
 * You can redefine this method so that it overwrites the Python default. 


### Inheritance 

* `Shirt` and `Pants` object inherit from the `Clothing` class. This tells Python that they will inherit all of the attributes and functions from the Clothing class. 
* `Clothing.__init__(self, color, size, style, price)` --> this means that the `Shirt` object initializes itself using the Clothing's init method. Then you can add extra attributes like the `long_or_short` attribute in the `Shirt` class or `waist` in the `Pants` class. 
* You can also extend the `Shirt` or `Pants` class by adding more methods. To give an example, I could add a `double_price()` method just to the `Shirt` class. 
* You can also overwrite any of the `Clothing` methods. For example, the Pants class is overwriting the `calculate_discount()` method. 
* One of the main benefits of inheritance is that you can add atributes and methods to the `Clothing` class, and then both the `Shirt` and `Pants` class will have those attributes and methods as well. 


```python 
class Clothing: 
    def __init__(self, color, size, style, price): 
        self.color = color
        self.size = size
        self.style = style
        self.price = price 
        
    def change_price(self, price): 
        self.price = price 
        
    def calculate_discount(self, discount): 
        return self.price * (1 - discount) 
```



```python 

class Shirt(Clothing): 
    def __init__(self, color, size, style, price, long_or_short): 
        Clothing.__init__(self, color, size, style, price) 
        self.long_or_short = long_or_short 
        
    def double_price(self): 
        self.price = 2*self.price 
        
class Pants(Clothing): 
    def __init__(self, color, size, style, price, waist): 
        Clothing.__init__(self, color, size, stype, price)
        self.waist = waist
        
    def calculate_discount(self, discount): 
        return self.price * (1- discount/2)
```
 
     

# A/B Testing 
https://colab.research.google.com/drive/1hE_e6iXJImfqpI4M8WmJDRYMiI2oJJbX?usp=sharing

# Previous Projects 
**APS 1070**
* Project 1 
  * Vectorized coding (as opposed to for loops)
  * Data Standardization (always standardize on training set and use same scaler for testing set)
    * Effects of data standardization on accuracy by number of neighbours (for KNN specifically)
  * KNN classification 
  * Cross validation 
  * Recursive Feature Selection using Random Forests 

* Project 2 - Anomaly Detection 
  * Gaussian Mixture Models 
    * Fitting Gaussians to a distribution
  * Naive Bayes
    * Pairplots 

* Project 3 - Principal Component Analysis to COVID-19 dataset (number of total cases fo different countries at the end of each day)
  * How to obtain principal components based on covariance matrix, eigenvalues, and eigenvectors. 
  * Scree plots - Line plot of the eigenvalues of factors or principal components in an analysis. 
  * Data reconstruction (compression)
  * ARIMA 
    * Out of sample timeseries forecasts with ARIMA 

* Project 4 - Linear Regression with Gradient Descent 
  * Linear Regression - direct solution
  * Gradient Descent
  * Mini-batch Gradient Descent
  * Regularizers 
  * Momentum 
  * Jointplots 



**MIE1516 - Probabilistic Graphical Models and Deep Learning**

* Project 1 - Bayesian Networks and Markov Random Fields
  * Variable elimination 

* Project 2 - Monte Carlo Methods 
  * Monte-Carlo simulations 
  * Gibbs Sampling 

* Project 3 - Neural Networks
  * Implement your own PyTorch

* Project 4 - Convolutional Networks 
  * LSTM 
  * Recurrent Neural Networks 
  * CNNs with pretrained word embeddings 
  * Multinomial Naive Bayes with TF-IDF feature selection
  * Bidirectional LSTM RNN 

* Final Project
  * Twitter sentiment analysis 
    * ARIMA 
    * Sentiment analysis
    * Linear Regression
    * Twitter API


**ECE 1513 - Machine Learning**
* Project 3
  * Multilayer perceptron using backpropagation

* Project 4 
  * Convolutional Neural Networks on CIFAR and MNIST data 
  * Architecture of Convolutional Networks 
  * Activation functions (sigmoid, relu, tanh, etc)

* Project 5 - Adversarial Networks (GANs)
  * Image Classification of MNIST data 
  * Self generate many more images to get better at training (data augmentation)

* Project 6 - Gated Recurrent Units (LSTM with a forget gate) on Ornstein-Uhlenbeck process
<hr>

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


### Annotated Scatterplot 
Summary_df looks like this: 
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/summary_df.png?raw=true)

```
x = summary_df['fraud_corr']
y = summary_df['fraud_rate']

fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x,y)
labels = summary_df.index

for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]), fontsize=8)
    
ax.set_title("Correlation of Intellicheck Scanning Rates to Fraud Rates by Merchant", fontsize=20)
ax.set_xlabel("Fraud Correlation")
ax.set_ylabel("Fraud Rate")
```

which yields the following plot: 
![](https://github.com/armandordorica/Advanced-Python/blob/master/images/annotated_scatter_plot.png?raw=true)


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

### Bar Plot with Multiple Axis 
```
sns.set_style('white')
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.bar(merchant_names_hist_filtered.index, merchant_names_hist_filtered.bk_scanID)

ax1.set_ylabel('Number of Scans')
ax1.set_title("Number of Scans by Merchant as of {}".format(date), fontsize=20)

ax2 = ax1.twinx()

ax2.bar(merchant_names_hist_filtered.index, merchant_names_hist_filtered['% of total scans'])

ax2.set_ylabel('% of Total apps')
ax1.set_xticklabels(merchant_names_hist_filtered.index, rotation = 90)
```

![](https://github.com/armandordorica/Advanced-Python/blob/master/images/bar_plot_two_axes.png?raw=true)

### Comparison Bar Plot 
```
X = list(provinces_summary_df['Province/State'])
general_pct = provinces_summary_df['pop_pct_x']
bad_pop_pct = provinces_summary_df['pop_pct_y']
  
X_axis = np.arange(len(X)) 
  
plt.figure(figsize=(20,10))
plt.bar(X_axis - 0.2, general_pct, 0.4, label = 'General Population Distribution') 
plt.bar(X_axis + 0.2, bad_pop_pct, 0.4, label = 'Bad Population Distribution') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Provinces") 
plt.ylabel("% of the Population") 
plt.title("Comparison of distributions across provinces general pop vs bad pop", fontsize=20) 
plt.legend() 
plt.show() 
```

![](https://github.com/armandordorica/Advanced-Python/blob/master/images/bar_plot_side_by_side.png)

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



### Random Forests Validation 
In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run, as follows:

Each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.

Put each case left out in the construction of the kth tree down the kth tree to get a classification. In this way, a test set classification is obtained for each case in about one-third of the trees. At the end of the run, take j to be the class that got most of the votes every time case n was oob. The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. This has proven to be unbiased in many tests (https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm).

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

### See if two objects are equal
* The `is` keyword is used to test if two variables refer to the same object.
* The test returns True if the two objects are the same object.
* The test returns False if they are not the same object, even if the two objects are 100% equal.
* Use the == operator to test if two variables are equal.

```python
x = ["apple", "banana", "cherry"]

y = ["apple", "banana", "cherry"]

print(x is y) //--> False

print(x ==y) //--> True
``` 


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

### Find a Tree's Maximum Depth (Binary tree case) 
* The depth of each root node is 1
* For each know, if we know its depth, we know the depth of its children. Therefore, if we pass the depth of a node as a parameter when calling the function recursively, all nodes will know their depth. 
 * For the leaf nodes, we can use the depth to update the final answer. 

#### Pseudocode for Recursive Function `maximum_depth(root, depth)`: 
```
1. return if root is null 
2. If root is a leaf node: 
3.  answer = max(answer, depth) // update answer if needed 
4. maximum_depth(root.left, depth+1) //call function recursively for left child 
5. maimum_depth(root.right, depth+1) //call function recursively for right child
```

```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# private int answer; // don't forget to initialize answer before call maximum_depth
# private void maximum_depth(TreeNode root, int depth) {
#     if (root == null) {
#         return;
#     }
#     if (root.left == null && root.right == null) {
#         answer = Math.max(answer, depth);
#     }
#     maximum_depth(root.left, depth + 1);
#     maximum_depth(root.right, depth + 1);
# }
   
            
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        
        if root is None: 
            return 0 
        return (1 + max(self.maxDepth(root.left), self.maxDepth(root.right)))
```

### Check if a tree is symmetric 
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


    
    
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        
        if root is None: 
            return True #null tree reflected upon itself is symmetric
        
        if root.left is None or root.right is None: 
            if (root.left is None and root.right is not None) or (root.right is None and root.left is not None): 
                return False  # only one of the two is None 
            # return root.left == root.right
            return True #both being None 
        
        return self.isMirror(root.left, root.right)
    
    def isMirror(self, left, right):
        if left is not None and right is not None: ## If they are both not empty 
            if left.val == right.val: #The roots of the subtrees must be equal 
                if self.isMirror(left.right,right.left) and self.isMirror(left.left, right.right): 
                    return True 
        
        #If up to one of the two is None
        # return left == right
        
        # If they are both None
        if left == right: 
            return True 
        
        # If only one of them is None 
        if left != right: 
            return False
        # return False
```

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

### Path Sum 

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution: 
    
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root is None: 
            return False 
        ## if your Node is a leaf and the sum of the values are already 0 
        elif (root.left is None and root.right is None and targetSum - root.val ==0): 
            return True 
        else: 
            # check if you can accomplish this from either left or right children
            return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val) 
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

**Generate list of dates** 
```python
min_date = dates['min_date'].astype(str).iloc[0]
max_date = dates['max_date'].astype(str).iloc[0]
min_date, max_date

dates_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(min_date,max_date)]
dates_list
```

**Show all columns on Pandas**
```python
pd.set_option('display.max_columns', None)
```

**Find common elements in two lists with numpy**
```
common_elements = np.intersect1d(list_A, list_B)
```


**Filter out for elements in list that satisfy a condition (numpy)**
```
myarray[myarray < threshold]
```


**Equivalent to `row_number` on pandas**
Counts the sequence every time it sees a new value. 

```
df.groupby((df['col_name'] != df['col_name'].shift(1)).cumsum()).cumcount()+1
```

```python
df['rank_num'] = df.groupby('scan_reference')['date'].rank(method='first')
```

### Group by `grouping_var` having count > `threshold`
```python
def groupby_having_count(df, grouping_var, threshold=1): 
    g = df.groupby(grouping_var) 
    return g.filter(lambda x: len(x) > threshold)
```
    
    
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


### Example of `zip` applied to calculating dot product 
```python
def dot_product(x,y): 
    z = list(zip(x,y))
    return sum([i[0]*i[1] for i in z])
```


### Handy People at Jumio 
* Ronald Streicher - Product Manager that can help with how back office agents handle transactions.
* Roman Sedlecky - Developer that works on the software used by the Back Office agents. 
* Tomasz Szydzik - Machine Learning Engineer who can help with Amazon Sagemaker. 
* Bill Inglebright - InfoSec guy who can help getting access to PII data
* Christian Schwaiger -  InfoSec guy who can help getting access to PII data (Bill seems more helpful though)
* Mark Burrett - Lead Web - See which events get triggered from a PII standpoint
* Lucas Danzer - Lead Mobile - See which events get triggered from a PII standpoint
* Lei Guang - Lead Data Scientist (AML workflow)
* Linda Liu - BI Analyst
* Mariano - BI Analyst
* Ahmed Shaaban - BI Analyst 
* Dorota Koseta - Legal Counsel (talk to her regarding legal implications of PII data usage) 
* Joerg Alpers - PO of Fraud Squad 
* Sujai Xavier - Scrum master
* Mohan Kenchappa Lakshmipati - Sr. Python Engineer in MLDI working on the BI migration. He can help with migrating historical data. Alix told him tat we need a plan to have the birthday cake in October and start really pushing a smarter routing to reach decent FAR (False Acceptance Rate) and automation rate in November. 


 ```
 Hey man, in order to get started you'd need to:
clone this repo: https://bitbucket.int.jumio.com/projects/AIML/repos/sandbox-mono/browse
 follow these guides:
https://confluence.int.jumio.com/display/OCR/How+to+create+Sandbox+user+accounts+with+YubiKey+activation
https://confluence.int.jumio.com/display/OCR/How+to+use+the+sandbox+--+v2
https://confluence.int.jumio.com/display/OCR/Onboarding+to+the+new+Prod+ML+environment

Once you get through this let me know and I can demo the way I use it within my work. (Send me an invite in calendar.) (e
```
* Mariano - BI Data Analyst who can help with tables and Tableau. 
* Vinay - Product Manager who can help with test accounts. 
* Vishnu - MLDI help with Sandbox. 
* Thomas Krump - Architecture Development Infrastructure, events data. https://docs.google.com/presentation/d/16JYl9YRNu3Bw69SfMCz05TDHDvMueA__bIDADefI5RU/edit#slide=id.p
 * https://jenkins-qa.int.jumio.com/job/user_create_new/build?delay=0sec 
 * He'll send a link to a web app that we can use to play around with IDs and see what events they generate. 
* 



### Leave only last commit on Git
```
rm -rf .git
git init

git add *.ipynb
git commit -m "removed history"
git remote add origin ssh://git@bitbucket.int.jumio.com:7999/aiml/coinlist_analysis.git
git push -u --force origin master
```

### Show hidden file like `.bash_profile` or `.bashrc`. 
```
ls -a
```

### When to use `.bash_profile` vs `.bashrc`

* `.bash_profile`will be executed at login shells, i.e. interactive shells where you login with your user name and password at the beginning. When you ssh into a remote host, it will ask you for user name and password (or some other authentication) to log in, so it is a login shell.
* When you open a terminal application, it does not ask for login. You will just get a command prompt. In other versions of Unix or Linux, this will not run the `.bash_profile` but a different file .bashrc. The underlying idea is that the `.bash_profile` should be run only once when you login, and the .bashrc for every new interactive shell.

**However, Terminal.app on macOS, does not follow this convention. When Terminal.app opens a new window, it will run `.bash_profile`.**

If you want to have an approach that is more resilient to other terminal applications and might work (at least partly) across Unix/Linux platforms, put your configuration code in .bashrc and source .bashrc from .bash_profile with the following code in .bash_profile:
```
if [ -r ~/.bashrc ]; then
   source ~/.bashrc
fi
```

The if [ -r ... ] tests wether a file exists and is readable and the source command reads and evaluates a file in place. Sometimes you see
```
[ -r ~/.bashrc ] && . ~/.bashrc
```

(mind the spaces) Which is a shorter way to do the same thing.
Since either file can drastically change your environment, you want to restrict access to just you:
```
$ chmod 700 ~/.bash_profile
$ chmod 700 ~/.bashrc
```

### Adding a new SSH key to your github account
https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account


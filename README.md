# Decision Trees in Practice

In this assignment we will explore various techniques for preventing overfitting in decision trees. We will extend the implementation of the binary decision trees that we implemented in the previous assignment. You will have to use your solutions from this previous assignment and extend them.

In this assignment you will:

* Implement binary decision trees with different early stopping methods.
* Compare models with different stopping parameters.
* Visualize the concept of overfitting in decision trees.

Let's get started!

# Fire up Turi Create

Make sure you have the latest version of Turi Create.


```python
import turicreate
```

# Load LendingClub Dataset

This assignment will use the [LendingClub](https://www.lendingclub.com/) dataset used in the previous two assignments.


```python
loans = turicreate.SFrame('lending-club-data.sframe/')
```

As before, we reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.


```python
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')
```

We will be using the same 4 categorical features as in the previous assignment: 
1. grade of the loan 
2. the length of the loan term
3. the home ownership status: own, mortgage, rent
4. number of years of employment.

In the dataset, each of these features is a categorical feature. Since we are building a binary decision tree, we will have to convert this to binary data in a subsequent section using 1-hot encoding.


```python
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]
```

## Subsample dataset to make sure classes are balanced

Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance out our dataset. This means we are throwing away many data points. We used `seed = 1` so everyone gets the same results.


```python
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print("Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data)))
print("Percentage of risky loans                :", len(risky_loans) / float(len(loans_data)))
print("Total number of loans in our new dataset :", len(loans_data))
```

    Percentage of safe loans                 : 0.5022361744216048
    Percentage of risky loans                : 0.4977638255783951
    Total number of loans in our new dataset : 46508


**Note:** There are many approaches for dealing with imbalanced data, including some where we modify the learning algorithm. These approaches are beyond the scope of this course, but some of them are reviewed in this [paper](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5128907&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F69%2F5173046%2F05128907.pdf%3Farnumber%3D5128907 ). For this assignment, we use the simplest possible approach, where we subsample the overly represented class to get a more balanced dataset. In general, and especially when the data is highly imbalanced, we recommend using more advanced methods.

## Transform categorical data into binary features

Since we are implementing **binary decision trees**, we transform our categorical data into binary data using 1-hot encoding, just as in the previous assignment. Here is the summary of that discussion:

For instance, the **home_ownership** feature represents the home ownership status of the loanee, which is either `own`, `mortgage` or `rent`. For example, if a data point has the feature 
```
   {'home_ownership': 'RENT'}
```
we want to turn this into three features: 
```
 { 
   'home_ownership = OWN'      : 0, 
   'home_ownership = MORTGAGE' : 0, 
   'home_ownership = RENT'     : 1
 }
```

Since this code requires a few Python and Turi Create tricks, feel free to use this block of code as is. Refer to the API documentation for a deeper understanding.



```python
loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data = loans_data.remove_column(feature)
    loans_data = loans_data.add_columns(loans_data_unpacked)
```

The feature columns now look like this:


```python
features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features
```




    ['grade.A',
     'grade.B',
     'grade.C',
     'grade.D',
     'grade.E',
     'grade.F',
     'grade.G',
     'term. 36 months',
     'term. 60 months',
     'home_ownership.MORTGAGE',
     'home_ownership.OTHER',
     'home_ownership.OWN',
     'home_ownership.RENT',
     'emp_length.1 year',
     'emp_length.10+ years',
     'emp_length.2 years',
     'emp_length.3 years',
     'emp_length.4 years',
     'emp_length.5 years',
     'emp_length.6 years',
     'emp_length.7 years',
     'emp_length.8 years',
     'emp_length.9 years',
     'emp_length.< 1 year',
     'emp_length.n/a']



## Train-Validation split

We split the data into a train-validation split with 80% of the data in the training set and 20% of the data in the validation set. We use `seed=1` so that everyone gets the same result.


```python
train_data, validation_set = loans_data.random_split(.8, seed=1)
```

# Early stopping methods for decision trees

In this section, we will extend the **binary tree implementation** from the previous assignment in order to handle some early stopping conditions. Recall the 3 early stopping methods that were discussed in lecture:

1. Reached a **maximum depth**. (set by parameter `max_depth`).
2. Reached a **minimum node size**. (set by parameter `min_node_size`).
3. Don't split if the **gain in error reduction** is too small. (set by parameter `min_error_reduction`).

For the rest of this assignment, we will refer to these three as **early stopping conditions 1, 2, and 3**.

## Early stopping condition 1: Maximum depth

Recall that we already implemented the maximum depth stopping condition in the previous assignment. In this assignment, we will experiment with this condition a bit more and also write code to implement the 2nd and 3rd early stopping conditions.

We will be reusing code from the previous assignment and then building upon this.  We will **alert you** when you reach a function that was part of the previous assignment so that you can simply copy and past your previous code.

## Early stopping condition 2: Minimum node size

The function **reached_minimum_node_size** takes 2 arguments:

1. The `data` (from a node)
2. The minimum number of data points that a node is allowed to split on, `min_node_size`.

This function simply calculates whether the number of data points at a given node is less than or equal to the specified minimum node size. This function will be used to detect this early stopping condition in the **decision_tree_create** function.

Fill in the parts of the function below where you find `## YOUR CODE HERE`.  There is **one** instance in the function below.


```python
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    return (len(data)<=min_node_size)
    
```

**Quiz Question:** Given an intermediate node with 6 safe loans and 3 risky loans, if the `min_node_size` parameter is 10, what should the tree learning algorithm do next? Stop

## Early stopping condition 3: Minimum gain in error reduction

The function **error_reduction** takes 2 arguments:

1. The error **before** a split, `error_before_split`.
2. The error **after** a split, `error_after_split`.

This function computes the gain in error reduction, i.e., the difference between the error before the split and that after the split. This function will be used to detect this early stopping condition in the **decision_tree_create** function.

Fill in the parts of the function below where you find `## YOUR CODE HERE`.  There is **one** instance in the function below. 


```python
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split)

```

**Quiz Question:** Assume an intermediate node has 6 safe loans and 3 risky loans.  For each of 4 possible features to split on, the error reduction is 0.0, 0.05, 0.1, and 0.14, respectively. If the **minimum gain in error reduction** parameter is set to 0.2, what should the tree learning algorithm do next? Stop

## Grabbing binary decision tree helper functions from past assignment

Recall from the previous assignment that we wrote a function `intermediate_node_num_mistakes` that calculates the number of **misclassified examples** when predicting the **majority class**. This is used to help determine which feature is best to split on at a given node of the tree.

**Please copy and paste your code for `intermediate_node_num_mistakes` here**.


```python
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    safe = len(labels_in_node[labels_in_node == 1])
    # Count the number of -1's (risky loans)
    risky = len(labels_in_node[labels_in_node == -1])     
    # Return the number of mistakes that the majority classifier makes.
    return min(safe,risky)

```

We then wrote a function `best_splitting_feature` that finds the best feature to split on given the data and a list of features to consider.

**Please copy and paste your `best_splitting_feature` code here**.


```python
def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature    
    return best_feature # Return the best feature we found
```

Finally, recall the function `create_leaf` from the previous assignment, which creates a leaf node given a set of target values.  

**Please copy and paste your `create_leaf` code here**.


```python
def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}
    
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1
    else:
        leaf['prediction'] = -1
        
    # Return the leaf node        
    return leaf 
```

## Incorporating new early stopping conditions in binary decision tree implementation

Now, you will implement a function that builds a decision tree handling the three early stopping conditions described in this assignment.  In particular, you will write code to detect early stopping conditions 2 and 3.  You implemented above the functions needed to detect these conditions.  The 1st early stopping condition, **max_depth**, was implemented in the previous assigment and you will not need to reimplement this.  In addition to these early stopping conditions, the typical stopping conditions of having no mistakes or no more features to split on (which we denote by "stopping conditions" 1 and 2) are also included as in the previous assignment.

**Implementing early stopping condition 2: minimum node size:**

* **Step 1:** Use the function **reached_minimum_node_size** that you implemented earlier to write an if condition to detect whether we have hit the base case, i.e., the node does not have enough data points and should be turned into a leaf. Don't forget to use the `min_node_size` argument.
* **Step 2:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.


**Implementing early stopping condition 3: minimum error reduction:**

**Note:** This has to come after finding the best splitting feature so we can calculate the error after splitting in order to calculate the error reduction.

* **Step 1:** Calculate the **classification error before splitting**.  Recall that classification error is defined as:

$$
\text{classification error} = \frac{\text{# mistakes}}{\text{# total examples}}
$$
* **Step 2:** Calculate the **classification error after splitting**. This requires calculating the number of mistakes in the left and right splits, and then dividing by the total number of examples.
* **Step 3:** Use the function **error_reduction** to that you implemented earlier to write an if condition to detect whether  the reduction in error is less than the constant provided (`min_error_reduction`). Don't forget to use that argument.
* **Step 4:** Return a leaf. This line of code should be the same as the other (pre-implemented) stopping conditions.

Fill in the places where you find `## YOUR CODE HERE`. There are **seven** places in this function for you to fill in.


```python
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached. All data points have the same target value.")
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print("Stopping condition 2 reached. No remaining features.")
        return create_leaf(target_values)
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print("Early stopping condition 1 reached. Reached maximum depth.")
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):
        print("Early stopping condition 2 reached. Reached minimum node size.")
        return create_leaf(target_values)
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = len(left_split[left_split[target] == 1])
    print("left errors: " + str(left_mistakes))
    right_mistakes =  len(right_split[right_split[target] == -1])
    print("right errors: " + str(right_mistakes))
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    print("error before: " + str(error_before_split))
    print("error after: " + str(error_after_split))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if (error_reduction(error_before_split, error_after_split) <= min_error_reduction):
        print("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)
    
    
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))

    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
```

Here is a function to count the nodes in your tree:


```python
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])
```

Run the following test code to check your implementation. Make sure you get **'Test passed'** before proceeding.


```python
small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print('Test passed!')
else:
    print('Test failed... try again!')
    print('Number of nodes found                :', count_nodes(small_decision_tree))
    print('Number of nodes that should be there : 7' )
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Early stopping condition 3 reached. Minimum error reduction.
    Test failed... try again!
    Number of nodes found                : 5
    Number of nodes that should be there : 7


## Build a tree!

Now that your code is working, we will train a tree model on the **train_data** with
* `max_depth = 6`
* `min_node_size = 100`, 
* `min_error_reduction = 0.0`

**Warning**: This code block may take a minute to learn. 


```python
my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Early stopping condition 3 reached. Minimum error reduction.


Let's now train a tree model **ignoring early stopping conditions 2 and 3** so that we get the same tree as in the previous assignment.  To ignore these conditions, we set `min_node_size=0` and `min_error_reduction=-1` (a negative value).


```python
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction=-1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    left errors: 420
    right errors: 38
    error before: 0.4398854961832061
    error after: 0.43702290076335876
    Split on feature emp_length.5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    left errors: 19
    right errors: 23
    error before: 0.4810126582278481
    error after: 0.5316455696202531
    Split on feature home_ownership.MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Split on feature emp_length.n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    left errors: 56
    right errors: 6
    error before: 0.3645833333333333
    error after: 0.6458333333333334
    Split on feature emp_length.< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    left errors: 273
    right errors: 87
    error before: 0.44635193133047213
    error after: 0.38626609442060084
    Split on feature grade.A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    left errors: 101
    right errors: 4
    error before: 0.3016759776536313
    error after: 0.29329608938547486
    Split on feature emp_length.8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 7
    right errors: 2
    error before: 0.36363636363636365
    error after: 0.8181818181818182
    Split on feature home_ownership.OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.


## Making predictions

Recall that in the previous assignment you implemented a function `classify` to classify a new point `x` using a given `tree`.

**Please copy and paste your `classify` code here**.


```python
def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)
```

Now, let's consider the first example of the validation set and see what the `my_decision_tree_new` model predicts for this data point.


```python
validation_set[0]
```




    {'safe_loans': -1,
     'grade.A': 0,
     'grade.B': 0,
     'grade.C': 0,
     'grade.D': 1,
     'grade.E': 0,
     'grade.F': 0,
     'grade.G': 0,
     'term. 36 months': 0,
     'term. 60 months': 1,
     'home_ownership.MORTGAGE': 0,
     'home_ownership.OTHER': 0,
     'home_ownership.OWN': 0,
     'home_ownership.RENT': 1,
     'emp_length.1 year': 0,
     'emp_length.10+ years': 0,
     'emp_length.2 years': 1,
     'emp_length.3 years': 0,
     'emp_length.4 years': 0,
     'emp_length.5 years': 0,
     'emp_length.6 years': 0,
     'emp_length.7 years': 0,
     'emp_length.8 years': 0,
     'emp_length.9 years': 0,
     'emp_length.< 1 year': 0,
     'emp_length.n/a': 0}




```python
print('Predicted class: %s ' % classify(my_decision_tree_new, validation_set[0]))
```

    Predicted class: -1 


Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:


```python
classify(my_decision_tree_new, validation_set[0], annotate = True)
```

    Split on term. 36 months = 0
    Split on grade.A = 0
    At leaf, predicting -1





    -1



Let's now recall the prediction path for the decision tree learned in the previous assignment, which we recreated here as `my_decision_tree_old`.


```python
classify(my_decision_tree_old, validation_set[0], annotate = True)
```

    Split on term. 36 months = 0
    Split on grade.A = 0
    Split on grade.B = 0
    Split on grade.C = 0
    Split on grade.D = 1
    Split on grade.E = 0
    At leaf, predicting -1





    -1



**Quiz Question:** For `my_decision_tree_new` trained with `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, is the prediction path for `validation_set[0]` shorter, longer, or the same as for `my_decision_tree_old` that ignored the early stopping conditions 2 and 3? Shorter

**Quiz Question:** For `my_decision_tree_new` trained with `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, is the prediction path for **any point** always shorter, always longer, always the same, shorter or the same, or longer or the same as for `my_decision_tree_old` that ignored the early stopping conditions 2 and 3? shorter or the same

**Quiz Question:** For a tree trained on **any** dataset using `max_depth = 6`, `min_node_size = 100`, `min_error_reduction=0.0`, what is the maximum number of splits encountered while making a single prediction? 6

## Evaluating the model

Now let us evaluate the model that we have trained. You implemented this evaluation in the function `evaluate_classification_error` from the previous assignment.

**Please copy and paste your `evaluate_classification_error` code here**.


```python
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    mistakes = len(data[data[target] != prediction])
    return mistakes / len(data) 
```

Now, let's use this function to evaluate the classification error of `my_decision_tree_new` on the **validation_set**.


```python
evaluate_classification_error(my_decision_tree_new, validation_set, target)
```




    0.4226626454114606



Now, evaluate the validation error using `my_decision_tree_old`.


```python
evaluate_classification_error(my_decision_tree_old, validation_set, target)
```




    0.3837785437311504



**Quiz Question:** Is the validation error of the new decision tree (using early stopping conditions 2 and 3) lower than, higher than, or the same as that of the old decision tree from the previous assignment? higher

# Exploring the effect of max_depth

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**too small**, **just right**, and **too large**).

Train three models with these parameters:

1. **model_1**: max_depth = 2 (too small)
2. **model_2**: max_depth = 6 (just right)
3. **model_3**: max_depth = 14 (may be too large)

For each of these three, we set `min_node_size = 0` and `min_error_reduction = -1`.

** Note:** Each tree can take up to a few minutes to train. In particular, `model_3` will probably take the longest to train.


```python
model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, 
                                min_node_size = 0, min_error_reduction = -1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction = -1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14, 
                                min_node_size = 0, min_error_reduction = -1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    left errors: 420
    right errors: 38
    error before: 0.4398854961832061
    error after: 0.43702290076335876
    Split on feature emp_length.5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    left errors: 19
    right errors: 23
    error before: 0.4810126582278481
    error after: 0.5316455696202531
    Split on feature home_ownership.MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Split on feature emp_length.n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    left errors: 56
    right errors: 6
    error before: 0.3645833333333333
    error after: 0.6458333333333334
    Split on feature emp_length.< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    left errors: 273
    right errors: 87
    error before: 0.44635193133047213
    error after: 0.38626609442060084
    Split on feature grade.A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    left errors: 101
    right errors: 4
    error before: 0.3016759776536313
    error after: 0.29329608938547486
    Split on feature emp_length.8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 7
    right errors: 2
    error before: 0.36363636363636365
    error after: 0.8181818181818182
    Split on feature home_ownership.OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    left errors: 436
    right errors: 0
    error before: 0.25812167749556997
    error after: 0.2575310100413467
    Split on feature home_ownership.OTHER. (1692, 1)
    --------------------------------------------------------------------
    Subtree, depth = 7 (1692 data points).
    left errors: 83
    right errors: 1000
    error before: 0.2576832151300236
    error after: 0.6400709219858156
    Split on feature grade.F. (339, 1353)
    --------------------------------------------------------------------
    Subtree, depth = 8 (339 data points).
    left errors: 0
    right errors: 256
    error before: 0.2448377581120944
    error after: 0.7551622418879056
    Split on feature grade.G. (0, 339)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (339 data points).
    left errors: 0
    right errors: 256
    error before: 0.2448377581120944
    error after: 0.7551622418879056
    Split on feature term. 60 months. (0, 339)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (339 data points).
    left errors: 36
    right errors: 117
    error before: 0.2448377581120944
    error after: 0.45132743362831856
    Split on feature home_ownership.MORTGAGE. (175, 164)
    --------------------------------------------------------------------
    Subtree, depth = 11 (175 data points).
    left errors: 29
    right errors: 26
    error before: 0.2057142857142857
    error after: 0.3142857142857143
    Split on feature home_ownership.OWN. (142, 33)
    --------------------------------------------------------------------
    Subtree, depth = 12 (142 data points).
    left errors: 24
    right errors: 4
    error before: 0.20422535211267606
    error after: 0.19718309859154928
    Split on feature emp_length.6 years. (133, 9)
    --------------------------------------------------------------------
    Subtree, depth = 13 (133 data points).
    left errors: 0
    right errors: 109
    error before: 0.18045112781954886
    error after: 0.8195488721804511
    Split on feature home_ownership.RENT. (0, 133)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (9 data points).
    left errors: 0
    right errors: 4
    error before: 0.4444444444444444
    error after: 0.4444444444444444
    Split on feature home_ownership.RENT. (0, 9)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (33 data points).
    left errors: 5
    right errors: 0
    error before: 0.21212121212121213
    error after: 0.15151515151515152
    Split on feature emp_length.n/a. (31, 2)
    --------------------------------------------------------------------
    Subtree, depth = 13 (31 data points).
    left errors: 4
    right errors: 0
    error before: 0.16129032258064516
    error after: 0.12903225806451613
    Split on feature emp_length.2 years. (30, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (30 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (164 data points).
    left errors: 44
    right errors: 2
    error before: 0.2865853658536585
    error after: 0.2804878048780488
    Split on feature emp_length.2 years. (159, 5)
    --------------------------------------------------------------------
    Subtree, depth = 12 (159 data points).
    left errors: 38
    right errors: 5
    error before: 0.27672955974842767
    error after: 0.27044025157232704
    Split on feature emp_length.3 years. (148, 11)
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    left errors: 38
    right errors: 0
    error before: 0.25675675675675674
    error after: 0.25675675675675674
    Split on feature home_ownership.OWN. (148, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (148 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (11 data points).
    left errors: 6
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.5454545454545454
    Split on feature home_ownership.OWN. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (5 data points).
    left errors: 3
    right errors: 0
    error before: 0.4
    error after: 0.6
    Split on feature home_ownership.OWN. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (5 data points).
    left errors: 3
    right errors: 0
    error before: 0.4
    error after: 0.6
    Split on feature home_ownership.RENT. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (1353 data points).
    left errors: 353
    right errors: 0
    error before: 0.2609016999260902
    error after: 0.2609016999260902
    Split on feature grade.G. (1353, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1353 data points).
    left errors: 0
    right errors: 1000
    error before: 0.2609016999260902
    error after: 0.7390983000739099
    Split on feature term. 60 months. (0, 1353)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1353 data points).
    left errors: 168
    right errors: 458
    error before: 0.2609016999260902
    error after: 0.4626755358462675
    Split on feature home_ownership.MORTGAGE. (710, 643)
    --------------------------------------------------------------------
    Subtree, depth = 11 (710 data points).
    left errors: 142
    right errors: 82
    error before: 0.23661971830985915
    error after: 0.3154929577464789
    Split on feature home_ownership.OWN. (602, 108)
    --------------------------------------------------------------------
    Subtree, depth = 12 (602 data points).
    left errors: 0
    right errors: 460
    error before: 0.23588039867109634
    error after: 0.7641196013289037
    Split on feature home_ownership.RENT. (0, 602)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (602 data points).
    left errors: 132
    right errors: 27
    error before: 0.23588039867109634
    error after: 0.26411960132890366
    Split on feature emp_length.1 year. (565, 37)
    --------------------------------------------------------------------
    Subtree, depth = 14 (565 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (37 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (108 data points).
    left errors: 26
    right errors: 0
    error before: 0.24074074074074073
    error after: 0.24074074074074073
    Split on feature home_ownership.RENT. (108, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (108 data points).
    left errors: 25
    right errors: 7
    error before: 0.24074074074074073
    error after: 0.2962962962962963
    Split on feature emp_length.1 year. (100, 8)
    --------------------------------------------------------------------
    Subtree, depth = 14 (100 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (643 data points).
    left errors: 185
    right errors: 0
    error before: 0.28771384136858474
    error after: 0.28771384136858474
    Split on feature home_ownership.OWN. (643, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (643 data points).
    left errors: 185
    right errors: 0
    error before: 0.28771384136858474
    error after: 0.28771384136858474
    Split on feature home_ownership.RENT. (643, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (643 data points).
    left errors: 172
    right errors: 28
    error before: 0.28771384136858474
    error after: 0.3110419906687403
    Split on feature emp_length.1 year. (602, 41)
    --------------------------------------------------------------------
    Subtree, depth = 14 (602 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (41 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    left errors: 624
    right errors: 0
    error before: 0.29254571026722925
    error after: 0.29254571026722925
    Split on feature grade.F. (2133, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2133 data points).
    left errors: 624
    right errors: 0
    error before: 0.29254571026722925
    error after: 0.29254571026722925
    Split on feature grade.G. (2133, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2133 data points).
    left errors: 0
    right errors: 1509
    error before: 0.29254571026722925
    error after: 0.7074542897327707
    Split on feature term. 60 months. (0, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2133 data points).
    left errors: 273
    right errors: 737
    error before: 0.29254571026722925
    error after: 0.473511486169714
    Split on feature home_ownership.MORTGAGE. (1045, 1088)
    --------------------------------------------------------------------
    Subtree, depth = 10 (1045 data points).
    left errors: 273
    right errors: 1
    error before: 0.261244019138756
    error after: 0.26220095693779905
    Split on feature home_ownership.OTHER. (1044, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1044 data points).
    left errors: 217
    right errors: 109
    error before: 0.2614942528735632
    error after: 0.31226053639846746
    Split on feature home_ownership.OWN. (879, 165)
    --------------------------------------------------------------------
    Subtree, depth = 12 (879 data points).
    left errors: 0
    right errors: 662
    error before: 0.24687144482366324
    error after: 0.7531285551763367
    Split on feature home_ownership.RENT. (0, 879)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (879 data points).
    left errors: 202
    right errors: 55
    error before: 0.24687144482366324
    error after: 0.2923777019340159
    Split on feature emp_length.1 year. (809, 70)
    --------------------------------------------------------------------
    Subtree, depth = 14 (809 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (70 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (165 data points).
    left errors: 51
    right errors: 3
    error before: 0.3393939393939394
    error after: 0.32727272727272727
    Split on feature emp_length.9 years. (157, 8)
    --------------------------------------------------------------------
    Subtree, depth = 13 (157 data points).
    left errors: 51
    right errors: 0
    error before: 0.3248407643312102
    error after: 0.3248407643312102
    Split on feature home_ownership.RENT. (157, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (157 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    left errors: 5
    right errors: 0
    error before: 0.375
    error after: 0.625
    Split on feature home_ownership.RENT. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1088 data points).
    left errors: 351
    right errors: 0
    error before: 0.3226102941176471
    error after: 0.3226102941176471
    Split on feature home_ownership.OTHER. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1088 data points).
    left errors: 351
    right errors: 0
    error before: 0.3226102941176471
    error after: 0.3226102941176471
    Split on feature home_ownership.OWN. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1088 data points).
    left errors: 351
    right errors: 0
    error before: 0.3226102941176471
    error after: 0.3226102941176471
    Split on feature home_ownership.RENT. (1088, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1088 data points).
    left errors: 331
    right errors: 33
    error before: 0.3226102941176471
    error after: 0.33455882352941174
    Split on feature emp_length.1 year. (1035, 53)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1035 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (53 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.F. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.G. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2058 data points).
    left errors: 0
    right errors: 1335
    error before: 0.35131195335276966
    error after: 0.6486880466472303
    Split on feature term. 60 months. (0, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2058 data points).
    left errors: 285
    right errors: 697
    error before: 0.35131195335276966
    error after: 0.4771622934888241
    Split on feature home_ownership.MORTGAGE. (923, 1135)
    --------------------------------------------------------------------
    Subtree, depth = 10 (923 data points).
    left errors: 285
    right errors: 1
    error before: 0.3087757313109426
    error after: 0.30985915492957744
    Split on feature home_ownership.OTHER. (922, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (922 data points).
    left errors: 237
    right errors: 112
    error before: 0.3091106290672451
    error after: 0.37852494577006507
    Split on feature home_ownership.OWN. (762, 160)
    --------------------------------------------------------------------
    Subtree, depth = 12 (762 data points).
    left errors: 0
    right errors: 525
    error before: 0.3110236220472441
    error after: 0.6889763779527559
    Split on feature home_ownership.RENT. (0, 762)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (762 data points).
    left errors: 219
    right errors: 40
    error before: 0.3110236220472441
    error after: 0.33989501312335957
    Split on feature emp_length.1 year. (704, 58)
    --------------------------------------------------------------------
    Subtree, depth = 14 (704 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (58 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (160 data points).
    left errors: 48
    right errors: 0
    error before: 0.3
    error after: 0.3
    Split on feature home_ownership.RENT. (160, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (160 data points).
    left errors: 47
    right errors: 5
    error before: 0.3
    error after: 0.325
    Split on feature emp_length.1 year. (154, 6)
    --------------------------------------------------------------------
    Subtree, depth = 14 (154 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1135 data points).
    left errors: 438
    right errors: 0
    error before: 0.3859030837004405
    error after: 0.3859030837004405
    Split on feature home_ownership.OTHER. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1135 data points).
    left errors: 438
    right errors: 0
    error before: 0.3859030837004405
    error after: 0.3859030837004405
    Split on feature home_ownership.OWN. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1135 data points).
    left errors: 438
    right errors: 0
    error before: 0.3859030837004405
    error after: 0.3859030837004405
    Split on feature home_ownership.RENT. (1135, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1135 data points).
    left errors: 423
    right errors: 24
    error before: 0.3859030837004405
    error after: 0.39383259911894275
    Split on feature emp_length.1 year. (1096, 39)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1096 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (39 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.F. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.G. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (2190 data points).
    left errors: 0
    right errors: 1276
    error before: 0.41735159817351597
    error after: 0.582648401826484
    Split on feature term. 60 months. (0, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (2190 data points).
    left errors: 306
    right errors: 779
    error before: 0.41735159817351597
    error after: 0.4954337899543379
    Split on feature home_ownership.MORTGAGE. (803, 1387)
    --------------------------------------------------------------------
    Subtree, depth = 10 (803 data points).
    left errors: 276
    right errors: 27
    error before: 0.38107098381070986
    error after: 0.37733499377334995
    Split on feature emp_length.4 years. (746, 57)
    --------------------------------------------------------------------
    Subtree, depth = 11 (746 data points).
    left errors: 276
    right errors: 0
    error before: 0.3699731903485255
    error after: 0.3699731903485255
    Split on feature home_ownership.OTHER. (746, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (746 data points).
    left errors: 216
    right errors: 88
    error before: 0.3699731903485255
    error after: 0.4075067024128686
    Split on feature home_ownership.OWN. (598, 148)
    --------------------------------------------------------------------
    Subtree, depth = 13 (598 data points).
    left errors: 0
    right errors: 382
    error before: 0.3612040133779264
    error after: 0.6387959866220736
    Split on feature home_ownership.RENT. (0, 598)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (598 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    left errors: 53
    right errors: 4
    error before: 0.40540540540540543
    error after: 0.38513513513513514
    Split on feature emp_length.< 1 year. (137, 11)
    --------------------------------------------------------------------
    Subtree, depth = 14 (137 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (57 data points).
    left errors: 30
    right errors: 0
    error before: 0.47368421052631576
    error after: 0.5263157894736842
    Split on feature home_ownership.OTHER. (57, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (57 data points).
    left errors: 26
    right errors: 4
    error before: 0.47368421052631576
    error after: 0.5263157894736842
    Split on feature home_ownership.OWN. (49, 8)
    --------------------------------------------------------------------
    Subtree, depth = 13 (49 data points).
    left errors: 0
    right errors: 23
    error before: 0.46938775510204084
    error after: 0.46938775510204084
    Split on feature home_ownership.RENT. (0, 49)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (49 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    left errors: 4
    right errors: 0
    error before: 0.5
    error after: 0.5
    Split on feature home_ownership.RENT. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (8 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1387 data points).
    left errors: 570
    right errors: 36
    error before: 0.4383561643835616
    error after: 0.43691420331651043
    Split on feature emp_length.6 years. (1313, 74)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1313 data points).
    left errors: 570
    right errors: 0
    error before: 0.43412033511043413
    error after: 0.43412033511043413
    Split on feature home_ownership.OTHER. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1313 data points).
    left errors: 570
    right errors: 0
    error before: 0.43412033511043413
    error after: 0.43412033511043413
    Split on feature home_ownership.OWN. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1313 data points).
    left errors: 570
    right errors: 0
    error before: 0.43412033511043413
    error after: 0.43412033511043413
    Split on feature home_ownership.RENT. (1313, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1313 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (74 data points).
    left errors: 38
    right errors: 0
    error before: 0.4864864864864865
    error after: 0.5135135135135135
    Split on feature home_ownership.OTHER. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (74 data points).
    left errors: 38
    right errors: 0
    error before: 0.4864864864864865
    error after: 0.5135135135135135
    Split on feature home_ownership.OWN. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (74 data points).
    left errors: 38
    right errors: 0
    error before: 0.4864864864864865
    error after: 0.5135135135135135
    Split on feature home_ownership.RENT. (74, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (74 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    left errors: 420
    right errors: 38
    error before: 0.4398854961832061
    error after: 0.43702290076335876
    Split on feature emp_length.5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.E. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.F. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.G. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (969 data points).
    left errors: 0
    right errors: 549
    error before: 0.43343653250773995
    error after: 0.56656346749226
    Split on feature term. 60 months. (0, 969)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (969 data points).
    left errors: 146
    right errors: 328
    error before: 0.43343653250773995
    error after: 0.4891640866873065
    Split on feature home_ownership.MORTGAGE. (367, 602)
    --------------------------------------------------------------------
    Subtree, depth = 11 (367 data points).
    left errors: 146
    right errors: 0
    error before: 0.3978201634877384
    error after: 0.3978201634877384
    Split on feature home_ownership.OTHER. (367, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (367 data points).
    left errors: 120
    right errors: 50
    error before: 0.3978201634877384
    error after: 0.46321525885558584
    Split on feature home_ownership.OWN. (291, 76)
    --------------------------------------------------------------------
    Subtree, depth = 13 (291 data points).
    left errors: 0
    right errors: 171
    error before: 0.41237113402061853
    error after: 0.5876288659793815
    Split on feature home_ownership.RENT. (0, 291)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (291 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (76 data points).
    left errors: 23
    right errors: 2
    error before: 0.34210526315789475
    error after: 0.32894736842105265
    Split on feature emp_length.9 years. (71, 5)
    --------------------------------------------------------------------
    Subtree, depth = 14 (71 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (602 data points).
    left errors: 261
    right errors: 9
    error before: 0.45514950166112955
    error after: 0.4485049833887043
    Split on feature emp_length.9 years. (580, 22)
    --------------------------------------------------------------------
    Subtree, depth = 12 (580 data points).
    left errors: 243
    right errors: 17
    error before: 0.45
    error after: 0.4482758620689655
    Split on feature emp_length.3 years. (545, 35)
    --------------------------------------------------------------------
    Subtree, depth = 13 (545 data points).
    left errors: 223
    right errors: 19
    error before: 0.44587155963302755
    error after: 0.44403669724770645
    Split on feature emp_length.4 years. (506, 39)
    --------------------------------------------------------------------
    Subtree, depth = 14 (506 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (39 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (35 data points).
    left errors: 18
    right errors: 0
    error before: 0.4857142857142857
    error after: 0.5142857142857142
    Split on feature home_ownership.OTHER. (35, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (35 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (22 data points).
    left errors: 13
    right errors: 0
    error before: 0.4090909090909091
    error after: 0.5909090909090909
    Split on feature home_ownership.OTHER. (22, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (22 data points).
    left errors: 13
    right errors: 0
    error before: 0.4090909090909091
    error after: 0.5909090909090909
    Split on feature home_ownership.OWN. (22, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (22 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    left errors: 19
    right errors: 23
    error before: 0.4810126582278481
    error after: 0.5316455696202531
    Split on feature home_ownership.MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.D. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.E. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.F. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.G. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (34 data points).
    left errors: 0
    right errors: 15
    error before: 0.4411764705882353
    error after: 0.4411764705882353
    Split on feature term. 60 months. (0, 34)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature home_ownership.OTHER. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (34 data points).
    left errors: 14
    right errors: 4
    error before: 0.4411764705882353
    error after: 0.5294117647058824
    Split on feature home_ownership.OWN. (25, 9)
    --------------------------------------------------------------------
    Subtree, depth = 13 (25 data points).
    left errors: 0
    right errors: 11
    error before: 0.44
    error after: 0.44
    Split on feature home_ownership.RENT. (0, 25)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (25 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (9 data points).
    left errors: 5
    right errors: 0
    error before: 0.4444444444444444
    error after: 0.5555555555555556
    Split on feature home_ownership.RENT. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.D. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.E. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.F. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.G. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (45 data points).
    left errors: 0
    right errors: 23
    error before: 0.4888888888888889
    error after: 0.5111111111111111
    Split on feature term. 60 months. (0, 45)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature home_ownership.OTHER. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature home_ownership.OWN. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature home_ownership.RENT. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Split on feature emp_length.n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    left errors: 56
    right errors: 6
    error before: 0.3645833333333333
    error after: 0.6458333333333334
    Split on feature emp_length.< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.D. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.E. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.F. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.G. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (85 data points).
    left errors: 0
    right errors: 29
    error before: 0.3411764705882353
    error after: 0.3411764705882353
    Split on feature term. 60 months. (0, 85)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (85 data points).
    left errors: 16
    right errors: 19
    error before: 0.3411764705882353
    error after: 0.4117647058823529
    Split on feature home_ownership.MORTGAGE. (26, 59)
    --------------------------------------------------------------------
    Subtree, depth = 12 (26 data points).
    left errors: 16
    right errors: 2
    error before: 0.38461538461538464
    error after: 0.6923076923076923
    Split on feature emp_length.3 years. (24, 2)
    --------------------------------------------------------------------
    Subtree, depth = 13 (24 data points).
    left errors: 16
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.6666666666666666
    Split on feature home_ownership.OTHER. (24, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (24 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (59 data points).
    left errors: 40
    right errors: 0
    error before: 0.3220338983050847
    error after: 0.6779661016949152
    Split on feature home_ownership.OTHER. (59, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (59 data points).
    left errors: 40
    right errors: 0
    error before: 0.3220338983050847
    error after: 0.6779661016949152
    Split on feature home_ownership.OWN. (59, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (59 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.D. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.E. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.F. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.G. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (11 data points).
    left errors: 0
    right errors: 6
    error before: 0.45454545454545453
    error after: 0.5454545454545454
    Split on feature term. 60 months. (0, 11)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (11 data points).
    left errors: 4
    right errors: 2
    error before: 0.45454545454545453
    error after: 0.5454545454545454
    Split on feature home_ownership.MORTGAGE. (8, 3)
    --------------------------------------------------------------------
    Subtree, depth = 12 (8 data points).
    left errors: 4
    right errors: 0
    error before: 0.5
    error after: 0.5
    Split on feature home_ownership.OTHER. (8, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    left errors: 3
    right errors: 1
    error before: 0.5
    error after: 0.5
    Split on feature home_ownership.OWN. (6, 2)
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (2 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    left errors: 1
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.OTHER. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3 data points).
    left errors: 1
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.OWN. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (3 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.E. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.F. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.G. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (5 data points).
    left errors: 0
    right errors: 4
    error before: 0.2
    error after: 0.8
    Split on feature term. 60 months. (0, 5)
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (5 data points).
    left errors: 0
    right errors: 2
    error before: 0.2
    error after: 0.4
    Split on feature home_ownership.MORTGAGE. (2, 3)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (3 data points).
    left errors: 1
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.OTHER. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    left errors: 1
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.OWN. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3 data points).
    left errors: 1
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.RENT. (3, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (3 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    left errors: 8873
    right errors: 1132
    error before: 0.3923829828471751
    error after: 0.4847853474173854
    Split on feature grade.A. (15839, 4799)
    --------------------------------------------------------------------
    Subtree, depth = 7 (15839 data points).
    left errors: 8860
    right errors: 15
    error before: 0.43980049245533176
    error after: 0.5603257781425595
    Split on feature home_ownership.OTHER. (15811, 28)
    --------------------------------------------------------------------
    Subtree, depth = 8 (15811 data points).
    left errors: 3469
    right errors: 3526
    error before: 0.4396306368983619
    error after: 0.44241350958193665
    Split on feature grade.B. (6894, 8917)
    --------------------------------------------------------------------
    Subtree, depth = 9 (6894 data points).
    left errors: 1972
    right errors: 1295
    error before: 0.49680881926312737
    error after: 0.47389033942558745
    Split on feature home_ownership.MORTGAGE. (4102, 2792)
    --------------------------------------------------------------------
    Subtree, depth = 10 (4102 data points).
    left errors: 1792
    right errors: 154
    error before: 0.4807411019015115
    error after: 0.47440273037542663
    Split on feature emp_length.4 years. (3768, 334)
    --------------------------------------------------------------------
    Subtree, depth = 11 (3768 data points).
    left errors: 1726
    right errors: 63
    error before: 0.47558386411889597
    error after: 0.4747876857749469
    Split on feature emp_length.9 years. (3639, 129)
    --------------------------------------------------------------------
    Subtree, depth = 12 (3639 data points).
    left errors: 1467
    right errors: 257
    error before: 0.47430612805715855
    error after: 0.47375652651827427
    Split on feature emp_length.2 years. (3123, 516)
    --------------------------------------------------------------------
    Subtree, depth = 13 (3123 data points).
    left errors: 0
    right errors: 1656
    error before: 0.4697406340057637
    error after: 0.5302593659942363
    Split on feature grade.C. (0, 3123)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (3123 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (516 data points).
    left errors: 227
    right errors: 26
    error before: 0.49806201550387597
    error after: 0.4903100775193798
    Split on feature home_ownership.OWN. (458, 58)
    --------------------------------------------------------------------
    Subtree, depth = 14 (458 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (58 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (129 data points).
    left errors: 55
    right errors: 5
    error before: 0.4883720930232558
    error after: 0.46511627906976744
    Split on feature home_ownership.OWN. (113, 16)
    --------------------------------------------------------------------
    Subtree, depth = 13 (113 data points).
    left errors: 0
    right errors: 58
    error before: 0.48672566371681414
    error after: 0.5132743362831859
    Split on feature grade.C. (0, 113)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (113 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (16 data points).
    left errors: 0
    right errors: 5
    error before: 0.3125
    error after: 0.3125
    Split on feature grade.C. (0, 16)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (16 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 11 (334 data points).
    left errors: 0
    right errors: 154
    error before: 0.46107784431137727
    error after: 0.46107784431137727
    Split on feature grade.C. (0, 334)
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (334 data points).
    left errors: 180
    right errors: 0
    error before: 0.46107784431137727
    error after: 0.5389221556886228
    Split on feature term. 60 months. (334, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (334 data points).
    left errors: 155
    right errors: 23
    error before: 0.46107784431137727
    error after: 0.5329341317365269
    Split on feature home_ownership.OWN. (286, 48)
    --------------------------------------------------------------------
    Subtree, depth = 14 (286 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (48 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (2792 data points).
    left errors: 1383
    right errors: 116
    error before: 0.46382521489971346
    error after: 0.53689111747851
    Split on feature emp_length.2 years. (2562, 230)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2562 data points).
    left errors: 1270
    right errors: 114
    error before: 0.4601873536299766
    error after: 0.5402029664324747
    Split on feature emp_length.5 years. (2335, 227)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2335 data points).
    left errors: 0
    right errors: 1065
    error before: 0.45610278372591007
    error after: 0.45610278372591007
    Split on feature grade.C. (0, 2335)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2335 data points).
    left errors: 1270
    right errors: 0
    error before: 0.45610278372591007
    error after: 0.5438972162740899
    Split on feature term. 60 months. (2335, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2335 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (227 data points).
    left errors: 0
    right errors: 114
    error before: 0.4977973568281938
    error after: 0.5022026431718062
    Split on feature grade.C. (0, 227)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (227 data points).
    left errors: 113
    right errors: 0
    error before: 0.4977973568281938
    error after: 0.4977973568281938
    Split on feature term. 60 months. (227, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (227 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (230 data points).
    left errors: 0
    right errors: 116
    error before: 0.4956521739130435
    error after: 0.5043478260869565
    Split on feature grade.C. (0, 230)
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (230 data points).
    left errors: 114
    right errors: 0
    error before: 0.4956521739130435
    error after: 0.4956521739130435
    Split on feature term. 60 months. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (230 data points).
    left errors: 114
    right errors: 0
    error before: 0.4956521739130435
    error after: 0.4956521739130435
    Split on feature home_ownership.OWN. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (8917 data points).
    left errors: 5391
    right errors: 0
    error before: 0.3954244701132668
    error after: 0.6045755298867332
    Split on feature grade.C. (8917, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (8917 data points).
    left errors: 5391
    right errors: 0
    error before: 0.3954244701132668
    error after: 0.6045755298867332
    Split on feature term. 60 months. (8917, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (8917 data points).
    left errors: 2782
    right errors: 1560
    error before: 0.3954244701132668
    error after: 0.4869350678479309
    Split on feature home_ownership.MORTGAGE. (4748, 4169)
    --------------------------------------------------------------------
    Subtree, depth = 12 (4748 data points).
    left errors: 2368
    right errors: 245
    error before: 0.4140690817186184
    error after: 0.5503369839932604
    Split on feature home_ownership.OWN. (4089, 659)
    --------------------------------------------------------------------
    Subtree, depth = 13 (4089 data points).
    left errors: 0
    right errors: 1721
    error before: 0.42088530202983615
    error after: 0.42088530202983615
    Split on feature home_ownership.RENT. (0, 4089)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (4089 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (659 data points).
    left errors: 414
    right errors: 0
    error before: 0.37177541729893776
    error after: 0.6282245827010622
    Split on feature home_ownership.RENT. (659, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (659 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (4169 data points).
    left errors: 2609
    right errors: 0
    error before: 0.37419045334612616
    error after: 0.6258095466538738
    Split on feature home_ownership.OWN. (4169, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (4169 data points).
    left errors: 2609
    right errors: 0
    error before: 0.37419045334612616
    error after: 0.6258095466538738
    Split on feature home_ownership.RENT. (4169, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (4169 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (28 data points).
    left errors: 7
    right errors: 11
    error before: 0.4642857142857143
    error after: 0.6428571428571429
    Split on feature grade.B. (11, 17)
    --------------------------------------------------------------------
    Subtree, depth = 9 (11 data points).
    left errors: 7
    right errors: 1
    error before: 0.36363636363636365
    error after: 0.7272727272727273
    Split on feature emp_length.6 years. (10, 1)
    --------------------------------------------------------------------
    Subtree, depth = 10 (10 data points).
    left errors: 0
    right errors: 3
    error before: 0.3
    error after: 0.3
    Split on feature grade.C. (0, 10)
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (10 data points).
    left errors: 7
    right errors: 0
    error before: 0.3
    error after: 0.7
    Split on feature term. 60 months. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (10 data points).
    left errors: 7
    right errors: 0
    error before: 0.3
    error after: 0.7
    Split on feature home_ownership.MORTGAGE. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (10 data points).
    left errors: 7
    right errors: 0
    error before: 0.3
    error after: 0.7
    Split on feature home_ownership.OWN. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (10 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (17 data points).
    left errors: 5
    right errors: 0
    error before: 0.35294117647058826
    error after: 0.29411764705882354
    Split on feature emp_length.1 year. (16, 1)
    --------------------------------------------------------------------
    Subtree, depth = 10 (16 data points).
    left errors: 4
    right errors: 0
    error before: 0.3125
    error after: 0.25
    Split on feature emp_length.3 years. (15, 1)
    --------------------------------------------------------------------
    Subtree, depth = 11 (15 data points).
    left errors: 3
    right errors: 0
    error before: 0.26666666666666666
    error after: 0.2
    Split on feature emp_length.4 years. (14, 1)
    --------------------------------------------------------------------
    Subtree, depth = 12 (14 data points).
    left errors: 2
    right errors: 0
    error before: 0.21428571428571427
    error after: 0.14285714285714285
    Split on feature emp_length.< 1 year. (13, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (13 data points).
    left errors: 2
    right errors: 0
    error before: 0.15384615384615385
    error after: 0.15384615384615385
    Split on feature grade.C. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (13 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (4799 data points).
    left errors: 3667
    right errors: 0
    error before: 0.23588247551573244
    error after: 0.7641175244842675
    Split on feature grade.B. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (4799 data points).
    left errors: 3667
    right errors: 0
    error before: 0.23588247551573244
    error after: 0.7641175244842675
    Split on feature grade.C. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (4799 data points).
    left errors: 3667
    right errors: 0
    error before: 0.23588247551573244
    error after: 0.7641175244842675
    Split on feature term. 60 months. (4799, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (4799 data points).
    left errors: 1568
    right errors: 537
    error before: 0.23588247551573244
    error after: 0.43863304855178165
    Split on feature home_ownership.MORTGAGE. (2163, 2636)
    --------------------------------------------------------------------
    Subtree, depth = 11 (2163 data points).
    left errors: 1560
    right errors: 1
    error before: 0.2750809061488673
    error after: 0.7216828478964401
    Split on feature home_ownership.OTHER. (2154, 9)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2154 data points).
    left errors: 1255
    right errors: 96
    error before: 0.2757660167130919
    error after: 0.627205199628598
    Split on feature home_ownership.OWN. (1753, 401)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1753 data points).
    left errors: 0
    right errors: 498
    error before: 0.2840844266970907
    error after: 0.2840844266970907
    Split on feature home_ownership.RENT. (0, 1753)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1753 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (401 data points).
    left errors: 305
    right errors: 0
    error before: 0.23940149625935161
    error after: 0.7605985037406484
    Split on feature home_ownership.RENT. (401, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (401 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (9 data points).
    left errors: 8
    right errors: 1
    error before: 0.1111111111111111
    error after: 1.0
    Split on feature emp_length.3 years. (8, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (8 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (2636 data points).
    left errors: 2099
    right errors: 0
    error before: 0.20371775417298937
    error after: 0.7962822458270106
    Split on feature home_ownership.OTHER. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2636 data points).
    left errors: 2099
    right errors: 0
    error before: 0.20371775417298937
    error after: 0.7962822458270106
    Split on feature home_ownership.OWN. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (2636 data points).
    left errors: 2099
    right errors: 0
    error before: 0.20371775417298937
    error after: 0.7962822458270106
    Split on feature home_ownership.RENT. (2636, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2636 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    left errors: 26
    right errors: 0
    error before: 0.2708333333333333
    error after: 0.2708333333333333
    Split on feature grade.A. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (96 data points).
    left errors: 26
    right errors: 0
    error before: 0.2708333333333333
    error after: 0.2708333333333333
    Split on feature grade.B. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (96 data points).
    left errors: 26
    right errors: 0
    error before: 0.2708333333333333
    error after: 0.2708333333333333
    Split on feature grade.C. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (96 data points).
    left errors: 26
    right errors: 0
    error before: 0.2708333333333333
    error after: 0.2708333333333333
    Split on feature term. 60 months. (96, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (96 data points).
    left errors: 14
    right errors: 40
    error before: 0.2708333333333333
    error after: 0.5625
    Split on feature home_ownership.MORTGAGE. (44, 52)
    --------------------------------------------------------------------
    Subtree, depth = 11 (44 data points).
    left errors: 13
    right errors: 0
    error before: 0.3181818181818182
    error after: 0.29545454545454547
    Split on feature emp_length.3 years. (43, 1)
    --------------------------------------------------------------------
    Subtree, depth = 12 (43 data points).
    left errors: 12
    right errors: 0
    error before: 0.3023255813953488
    error after: 0.27906976744186046
    Split on feature emp_length.7 years. (42, 1)
    --------------------------------------------------------------------
    Subtree, depth = 13 (42 data points).
    left errors: 11
    right errors: 0
    error before: 0.2857142857142857
    error after: 0.2619047619047619
    Split on feature emp_length.8 years. (41, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (41 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (52 data points).
    left errors: 9
    right errors: 2
    error before: 0.23076923076923078
    error after: 0.21153846153846154
    Split on feature emp_length.2 years. (47, 5)
    --------------------------------------------------------------------
    Subtree, depth = 12 (47 data points).
    left errors: 9
    right errors: 0
    error before: 0.19148936170212766
    error after: 0.19148936170212766
    Split on feature home_ownership.OTHER. (47, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (47 data points).
    left errors: 9
    right errors: 0
    error before: 0.19148936170212766
    error after: 0.19148936170212766
    Split on feature home_ownership.OWN. (47, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (47 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (5 data points).
    left errors: 3
    right errors: 0
    error before: 0.4
    error after: 0.6
    Split on feature home_ownership.OTHER. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (5 data points).
    left errors: 3
    right errors: 0
    error before: 0.4
    error after: 0.6
    Split on feature home_ownership.OWN. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    left errors: 273
    right errors: 87
    error before: 0.44635193133047213
    error after: 0.38626609442060084
    Split on feature grade.A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    left errors: 272
    right errors: 0
    error before: 0.3888888888888889
    error after: 0.38746438746438744
    Split on feature home_ownership.OTHER. (701, 1)
    --------------------------------------------------------------------
    Subtree, depth = 7 (701 data points).
    left errors: 107
    right errors: 219
    error before: 0.3880171184022825
    error after: 0.46504992867332384
    Split on feature grade.B. (317, 384)
    --------------------------------------------------------------------
    Subtree, depth = 8 (317 data points).
    left errors: 0
    right errors: 209
    error before: 0.33753943217665616
    error after: 0.6593059936908517
    Split on feature grade.C. (1, 316)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (316 data points).
    left errors: 107
    right errors: 0
    error before: 0.33860759493670883
    error after: 0.33860759493670883
    Split on feature grade.G. (316, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (316 data points).
    left errors: 107
    right errors: 0
    error before: 0.33860759493670883
    error after: 0.33860759493670883
    Split on feature term. 60 months. (316, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (316 data points).
    left errors: 66
    right errors: 86
    error before: 0.33860759493670883
    error after: 0.4810126582278481
    Split on feature home_ownership.MORTGAGE. (189, 127)
    --------------------------------------------------------------------
    Subtree, depth = 12 (189 data points).
    left errors: 46
    right errors: 30
    error before: 0.3492063492063492
    error after: 0.4021164021164021
    Split on feature home_ownership.OWN. (139, 50)
    --------------------------------------------------------------------
    Subtree, depth = 13 (139 data points).
    left errors: 0
    right errors: 93
    error before: 0.33093525179856115
    error after: 0.6690647482014388
    Split on feature home_ownership.RENT. (0, 139)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (139 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (50 data points).
    left errors: 20
    right errors: 0
    error before: 0.4
    error after: 0.4
    Split on feature home_ownership.RENT. (50, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (50 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (127 data points).
    left errors: 41
    right errors: 0
    error before: 0.3228346456692913
    error after: 0.3228346456692913
    Split on feature home_ownership.OWN. (127, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (127 data points).
    left errors: 41
    right errors: 0
    error before: 0.3228346456692913
    error after: 0.3228346456692913
    Split on feature home_ownership.RENT. (127, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (127 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (384 data points).
    left errors: 165
    right errors: 0
    error before: 0.4296875
    error after: 0.4296875
    Split on feature grade.C. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (384 data points).
    left errors: 165
    right errors: 0
    error before: 0.4296875
    error after: 0.4296875
    Split on feature grade.G. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (384 data points).
    left errors: 165
    right errors: 0
    error before: 0.4296875
    error after: 0.4296875
    Split on feature term. 60 months. (384, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (384 data points).
    left errors: 78
    right errors: 87
    error before: 0.4296875
    error after: 0.4296875
    Split on feature home_ownership.MORTGAGE. (210, 174)
    --------------------------------------------------------------------
    Subtree, depth = 12 (210 data points).
    left errors: 53
    right errors: 37
    error before: 0.37142857142857144
    error after: 0.42857142857142855
    Split on feature home_ownership.OWN. (148, 62)
    --------------------------------------------------------------------
    Subtree, depth = 13 (148 data points).
    left errors: 0
    right errors: 95
    error before: 0.3581081081081081
    error after: 0.6418918918918919
    Split on feature home_ownership.RENT. (0, 148)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (148 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (62 data points).
    left errors: 25
    right errors: 0
    error before: 0.4032258064516129
    error after: 0.4032258064516129
    Split on feature home_ownership.RENT. (62, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (62 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (174 data points).
    left errors: 87
    right errors: 0
    error before: 0.5
    error after: 0.5
    Split on feature home_ownership.OWN. (174, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (174 data points).
    left errors: 87
    right errors: 0
    error before: 0.5
    error after: 0.5
    Split on feature home_ownership.RENT. (174, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (174 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    left errors: 143
    right errors: 0
    error before: 0.3782608695652174
    error after: 0.6217391304347826
    Split on feature grade.B. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (230 data points).
    left errors: 143
    right errors: 0
    error before: 0.3782608695652174
    error after: 0.6217391304347826
    Split on feature grade.C. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (230 data points).
    left errors: 143
    right errors: 0
    error before: 0.3782608695652174
    error after: 0.6217391304347826
    Split on feature grade.G. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (230 data points).
    left errors: 143
    right errors: 0
    error before: 0.3782608695652174
    error after: 0.6217391304347826
    Split on feature term. 60 months. (230, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (230 data points).
    left errors: 79
    right errors: 47
    error before: 0.3782608695652174
    error after: 0.5478260869565217
    Split on feature home_ownership.MORTGAGE. (119, 111)
    --------------------------------------------------------------------
    Subtree, depth = 11 (119 data points).
    left errors: 79
    right errors: 0
    error before: 0.33613445378151263
    error after: 0.6638655462184874
    Split on feature home_ownership.OTHER. (119, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (119 data points).
    left errors: 44
    right errors: 13
    error before: 0.33613445378151263
    error after: 0.4789915966386555
    Split on feature home_ownership.OWN. (71, 48)
    --------------------------------------------------------------------
    Subtree, depth = 13 (71 data points).
    left errors: 0
    right errors: 27
    error before: 0.38028169014084506
    error after: 0.38028169014084506
    Split on feature home_ownership.RENT. (0, 71)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (71 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (48 data points).
    left errors: 35
    right errors: 0
    error before: 0.2708333333333333
    error after: 0.7291666666666666
    Split on feature home_ownership.RENT. (48, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (48 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (111 data points).
    left errors: 64
    right errors: 0
    error before: 0.42342342342342343
    error after: 0.5765765765765766
    Split on feature home_ownership.OTHER. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (111 data points).
    left errors: 64
    right errors: 0
    error before: 0.42342342342342343
    error after: 0.5765765765765766
    Split on feature home_ownership.OWN. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (111 data points).
    left errors: 64
    right errors: 0
    error before: 0.42342342342342343
    error after: 0.5765765765765766
    Split on feature home_ownership.RENT. (111, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (111 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    left errors: 101
    right errors: 4
    error before: 0.3016759776536313
    error after: 0.29329608938547486
    Split on feature emp_length.8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.B. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.C. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.G. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature term. 60 months. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (347 data points).
    left errors: 72
    right errors: 81
    error before: 0.2910662824207493
    error after: 0.4409221902017291
    Split on feature home_ownership.MORTGAGE. (237, 110)
    --------------------------------------------------------------------
    Subtree, depth = 11 (237 data points).
    left errors: 72
    right errors: 2
    error before: 0.3037974683544304
    error after: 0.31223628691983124
    Split on feature home_ownership.OTHER. (235, 2)
    --------------------------------------------------------------------
    Subtree, depth = 12 (235 data points).
    left errors: 64
    right errors: 24
    error before: 0.30638297872340425
    error after: 0.37446808510638296
    Split on feature home_ownership.OWN. (203, 32)
    --------------------------------------------------------------------
    Subtree, depth = 13 (203 data points).
    left errors: 0
    right errors: 139
    error before: 0.31527093596059114
    error after: 0.6847290640394089
    Split on feature home_ownership.RENT. (0, 203)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (203 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (32 data points).
    left errors: 8
    right errors: 0
    error before: 0.25
    error after: 0.25
    Split on feature home_ownership.RENT. (32, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (32 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (110 data points).
    left errors: 29
    right errors: 0
    error before: 0.2636363636363636
    error after: 0.2636363636363636
    Split on feature home_ownership.OTHER. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (110 data points).
    left errors: 29
    right errors: 0
    error before: 0.2636363636363636
    error after: 0.2636363636363636
    Split on feature home_ownership.OWN. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (110 data points).
    left errors: 29
    right errors: 0
    error before: 0.2636363636363636
    error after: 0.2636363636363636
    Split on feature home_ownership.RENT. (110, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (110 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 7
    right errors: 2
    error before: 0.36363636363636365
    error after: 0.8181818181818182
    Split on feature home_ownership.OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    left errors: 7
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.7777777777777778
    Split on feature grade.A. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (9 data points).
    left errors: 7
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.7777777777777778
    Split on feature grade.B. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (9 data points).
    left errors: 7
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.7777777777777778
    Split on feature grade.C. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (9 data points).
    left errors: 7
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.7777777777777778
    Split on feature grade.G. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 10 (9 data points).
    left errors: 7
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.7777777777777778
    Split on feature term. 60 months. (9, 0)
    --------------------------------------------------------------------
    Subtree, depth = 11 (9 data points).
    left errors: 4
    right errors: 0
    error before: 0.2222222222222222
    error after: 0.4444444444444444
    Split on feature home_ownership.MORTGAGE. (6, 3)
    --------------------------------------------------------------------
    Subtree, depth = 12 (6 data points).
    left errors: 4
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.6666666666666666
    Split on feature home_ownership.OTHER. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (6 data points).
    left errors: 0
    right errors: 2
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.RENT. (0, 6)
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (3 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.F. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.G. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature term. 60 months. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (1276 data points).
    left errors: 301
    right errors: 245
    error before: 0.3738244514106583
    error after: 0.4278996865203762
    Split on feature home_ownership.MORTGAGE. (855, 421)
    --------------------------------------------------------------------
    Subtree, depth = 10 (855 data points).
    left errors: 299
    right errors: 4
    error before: 0.352046783625731
    error after: 0.3543859649122807
    Split on feature home_ownership.OTHER. (849, 6)
    --------------------------------------------------------------------
    Subtree, depth = 11 (849 data points).
    left errors: 260
    right errors: 73
    error before: 0.35217903415783275
    error after: 0.392226148409894
    Split on feature home_ownership.OWN. (737, 112)
    --------------------------------------------------------------------
    Subtree, depth = 12 (737 data points).
    left errors: 0
    right errors: 477
    error before: 0.35278154681139756
    error after: 0.6472184531886025
    Split on feature home_ownership.RENT. (0, 737)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (737 data points).
    left errors: 235
    right errors: 42
    error before: 0.35278154681139756
    error after: 0.3758480325644505
    Split on feature emp_length.1 year. (670, 67)
    --------------------------------------------------------------------
    Subtree, depth = 14 (670 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (67 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (112 data points).
    left errors: 39
    right errors: 0
    error before: 0.3482142857142857
    error after: 0.3482142857142857
    Split on feature home_ownership.RENT. (112, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (112 data points).
    left errors: 35
    right errors: 6
    error before: 0.3482142857142857
    error after: 0.36607142857142855
    Split on feature emp_length.1 year. (102, 10)
    --------------------------------------------------------------------
    Subtree, depth = 14 (102 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (10 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (6 data points).
    left errors: 2
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.OWN. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (6 data points).
    left errors: 2
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature home_ownership.RENT. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (6 data points).
    left errors: 2
    right errors: 0
    error before: 0.3333333333333333
    error after: 0.3333333333333333
    Split on feature emp_length.1 year. (6, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (6 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (421 data points).
    left errors: 169
    right errors: 6
    error before: 0.4180522565320665
    error after: 0.4156769596199525
    Split on feature emp_length.6 years. (408, 13)
    --------------------------------------------------------------------
    Subtree, depth = 11 (408 data points).
    left errors: 169
    right errors: 0
    error before: 0.41421568627450983
    error after: 0.41421568627450983
    Split on feature home_ownership.OTHER. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (408 data points).
    left errors: 169
    right errors: 0
    error before: 0.41421568627450983
    error after: 0.41421568627450983
    Split on feature home_ownership.OWN. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (408 data points).
    left errors: 169
    right errors: 0
    error before: 0.41421568627450983
    error after: 0.41421568627450983
    Split on feature home_ownership.RENT. (408, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (408 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (13 data points).
    left errors: 7
    right errors: 0
    error before: 0.46153846153846156
    error after: 0.5384615384615384
    Split on feature home_ownership.OTHER. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (13 data points).
    left errors: 7
    right errors: 0
    error before: 0.46153846153846156
    error after: 0.5384615384615384
    Split on feature home_ownership.OWN. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (13 data points).
    left errors: 7
    right errors: 0
    error before: 0.46153846153846156
    error after: 0.5384615384615384
    Split on feature home_ownership.RENT. (13, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (13 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.F. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 7 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.G. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 8 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature term. 60 months. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 9 (4701 data points).
    left errors: 1179
    right errors: 873
    error before: 0.4169325675388215
    error after: 0.43650287172941926
    Split on feature home_ownership.MORTGAGE. (3047, 1654)
    --------------------------------------------------------------------
    Subtree, depth = 10 (3047 data points).
    left errors: 1176
    right errors: 7
    error before: 0.3869379717755169
    error after: 0.38825073843124386
    Split on feature home_ownership.OTHER. (3037, 10)
    --------------------------------------------------------------------
    Subtree, depth = 11 (3037 data points).
    left errors: 1028
    right errors: 256
    error before: 0.3872242344418834
    error after: 0.42278564372736255
    Split on feature home_ownership.OWN. (2633, 404)
    --------------------------------------------------------------------
    Subtree, depth = 12 (2633 data points).
    left errors: 0
    right errors: 1605
    error before: 0.3904291682491455
    error after: 0.6095708317508546
    Split on feature home_ownership.RENT. (0, 2633)
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (2633 data points).
    left errors: 932
    right errors: 145
    error before: 0.3904291682491455
    error after: 0.4090391188758071
    Split on feature emp_length.1 year. (2392, 241)
    --------------------------------------------------------------------
    Subtree, depth = 14 (2392 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (241 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 12 (404 data points).
    left errors: 148
    right errors: 0
    error before: 0.36633663366336633
    error after: 0.36633663366336633
    Split on feature home_ownership.RENT. (404, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (404 data points).
    left errors: 138
    right errors: 20
    error before: 0.36633663366336633
    error after: 0.3910891089108911
    Split on feature emp_length.1 year. (374, 30)
    --------------------------------------------------------------------
    Subtree, depth = 14 (374 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (30 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (10 data points).
    left errors: 3
    right errors: 0
    error before: 0.3
    error after: 0.3
    Split on feature home_ownership.OWN. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (10 data points).
    left errors: 3
    right errors: 0
    error before: 0.3
    error after: 0.3
    Split on feature home_ownership.RENT. (10, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (10 data points).
    left errors: 3
    right errors: 1
    error before: 0.3
    error after: 0.4
    Split on feature emp_length.1 year. (9, 1)
    --------------------------------------------------------------------
    Subtree, depth = 14 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (1 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 10 (1654 data points).
    left errors: 712
    right errors: 53
    error before: 0.47218863361547764
    error after: 0.4625151148730351
    Split on feature emp_length.5 years. (1532, 122)
    --------------------------------------------------------------------
    Subtree, depth = 11 (1532 data points).
    left errors: 648
    right errors: 54
    error before: 0.46475195822454307
    error after: 0.45822454308093996
    Split on feature emp_length.3 years. (1414, 118)
    --------------------------------------------------------------------
    Subtree, depth = 12 (1414 data points).
    left errors: 613
    right errors: 28
    error before: 0.4582743988684583
    error after: 0.4533239038189533
    Split on feature emp_length.9 years. (1351, 63)
    --------------------------------------------------------------------
    Subtree, depth = 13 (1351 data points).
    left errors: 613
    right errors: 0
    error before: 0.4537379718726869
    error after: 0.4537379718726869
    Split on feature home_ownership.OTHER. (1351, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (1351 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (63 data points).
    left errors: 35
    right errors: 0
    error before: 0.4444444444444444
    error after: 0.5555555555555556
    Split on feature home_ownership.OTHER. (63, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (63 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (118 data points).
    left errors: 64
    right errors: 0
    error before: 0.4576271186440678
    error after: 0.5423728813559322
    Split on feature home_ownership.OTHER. (118, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (118 data points).
    left errors: 64
    right errors: 0
    error before: 0.4576271186440678
    error after: 0.5423728813559322
    Split on feature home_ownership.OWN. (118, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (118 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 11 (122 data points).
    left errors: 69
    right errors: 0
    error before: 0.4344262295081967
    error after: 0.5655737704918032
    Split on feature home_ownership.OTHER. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 12 (122 data points).
    left errors: 69
    right errors: 0
    error before: 0.4344262295081967
    error after: 0.5655737704918032
    Split on feature home_ownership.OWN. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 13 (122 data points).
    left errors: 69
    right errors: 0
    error before: 0.4344262295081967
    error after: 0.5655737704918032
    Split on feature home_ownership.RENT. (122, 0)
    --------------------------------------------------------------------
    Subtree, depth = 14 (122 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 14 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 13 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 12 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 9 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 8 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 7 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.


### Evaluating the models

Let us evaluate the models on the **train** and **validation** data. Let us start by evaluating the classification error on the training data:


```python
print("Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data, target))
print("Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, target))
print("Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, target))
```

    Training data, classification error (model 1): 0.40003761014399314
    Training data, classification error (model 2): 0.38185041908446166
    Training data, classification error (model 3): 0.37446271222866967


Now evaluate the classification error on the validation data.


```python
print("Validation data, classification error (model 1):", evaluate_classification_error(model_1, validation_set, target))
print("Validation data, classification error (model 2):", evaluate_classification_error(model_2, validation_set, target))
print("Validation data, classification error (model 3):", evaluate_classification_error(model_3, validation_set, target))
```

    Validation data, classification error (model 1): 0.3981042654028436
    Validation data, classification error (model 2): 0.3837785437311504
    Validation data, classification error (model 3): 0.38000861697544164


**Quiz Question:** Which tree has the smallest error on the validation data? 3

**Quiz Question:** Does the tree with the smallest error in the training data also have the smallest error in the validation data? yes

**Quiz Question:** Is it always true that the tree with the lowest classification error on the **training** set will result in the lowest classification error in the **validation** set? no


### Measuring the complexity of the tree

Recall in the lecture that we talked about deeper trees being more complex. We will measure the complexity of the tree as

```
  complexity(T) = number of leaves in the tree T
```

Here, we provide a function `count_leaves` that counts the number of leaves in a tree. Using this implementation, compute the number of nodes in `model_1`, `model_2`, and `model_3`. 


```python
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])
```

Compute the number of nodes in `model_1`, `model_2`, and `model_3`.


```python
print("Model Complexity (model 1):", count_leaves(model_1))
print("Model Complexity (model 2):", count_leaves(model_2))
print("Model Complexity (model 3):", count_leaves(model_3))
```

    Model Complexity (model 1): 4
    Model Complexity (model 2): 41
    Model Complexity (model 3): 341


**Quiz Question:** Which tree has the largest complexity? 3

**Quiz Question:** Is it always true that the most complex tree will result in the lowest classification error in the **validation_set**? no

# Exploring the effect of min_error

We will compare three models trained with different values of the stopping criterion. We intentionally picked models at the extreme ends (**negative**, **just right**, and **too positive**).

Train three models with these parameters:
1. **model_4**: `min_error_reduction = -1` (ignoring this early stopping condition)
2. **model_5**: `min_error_reduction = 0` (just right)
3. **model_6**: `min_error_reduction = 5` (too positive)

For each of these three, we set `max_depth = 6`, and `min_node_size = 0`.

** Note:** Each tree can take up to 30 seconds to train.


```python
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction = -1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction = 0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction = 5)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    left errors: 420
    right errors: 38
    error before: 0.4398854961832061
    error after: 0.43702290076335876
    Split on feature emp_length.5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    left errors: 19
    right errors: 23
    error before: 0.4810126582278481
    error after: 0.5316455696202531
    Split on feature home_ownership.MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Split on feature emp_length.n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    left errors: 56
    right errors: 6
    error before: 0.3645833333333333
    error after: 0.6458333333333334
    Split on feature emp_length.< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    left errors: 273
    right errors: 87
    error before: 0.44635193133047213
    error after: 0.38626609442060084
    Split on feature grade.A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    left errors: 101
    right errors: 4
    error before: 0.3016759776536313
    error after: 0.29329608938547486
    Split on feature emp_length.8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 7
    right errors: 2
    error before: 0.36363636363636365
    error after: 0.8181818181818182
    Split on feature home_ownership.OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Early stopping condition 3 reached. Minimum error reduction.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Early stopping condition 3 reached. Minimum error reduction.


Calculate the accuracy of each model (**model_4**, **model_5**, or **model_6**) on the validation set. 


```python
print("Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set, target))
print("Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set, target))
print("Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set, target))
```

    Validation data, classification error (model 4): 0.3837785437311504
    Validation data, classification error (model 5): 0.4226626454114606
    Validation data, classification error (model 6): 0.503446790176648


Using the `count_leaves` function, compute the number of leaves in each of each models in (**model_4**, **model_5**, and **model_6**). 


```python
print("Model Complexity (model 4):", count_leaves(model_4))
print("Model Complexity (model 5):", count_leaves(model_5))
print("Model Complexity (model 6):", count_leaves(model_6))
```

    Model Complexity (model 4): 41
    Model Complexity (model 5): 3
    Model Complexity (model 6): 1


**Quiz Question:** Using the complexity definition above, which model (**model_4**, **model_5**, or **model_6**) has the largest complexity? 4

Did this match your expectation?

**Quiz Question:** **model_4** and **model_5** have similar classification error on the validation set but **model_5** has lower complexity. Should you pick **model_5** over **model_4**? 5


# Exploring the effect of min_node_size

We will compare three models trained with different values of the stopping criterion. Again, intentionally picked models at the extreme ends (**too small**, **just right**, and **just right**).

Train three models with these parameters:
1. **model_7**: min_node_size = 0 (too small)
2. **model_8**: min_node_size = 2000 (just right)
3. **model_9**: min_node_size = 50000 (too large)

For each of these three, we set `max_depth = 6`, and `min_error_reduction = -1`.

** Note:** Each tree can take up to 30 seconds to train.


```python
model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 0, min_error_reduction = -1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 2000, min_error_reduction = -1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, 
                                min_node_size = 50000, min_error_reduction = -1)
```

    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    left errors: 420
    right errors: 38
    error before: 0.4398854961832061
    error after: 0.43702290076335876
    Split on feature emp_length.5 years. (969, 79)
    --------------------------------------------------------------------
    Subtree, depth = 4 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.C. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (969 data points).
    left errors: 420
    right errors: 0
    error before: 0.43343653250773995
    error after: 0.43343653250773995
    Split on feature grade.D. (969, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (969 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (79 data points).
    left errors: 19
    right errors: 23
    error before: 0.4810126582278481
    error after: 0.5316455696202531
    Split on feature home_ownership.MORTGAGE. (34, 45)
    --------------------------------------------------------------------
    Subtree, depth = 5 (34 data points).
    left errors: 19
    right errors: 0
    error before: 0.4411764705882353
    error after: 0.5588235294117647
    Split on feature grade.C. (34, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (34 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (45 data points).
    left errors: 22
    right errors: 0
    error before: 0.4888888888888889
    error after: 0.4888888888888889
    Split on feature grade.C. (45, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (45 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    left errors: 61
    right errors: 4
    error before: 0.38613861386138615
    error after: 0.6435643564356436
    Split on feature emp_length.n/a. (96, 5)
    --------------------------------------------------------------------
    Subtree, depth = 3 (96 data points).
    left errors: 56
    right errors: 6
    error before: 0.3645833333333333
    error after: 0.6458333333333334
    Split on feature emp_length.< 1 year. (85, 11)
    --------------------------------------------------------------------
    Subtree, depth = 4 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.B. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (85 data points).
    left errors: 56
    right errors: 0
    error before: 0.3411764705882353
    error after: 0.6588235294117647
    Split on feature grade.C. (85, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (85 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.B. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 5
    right errors: 0
    error before: 0.45454545454545453
    error after: 0.45454545454545453
    Split on feature grade.C. (11, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (11 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.B. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.C. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (5 data points).
    left errors: 1
    right errors: 0
    error before: 0.2
    error after: 0.2
    Split on feature grade.D. (5, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (5 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    left errors: 273
    right errors: 87
    error before: 0.44635193133047213
    error after: 0.38626609442060084
    Split on feature grade.A. (702, 230)
    --------------------------------------------------------------------
    Subtree, depth = 6 (702 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (230 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    left errors: 101
    right errors: 4
    error before: 0.3016759776536313
    error after: 0.29329608938547486
    Split on feature emp_length.8 years. (347, 11)
    --------------------------------------------------------------------
    Subtree, depth = 5 (347 data points).
    left errors: 101
    right errors: 0
    error before: 0.2910662824207493
    error after: 0.2910662824207493
    Split on feature grade.A. (347, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (347 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (11 data points).
    left errors: 7
    right errors: 2
    error before: 0.36363636363636365
    error after: 0.8181818181818182
    Split on feature home_ownership.OWN. (9, 2)
    --------------------------------------------------------------------
    Subtree, depth = 6 (9 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.A. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.B. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (1276 data points).
    left errors: 477
    right errors: 0
    error before: 0.3738244514106583
    error after: 0.3738244514106583
    Split on feature grade.C. (1276, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1276 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    left errors: 3221
    right errors: 12474
    error before: 0.4963464431549538
    error after: 0.4216365785514722
    Split on feature term. 36 months. (9223, 28001)
    --------------------------------------------------------------------
    Subtree, depth = 1 (9223 data points).
    left errors: 3159
    right errors: 39
    error before: 0.34923560663558495
    error after: 0.34674184104955
    Split on feature grade.A. (9122, 101)
    --------------------------------------------------------------------
    Subtree, depth = 2 (9122 data points).
    left errors: 2698
    right errors: 587
    error before: 0.34630563472922604
    error after: 0.36011839508879634
    Split on feature grade.B. (8074, 1048)
    --------------------------------------------------------------------
    Subtree, depth = 3 (8074 data points).
    left errors: 1784
    right errors: 1276
    error before: 0.33415902898191724
    error after: 0.37899430270002477
    Split on feature grade.C. (5884, 2190)
    --------------------------------------------------------------------
    Subtree, depth = 4 (5884 data points).
    left errors: 1061
    right errors: 1335
    error before: 0.30319510537049627
    error after: 0.40720598232494903
    Split on feature grade.D. (3826, 2058)
    --------------------------------------------------------------------
    Subtree, depth = 5 (3826 data points).
    left errors: 437
    right errors: 1509
    error before: 0.2773131207527444
    error after: 0.5086251960271825
    Split on feature grade.E. (1693, 2133)
    --------------------------------------------------------------------
    Subtree, depth = 6 (1693 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (2133 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (2058 data points).
    left errors: 723
    right errors: 0
    error before: 0.35131195335276966
    error after: 0.35131195335276966
    Split on feature grade.E. (2058, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2058 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.D. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (2190 data points).
    left errors: 914
    right errors: 0
    error before: 0.41735159817351597
    error after: 0.41735159817351597
    Split on feature grade.E. (2190, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (2190 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1048 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 2 (101 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 1 (28001 data points).
    left errors: 13567
    right errors: 2741
    error before: 0.4454840898539338
    error after: 0.5824077711510304
    Split on feature grade.D. (23300, 4701)
    --------------------------------------------------------------------
    Subtree, depth = 2 (23300 data points).
    left errors: 13090
    right errors: 799
    error before: 0.417725321888412
    error after: 0.5960944206008584
    Split on feature grade.E. (22024, 1276)
    --------------------------------------------------------------------
    Subtree, depth = 3 (22024 data points).
    left errors: 12982
    right errors: 250
    error before: 0.4056483835815474
    error after: 0.6007991282237559
    Split on feature grade.F. (21666, 358)
    --------------------------------------------------------------------
    Subtree, depth = 4 (21666 data points).
    left errors: 12566
    right errors: 516
    error before: 0.4008123326871596
    error after: 0.603803193944429
    Split on feature emp_length.n/a. (20734, 932)
    --------------------------------------------------------------------
    Subtree, depth = 5 (20734 data points).
    left errors: 12540
    right errors: 70
    error before: 0.3939423169673001
    error after: 0.608179801292563
    Split on feature grade.G. (20638, 96)
    --------------------------------------------------------------------
    Subtree, depth = 6 (20638 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (96 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 5 (932 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 4 (358 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 3 (1276 data points).
    Early stopping condition 2 reached. Reached minimum node size.
    --------------------------------------------------------------------
    Subtree, depth = 2 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.A. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 3 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.B. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 4 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.C. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 5 (4701 data points).
    left errors: 1960
    right errors: 0
    error before: 0.4169325675388215
    error after: 0.4169325675388215
    Split on feature grade.E. (4701, 0)
    --------------------------------------------------------------------
    Subtree, depth = 6 (4701 data points).
    Early stopping condition 1 reached. Reached maximum depth.
    --------------------------------------------------------------------
    Subtree, depth = 6 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 5 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 4 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 3 (0 data points).
    Stopping condition 1 reached. All data points have the same target value.
    --------------------------------------------------------------------
    Subtree, depth = 0 (37224 data points).
    Early stopping condition 2 reached. Reached minimum node size.


Now, let us evaluate the models (**model_7**, **model_8**, or **model_9**) on the **validation_set**.


```python
print("Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_set, target))
print("Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_set, target))
print("Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_set, target))
```

    Validation data, classification error (model 7): 0.3837785437311504
    Validation data, classification error (model 8): 0.38453252908229213
    Validation data, classification error (model 9): 0.503446790176648


Using the `count_leaves` function, compute the number of leaves in each of each models (**model_7**, **model_8**, and **model_9**). 


```python
print("Model Complexity (model 7):", count_leaves(model_7))
print("Model Complexity (model 8):", count_leaves(model_8))
print("Model Complexity (model 9):", count_leaves(model_9))
```

    Model Complexity (model 7): 41
    Model Complexity (model 8): 19
    Model Complexity (model 9): 1


**Quiz Question:** Using the results obtained in this section, which model (**model_7**, **model_8**, or **model_9**) would you choose to use? 8

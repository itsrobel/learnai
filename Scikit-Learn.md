---
title: "Scikit-learn"
author: [Robel A.E. Schwarz]
date: "2017-02-20"
titlepage: true
titlepage-background: "`~/.local/share/pandoc/backgrounds/blank.jpg`{=latex}"
page-background: "`~/.local/share/pandoc/backgrounds/blankwhite.jpg`{=latex}"
page-background-opacity: 0.4
titlepage-text-color: "FF5555"
titlepage-logo: "`~/.local/share/pandoc/logos/LostSonBlu.png`{=latex}"
text-color: "282A36"
titlepage-rule-color: "6272a4"
titlepage-rule-height: 8
table-use-row-colors: true
toc: true
toc-own-page: true
listings-no-page-break: true

---
## California Housing Dataset


We are going to use the _California housing dataset_ to illustrate how the KNN ![[KNN#Theory]]algorithm works. The dataset was derived from the 1990 U.S. census. One row of the dataset represents the census of one block group.

A _block_ group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data. Besides block group, another term used is household, a household is a group of people residing within a home.

-   _MedInc_ - median income in block group
-   _HouseAge_ - median house age in a block group
-   _AveRooms_ - the average number of rooms (provided per household)
-   _AveBedrms_ - the average number of bedrooms (provided per household)
-   _Population_ - block group population
-   _AveOccup_ - the average number of household members
-   _Latitude_ - block group latitude
-   _Longitude_ - block group longitude
-   _MedHouseVal_ - median house value for California districts (hundreds of thousands of dollars)

The dataset is **already part of the Scikit-Learn library**, we only need to import it and load it as a dataframe:
```python  
from sklearn.datasets import fetch_california_housing
# as_frame=True loads the data in a dataframe format, with other metadata besides it
california_housing = fetch_california_housing(as_frame=True)
# Select only the dataframe part and assign it to the df variable
df = california_housing.frame
```

Importing the data directly from Scikit-Learn, imports more than only the columns and numbers and includes the data description as a `Bunch` object - so we've just extracted the `frame`. Further details of the dataset are available [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

Let's import Pandas and take a peek at the first few rows of data:
```python
import pandas as pd
df.head()
```

Executing the code will display the first five row of the dataset

### Regression KNN Scikit-Learn
#### Preprocessing Data for KNN Regression

The preprocessing is where the first differences between the regression and classification tasks appear. Since this section is all about regression, we'll prepare our dataset accordingly.

For the regression, we need to predict another median house value. To do so, we will assign _MedHouseVal_ to _y_ and all other columns to _X_ just by dropping _MedHouseVal_:

```python
y = df['MedHouseVal']
X = df.drop(['MedHouseVal'], axis = 1)
```

By looking at our variables descriptions, we can see that we have differences in measurements. To avoid guessing, let's use the `describe()` method to check:

```python
# .T transposes the results, transforming rows into columns
X.describe().T
```

the `mean` value of `MedInc` is approximately `3.87` and the `mean` value of `HouseAge` is about `28.64`, making it _7.4 times larger_ than `MedInc`. Other features also have differences in mean and _standard deviation_ - to see that, look at the `mean` and `std` values and observe how they are distant from each other. For `MedInc` `std` is approximately `1.9`, for `HouseAge`, `std` is `12.59` and the same applies to the other features.

We're using an algorithm based on _distance_ and distance-based algorithms suffer greatly from _data that isn't on the same scale_, such as this data. The scale of the points may (and in practice, almost always does) distort the real distance between values.

To perform Feature Scaling, we will use Scikit-Learn's `StandardScaler` class later. If we apply the scaling right now (before a train-test split), the calculation would include test data, effectively _leaking_ test data information into the rest of the pipeline. This sort of _data leakage_ is unfortunately commonly skipped, resulting in irreproducible or illusory findings.



**Advice:** If you'd like to learn more about feature scaling - read our [_"Feature Scaling Data with Scikit-Learn for Machine Learning in Python"_](https://stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/)

#### Splitting Data into Train and Test Sets

To be able to scale our data without leakage, but also to evaluate our results and to avoid over-fitting, we'll divide our dataset into train and test splits.

A straightforward way to create train and test splits is the `train_test_split` method from Scikit-Learn. The split doesn't linearly split at some point, but samples X% and Y% randomly. To make this process reproducible (to make the method always sample the same datapoints), we'll set the `random_state` argument to a certain `SEED`:

```python
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
```

This piece of code samples 75% of the data for training and 25% of the data for testing. By changing the `test_size` to 0.3, for instance, you could train with 70% of the data and test with 30%.

By using 75% of the data for training and 25% for testing, out of 20640 records, the training set contains 15480 and the test set contains 5160. We can inspect those numbers quickly by printing the lengths of the full dataset and of split data:

```python
len(X)       # 20640
len(X_train) # 15480
len(X_test)  # 5160
```

Great! We can now fit the data scaler on the `X_train` set, and scale both `X_train` and `X_test` without leaking any data from `X_test` into `X_train`.


**Advice:** If you'd like to learn more about the `train_test_split()` method the importance of a train-test-validation split, as well as how to separate out validation sets as well, read our [_"Scikit-Learn's train_test_split() - Training, Testing and Validation Sets"_](https://stackabuse.com/scikit-learns-traintestsplit-training-testing-and-validation-sets/).

#### Feature Scaling for KNN Regression

By importing `StandardScaler`, instantiating it, fitting it according to our train data (preventing leakage), and transforming both train and test datasets, we can perform feature scaling:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)

# Scale both X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

**Note:** Since you'll oftentimes call `scaler.fit(X_train)` followed by `scaler.transform(X_train)` - you can call a single `scaler.fit_transform(X_train)` followed by `scaler.transform(X_test)` to make the call shorter!


Now our data is scaled! The scaler maintains only the data points, and not the column names, when applied on a `DataFrame`. Let's organize the data into a DataFrame again with column names and use `describe()` to observe the changes in `mean` and `std`

```python
col_names=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
scaled_df = pd.DataFrame(X_train, columns=col_names)
scaled_df.describe().T
```

Observe how all standard deviations are now `1` and the means have become smaller. This is what makes our data _more uniform_! Let's train and evaluate a KNN-based regressor.
#### Training and Predicting KNN Regression
Scikit-Learn's intuitive and stable API makes training regressors and classifiers very straightforward. Let's import the `KNeighborsRegressor` class from the `sklearn.neighbors` module, instantiate it, and fit it to our train data

```python
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)
```

In the above code, the `n_neighbors` is the value for _K_, or the number of neighbors the algorithm will take into consideration for choosing a new median house value. `5` is the default value for `KNeighborsRegressor()`. There is no ideal value for K and it is selected after testing and evaluation, however, to start out, `5` is a commonly used value for KNN and was thus set as the default value.

The final step is to make predictions on our test data. To do so, execute the following script:

```python
y_pred = regressor.predict(X_test)
```
We can now evaluate how well our model generalizes to new data that we have labels (ground truth) for - the test set!

#### Evaluating the Algorithm for KNN Regression
The most commonly used regression metrics for evaluating the algorithm are _mean absolute error_ (MAE), _mean squared error_ (MSE), _root mean squared error_ (RMSE), and _coefficient of determination_ ($R^2$):

1.  **Mean Absolute Error (MAE)**: When we subtract the predicted values from the actual values, obtain the errors, sum the absolute values of those errors and get their mean. This metric gives a _notion of the overall error_ for each prediction of the model, the smaller (closer to 0) the better:
$$mae = (\frac{1}{n})\sum_{i=1}^n|Actual - Predicted|$$
2. **Mean Squared Error (MSE)**: It is similar to the MAE metric, but it squares the absolute values of the errors. Also, as with MAE, the smaller, or closer to 0, the better. The MSE value is squared so as to make _large errors even larger_. One thing to pay close attention to, it that it is usually a hard metric to interpret due to the size of its values and of the fact that they aren't on the same scale as the data.
$$mse = \sum_{i=1}^D(Actual - Predicted)^2$$
3. **Root Mean Squared Error (RMSE)**: Tries to solve the interpretation problem raised with the MSE by getting the square root of its final value, so as to scale it back to the _same units of the data_. It is _easier to interpret_ and good when we need to display or _show the actual value of the data with the error_. It shows how much the data may vary, so, if we have an RMSE of 4.35, our model can make an error either because it added 4.35 to the actual value, or needed 4.35 to get to the actual value. The closer to 0, the better as well.
$$rmse = \sqrt{\sum_{i=1}^D(Actual - Predicted)^2}  $$

The `mean_absolute_error()` and `mean_squared_error()` methods of `sklearn.metrics` can be used to calculate these metrics as can be seen in the following snippet:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')
```

The $R^2$ can be calculated directly with the `score()` method:

```python
regressor.score(X_test, y_test)
```

The results show that our KNN algorithm overall error and mean error are around `0.44`, and `0.43`. Also, the RMSE shows that we can go above or below the actual value of data by adding `0.65` or subtracting `0.65`. How good is that?

Let's check what the prices look like:

```python
y.describe()
```

The mean is `2.06` and the standard deviation from the mean is `1.15` so our score of ~`0.44` isn't really stellar, but isn't too bad.

With the $R^2$, the closest to 1 we get (or 100), the better. The $R^2$ tells how much of the changes in data, or data _variance_ are being understood or _explained_ by KNN.

$$R^2 = 1 - \frac{\sum(actual - predicted)^2}{\sum(actual - actual mean)^2}$$
With a value of `0.67`, we can see that our model explains 67% of the data variance. It is already more than 50%, which is ok, but not very good. Is there any way we could do better?

We have used a predetermined K with a value of `5`, so, we are using 5 neighbors to predict our targets which is not necessarily the best number. To understand which would be an ideal number of Ks, we can analyze our algorithm errors and choose the K that minimizes the loss.

#### Finding the Best K for KNN Regression
Ideally, you would see which metric fits more into your context - but it is usually interesting to test all metrics. Whenever you can test all of them, do it. Here, we will show how to choose the best K using only the mean absolute error, but you can change it to any other metric and compare the results.

To do this, we will create a for loop and run models that have from 1 to X neighbors. At each interaction, we will calculate the MAE and plot the number of Ks along with the MAE result:

```python
error = []

# Calculating MAE error for K values between 1 and 39
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(y_test, pred_i)
    error.append(mae)
```

Now, let's plot the _errors_:

```python
import matplotlib.pyplot as plt 

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
```

Looking at the plot, it seems the lowest MAE value is when K is `12`. Let's get a closer look at the plot to be sure by plotting less data:

```python
plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error[:14], color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')
```

You can also obtain the lowest error and the index of that point using the built-in `min()` function (works on lists) or convert the list into a NumPy array and get the `argmin()` (index of the element with the lowest value):

```python
import numpy as np 

print(min(error))               # 0.43631325936692505
print(np.array(error).argmin()) # 11
```

We started counting neighbors on 1, while arrays are 0-based, so the 11th index is 12 neighbors!

This means that we need 12 neighbors to be able to predict a point with the lowest MAE error. We can execute the model and metrics again with 12 neighbors to compare results:

```python
knn_reg12 = KNeighborsRegressor(n_neighbors=12)
knn_reg12.fit(X_train, y_train)
y_pred12 = knn_reg12.predict(X_test)
r2 = knn_reg12.score(X_test, y_test) 

mae12 = mean_absolute_error(y_test, y_pred12)
mse12 = mean_squared_error(y_test, y_pred12)
rmse12 = mean_squared_error(y_test, y_pred12, squared=False)
print(f'r2: {r2}, \nmae: {mae12} \nmse: {mse12} \nrmse: {rmse12}')
```

With 12 neighbors our KNN model now explains 69% of the variance in the data, and has lost a little less, going from `0.44` to `0.43`, `0.43` to `0.41`, and `0.65` to `0.64` with the respective metrics. It is not a very large improvement, but it is an improvement nonetheless.


**Note:** Going further in this analysis, doing an Exploratory Data Analysis (EDA) along with residual analysis may help to select features and achieve better results.


We have already seen how to use KNN for regression - but what if we wanted to classify a point instead of predicting its value? Now, we can look at how to use KNN for classification.

### Classification using K-Nearest Neighbors with Scikit-Learn

In this task, instead of predicting a continuous value, we want to predict the class to which these block groups belong. To do that, we can divide the median house value for districts into groups with different house value ranges or _bins_

#### Preprocessing Data for Classification

Let's create the data bins to transform our continuous values into categories:

```python
# Creating 4 categories and assigning them to a MedHouseValCat column
df["MedHouseValCat"] = pd.qcut(df["MedHouseVal"], 4, retbins=False, labels=[1, 2, 3, 4])
```

Then, we can split our dataset into its attributes and labels:

```python
y = df['MedHouseValCat']
X = df.drop(['MedHouseVal', 'MedHouseValCat'], axis = 1)
```

Since we have used the `MedHouseVal` column to create bins, we need to drop the `MedHouseVal` column and `MedHouseValCat` columns from `X`. This way, the `DataFrame` will contain the first 8 columns of the dataset (i.e. attributes, features) while our `y` will contain only the `MedHouseValCat` assigned label.

**Note:** You can also select columns using `.iloc()` instead of dropping them. When dropping, just be aware you need to assign `y` values before assigning `X` values, because you can't assign a dropped column of a `DataFrame` to another object in memory.

#### Splitting Data into Train and Test Sets

As it has been done with regression, we will also divide the dataset into training and test splits. Since we have different data, we need to repeat this process:

```python
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
```

We will use the standard Scikit-Learn value of 75% train data and 25% test data again. This means we will have the same train and test number of records as in the regression before.

#### Feature Scaling for Classification

Since we are dealing with the same unprocessed dataset and its varying measure units, we will perform feature scaling again, in the same way as we did for our regression data:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

#### Training and Predicting for Classification

After binning, splitting, and scaling the data, we can finally fit a classifier on it. For the prediction, we will use 5 neighbors again as a baseline. You can also instantiate the `KNeighbors_` class without any arguments and it will automatically use 5 neighbors. Here, instead of importing the `KNeighborsRegressor`, we will import the `KNeighborsClassifier`, class:

```python
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
```

After fitting the `KNeighborsClassifier`, we can predict the classes of the test data:

```python
y_pred = classifier.predict(X_test)
```

Time to evaluate the predictions! Would predicting classes be a better approach than predicting values in this case? Let's evaluate the algorithm to see what happens.

#### Evaluating KNN for Classification

For evaluating the KNN classifier, we can also use the `score` method, but it executes a different metric since we are scoring a classifier and not a regressor. The basic metric for classification is `accuracy` - it describes how many predictions our classifier got right. The lowest accuracy value is 0 and the highest is 1. We usually multiply that value by 100 to obtain a percentage.

$$accuracy = \frac{num\space of\space correct\space predictions}{total \space number \space of \space predictions}$$

**Note:** It is extremely hard to obtain 100% accuracy on any real data, if that happens, be aware that some leakage or something wrong might be happening - there is no consensus on an ideal accuracy value and it is also context-dependent. Depending on the _cost of error_ (how bad it is if we trust the classifier and it turns out to be wrong), an acceptable error rate might be 5%, 10% or even 30%.

Let's score our classifier:

```python
acc =  classifier.score(X_test, y_test)
print(acc) # 0.6191860465116279
```

By looking at the resulting score, we can deduce that our classifier got ~62% of our classes right. This already helps in the analysis, although by only knowing what the classifier got right, it is difficult to improve it.


-  There are 4 classes in our dataset - what if our classifier got **90% of classes 1, 2, and 3 right**, but only **30% of class 4 right**?

A systemic failure of some class, as opposed to a balanced failure shared between classes can both yield a 62% accuracy score. Accuracy isn't a really good metric for actual evaluation - but does serve as a good proxy. More often than not, with balanced datasets, a 62% accuracy is relatively evenly spread. Also, more often than not, datasets aren't balanced, so we're back at square one with accuracy being an insufficient metric.

We can look deeper into the results using other metrics to be able to determine that. This step is also different from the regression, here we will use:

1.  **Confusion Matrix**: To know how much we got right or wrong for **_each class_**. The values that were correct and correctly predicted are called _true positives_ the ones that were predicted as positives but weren't positives are called _false positives_. The same nomenclature of _true negatives_ and _false negatives_ is used for negative values;
2.  **Precision**: To understand what correct prediction values were considered correct by our classifier. Precision will divide those true positives values by anything that was predicted as a positive;
$$precision = \frac{true \space positive}{true \space positive + false \space positive}$$
3.  **Recall**: to understand how many of the true positives were identified by our classifier. The recall is calculated by dividing the true positives by anything that should have been predicted as positive.
$$recall = \frac{true \space positive}{true \space positive + false \space negative}$$
4. **F1 score**: Is the balanced or _harmonic mean_ of precision and recall. The lowest value is 0 and the highest is 1. When `f1-score` is equal to 1, it means all classes were correctly predicted - this is a very hard score to obtain with real data (exceptions almost always exist).
$$f^1score = 2[\frac{precision * recall}{precision + recall}]$$

**Note:** A weighted F1 score also exists, and it's just an F1 that doesn't apply the same weight to all classes. The weight is typically dictated by the classes **support** - how many instances "support" the F1 score (the proportion of labels belonging to a certain class). The lower the support (the fewer instances of a class), the lower the weighted F1 for that class, because it's more unreliable.

The `confusion_matrix()` and `classification_report()` methods of the `sklearn.metrics` module can be used to calculate and display all these metrics. The `confusion_matrix` is better visualized using a heatmap. The classification report already gives us `accuracy`, `precision`, `recall`, and `f1-score`, but you could also import each of these metrics from `sklearn.metrics`.

To obtain metrics, execute the following snippet:


```python
from sklearn.metrics import classification_report, confusion_matrix

#importing Seaborn's to use the heatmap 
import seaborn as sns

# Adding classes names for better interpretation
classes_names = ['class 1','class 2','class 3', 'class 4']
cm = pd.DataFrame(confusion_matrix(yc_test, yc_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d');

print(classification_report(y_test, y_pred))
```

The results show that KNN was able to classify all the 5160 records in the test set with 62% accuracy, which is above average. The supports are fairly equal (even distribution of classes in the dataset), so the weighted F1 and unweighted F1 are going to be roughly the same.

We can also see the result of the metrics for each of the 4 classes. From that, we are able to notice that `class 2` had the lowest precision, lowest `recall`, and lowest `f1-score`. `Class 3` is right behind `class 2` for having the lowest scores, and then, we have `class 1` with the best scores followed by `class 4`.

By looking at the confusion matrix, we can see that:

-   `class 1` was mostly mistaken for `class 2` in 238 cases
-   `class 2` for `class 1` in 256 entries, and for `class 3` in 260 cases
-   `class 3` was mostly mistaken by `class 2`, 374 entries, and `class 4`, in 193 cases
-   `class 4` was wrongly classified as `class 3` for 339 entries, and as `class 2` in 130 cases.

Also, notice that the diagonal displays the true positive values, when looking at it, it is plain to see that `class 2` and `class 3` have the least correctly predicted values.

With those results, we could go deeper into the analysis by further inspecting them to figure out why that happened, and also understanding if 4 classes are the best way to bin the data. Perhaps values from `class 2` and `class 3` were too close to each other, so it became hard to tell them apart.

> Always try to test the data with a different number of bins to see what happens.

Besides the arbitrary number of data bins, there is also another arbitrary number that we have chosen, the number of K neighbors. The same technique we applied to the regression task can be applied to the classification when determining the number of Ks that maximize or minimize a metric value.


#### Finding the Best K for KNN Classification

Let's repeat what has been done for regression and plot the graph of K values and the corresponding metric for the test set. You can also choose which metric better fits your context, here, we will choose `f1-score`.

In this way, we will plot the `f1-score` for the predicted values of the test set for all the K values between 1 and 40.

First, we import the `f1_score` from `sklearn.metrics` and then calculate its value for all the predictions of a K-Nearest Neighbors classifier, where K ranges from 1 to 40:

```python
from sklearn.metrics import f1_score

f1s = []

# Calculating f1 score for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # using average='weighted' to calculate a weighted average for the 4 classes 
    f1s.append(f1_score(y_test, pred_i, average='weighted'))
```

The next step is to plot the `f1_score` values against K values. The difference from the regression is that instead of choosing the K value that minimizes the error, this time we will choose the value that maximizes the `f1-score`.

Execute the following script to create the plot:

```python
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), f1s, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('F1 Score K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
```



From the output, we can see that the `f1-score` is the highest when the value of the K is `15`. Let's retrain our classifier with 15 neighbors and see what it does to our classification report results:

```python
classifier15 = KNeighborsClassifier(n_neighbors=15)
classifier15.fit(X_train, y_train)
y_pred15 = classifier15.predict(X_test)
print(classification_report(y_test, y_pred15))
```

Notice that our metrics have improved with 15 neighbors, we have 63% accuracy and higher `precision`, `recall`, and `f1-scores`, but we still need to further look at the bins to try to understand why the `f1-score` for classes `2` and `3` is still low.

Besides using KNN for regression and determining block values and for classification, to determine block classes - we can also use KNN for detecting which mean blocks values are different from most - the ones that don't follow what most of the data is doing. In other words, we can use KNN for _detecting outliers_.

### Implementing KNN for Outlier Detection with Scikit-Learn

_Outlier detection_ uses another method that differs from what we had done previously for regression and classification.

Here, we will see how far each of the neighbors is from a data point. Let's use the default 5 neighbors. For a data point, we will calculate the distance to each of the K-nearest neighbors. To do that, we will import another KNN algorithm from Scikit-learn which is not specific for either regression or classification called simply `NearestNeighbors`.

After importing, we will instantiate a `NearestNeighbors` class with 5 neighbors - you can also instantiate it with 12 neighbors to identify outliers in our regression example or with 15, to do the same for the classification example. We will then fit our train data and use the `kneighbors()` method to find our calculated distances for each data point and neighbors indexes:

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors = 5)
nbrs.fit(X_train)
# Distances and indexes of the 5 neighbors 
distances, indexes = nbrs.kneighbors(X_train)
```

Now we have 5 distances for each data point - the distance between itself and its 5 neighbors, and an index that identifies them. Let's take a peek at the first three results and the shape of the array to visualize this better.

To look at the first three distances shape, execute:

```python
distances[:3], distances.shape
```

Observe that there are 3 rows with 5 distances each. We can also look and the neighbors' indexes

```python
indexes[:3], indexes[:3].shape
```

In the output above, we can see the indexes of each of the 5 neighbors. Now, we can continue to calculate the mean of the 5 distances and plot a graph that counts each row on the X-axis and displays each mean distance on the Y-axis:

```python
dist_means = distances.mean(axis=1)
plt.plot(dist_means)
plt.title('Mean of the 5 neighbors distances for each data point')
plt.xlabel('Count')
plt.ylabel('Mean Distances')
```

Notice that there is a part of the graph in which the mean distances have uniform values. That Y-axis point in which the means aren't too high or too low is exactly the point we need to identify to cut off the outlier values.

In this case, it is where the mean distance is 3. Let's plot the graph again with a horizontal dotted line to be able to spot it:

```python
dist_means = distances.mean(axis=1)
plt.plot(dist_means)
plt.title('Mean of the 5 neighbors distances for each data point with cut-off line')
plt.xlabel('Count')
plt.ylabel('Mean Distances')
plt.axhline(y = 3, color = 'r', linestyle = '--')
```

This line marks the mean distance for which above it all values vary. This means that all points with a `mean` distance above `3` are our outliers. We can find out the indexes of those points using `np.where()`. This method will output either `True` or `False` for each index in regards to the `mean` _above 3_ condition:

```python
import numpy as np

# Visually determine cutoff values > 3
outlier_index = np.where(dist_means > 3)
outlier_index
```

Now we have our outlier point indexes. Let's locate them in the dataframe:

```python
# Filter outlier values
outlier_values = df.iloc[outlier_index]
outlier_values
```

Our outlier detection is finished. This is how we spot each data point that deviates from the general data trend. We can see that there are 16 points in our train data that should be further looked at, investigated, maybe treated, or even removed from our data (if they were erroneously input) to improve results. Those points might have resulted from typing errors, mean block values inconsistencies, or even both.
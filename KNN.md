[source](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)
# When
suppose you wanted to rent an apartment and recently found out your friend's neighbor might put her apartment for rent in 2 weeks. Since the apartment isn't on a rental website yet, how could you try to _estimate_ its rental value?

the friends pays _$1,200_ in rent. Your rent value might be _around_ that number, but the apartments arent exactly the same ( orientation, area, furniture quality, etc.), so, it would be nice to have more data on other apartments.
By asking other neighbors and looking at the apartments from the same building that were listed on a rental website, the closest three neighboring apartment rents are
-  $1,200 
-  $1,210
-  $1,210
-  $1,215
Those apartments are on the same block and floor as your friend's apartment.

Other apartments, that are further away, on the same floor, but in a different block have rents of 
- $1,400,
- $1,430
- $1,500
- $1,470. 
It seems they are more expensive due to having more light from the sun in the evening.

estimated rent given data would be $1,210

That is the general idea of what the **K-Nearest Neighbors (KNN)** algorithm does! It classifies or regresses new data based on its proximity to already existing data.

# Theory
When the estimated value is a continuous number, such as the rent value, KNN is used for _**regression**_. But we could also divide apartments into categories based on the minimum and maximum rent, for instance. When the value is discrete, making it a category, KNN is used for _**classification**_.
- Two types of estimation __Classification and Regression__


There is also the possibility of estimating which neighbors are so different from others that they will probably stop paying rent. This is the same as detecting which data points are so far away that they don't fit into any value or category, when that happens, KNN is used for _**outlier detection**_.
- caluclating outliers 

In our example, we also already knew the rents of each apartment, which means our data was labeled. KNN uses previously labeled data, which makes it a _**supervised learning algorithm**_.

Each time there is a new point added to the data, KNN uses just one part of the data for deciding the value (regression) or class (classification) of that added point. Since it doesn't have to look at all the points again, this makes it a _**lazy learning algorithm**_.

KNN also doesn't assume anything about the underlying data characteristics, it doesn't expect the data to fit into some type of distribution, such as uniform, or to be linearly separable. This means it is a _**non-parametric learning algorithm**_. This is an extremely useful feature since most of the real-world data doesn't really follow any theoretical assumption.

# Visualizing

As it has been shown, the intuition behind the KNN algorithm is one of the most direct of all the supervised machine learning algorithms. The algorithm first calculates the _distance_ of a new data point to all other training data points.

After calculating the distance, KNN selects a number of nearest data points - 2, 3, 10, or really, any integer. This number of points (2, 3, 10, etc.) is the **K** in K-Nearest Neighbors!

In the final step, if it is a regression task, KNN will calculate the average weighted sum of the K-nearest points for the prediction. If it is a classification task, the new data point will be assigned to the class to which the majority of the selected K-nearest points belong.

Let's visualize the algorithm in action with the help of a simple example. Consider a dataset with two variables and a K of 3.

When performing regression, the task is to find the value of a new data point, based on the average weighted sum of the 3 nearest points.

KNN with `K = 3`, when _**used for regression**_:
![[Pasted image 20221006231333.png|[200]]

![[KNN Regression.png]]

The KNN algorithm will start by calculating the distance of the new point from all the points. It then finds the 3 points with the least distance to the new point. This is shown in the second figure above, in which the three nearest points, `47`, `58`, and `79` have been encircled. After that, it calculates the weighted sum of `47`, `58` and `79` - in this case the weights are equal to 1 - we are considering all points as equals, but we could also assign different weights based on distance. After calculating the weighted sum, the new point value is `61,33`.

And when performing a classification, the KNN task to classify a new data point, into the `"Purple"` or `"Red"` class.

KNN with `K = 3`, when _**used for classification**_:
![[Pasted image 20221006231651.png]]
![[Pasted image 20221006231704.png]]

The KNN algorithm will start in the same way as before, by calculating the distance of the new point from all the points, finding the 3 nearest points with the least distance to the new point, and then, instead of calculating a number, it assigns the new point to the class to which majority of the three nearest points belong, the red class. Therefore the new data point will be classified as `"Red"`.

The outlier detection process is different from both above, we will talk more about it when implementing it after the regression and classification implementations.

### Pros and Cons of KNN

In this section, we'll present some of the pros and cons of using the KNN algorithm.

#### Pros

-   It is easy to implement
-   It is a lazy learning algorithm and therefore doesn't require training on all data points (only using the K-Nearest neighbors to predict). This makes the KNN algorithm much faster than other algorithms that require training with the whole dataset such as [Support Vector Machines](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/), [linear regression](https://stackabuse.com/linear-regression-in-python-with-scikit-learn/), etc.
-   Since KNN requires no training before making predictions, new data can be added seamlessly
-   There are only two parameters required to work with KNN, i.e. the value of K and the distance function

#### Cons

-   The KNN algorithm doesn't work well with high dimensional data because with a large number of dimensions, the distance between points gets "weird", and the distance metrics we use don't hold up
-   Finally, the KNN algorithm doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features
### Going Further - Hand-Held End-to-End Project



Your inquisitive nature makes you want to go further? We recommend checking out our **_Guided Project_**: ["Hands-On House Price Prediction - Machine Learning in Python"](https://stackabuse.com/courses/hands-on-house-price-prediction-machine-learning-in-python/#cta).

[Machine Learning in Python](https://stackabuse.com/courses/hands-on-house-price-prediction-machine-learning-in-python/#cta)

> In this guided project - you'll learn how to build powerful traditional machine learning models as well as deep learning models, utilize Ensemble Learning and train meta-learners to predict house prices from a bag of Scikit-Learn and Keras models.

Using Keras, the deep learning API built on top of Tensorflow, we'll experiment with architectures, build an ensemble of stacked models and train a _meta-learner_ neural network (level-1 model) to figure out the pricing of a house.

Deep learning is amazing - but before resorting to it, it's advised to also attempt solving the problem with simpler techniques, such as with _shallow learning_ algorithms. Our baseline performance will be based on a _Random Forest Regression_ algorithm. Additionally - we'll explore creating ensembles of models through Scikit-Learn via techniques such as _bagging_ and _voting_.

This is an end-to-end project, and like all Machine Learning projects, we'll start out with _Exploratory Data Analysis_, followed by _Data Preprocessing_ and finally _Building Shallow_ and _Deep Learning Models_ to fit the data we've explored and cleaned previously.

### Conclusion

KNN is a simple yet powerful algorithm. It can be used for many tasks such as regression, classification, or outlier detection.

KNN has been widely used to find document similarity and pattern recognition. It has also been employed for developing recommender systems and for dimensionality reduction and pre-processing steps for computer vision - particularly face recognition tasks.

In this guide - we've gone through regression, classification and outlier detection using Scikit-Learn's implementation of the K-Nearest Neighbor algorithm.
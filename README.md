
# LEARN: Machine Learning using Python By CodeBasics

Welcome to the "Studying Machine Learning with Code Basics" GitHub repository! This repository is designed to help learn and explore the exciting world of machine learning using Python. You will find valuable resources and code examples here to enhance understanding of machine learning concepts and techniques. I have documented all the learning and the learning Journey.

## Roadmap

- Supervised Learning Single variable
- Supervised Learning Multiple variables
- Gradient Descent and Cost Function
- Save Model Using Joblib And Pickle
- Dummy Variables and One Hot Encoding
- Training and Testing Data 
- Logistic Regression: Binary Classification
- Logistic Regression: Multiple Classification
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- K Fold cross Validation
- K Means Clustering Algorithms 
- Naive Bayes Classifier 
- Hyper Parameter Tuning
- Lasso Ridge Regression
- KNN Classification
- Principal Component Analysis
- Bias vs. Variance
- Bagging/Ensemble Learning

## Lessons Learned

Machine Learning is the field of artificial intelligence (AI) dedicated to enabling computers to learn and make decisions from data. Machine learning systems have the capacity to process vast datasets, identify patterns, and use that knowledge to improve their performance and make predictions or decisions. As a transformative technology, machine learning empowers businesses, scientists, and individuals to harness the potential of data-driven intelligence, ushering in a new era of innovation and problem-solving.

### 1) Linear Regression with Single Variable: ###

Whenever there a dependent variable/output 'y' that is only dependent on a single variable 'x', then we use the concept 'Linear Regression'. The Regression Line should be such that the Mean Squared Error (MSE) between the data points and the line should be minimum i.e. finding the the Best Fit Line.

<p align="center">
  $\ y = m*x+b$
</p>
Where,

- $\ y$ - output variable
- $\ x$ - input variable
- $\ m$ - slope/gradient/regression-coefficient/weight
- $\ b$ - y_intercept/bias

### 2) Linear Regression with Multiple Variable: ###

When $\ y$ is dependent on multiple variables $\ x_{1}, x_{2}, x_{3}...$ then we use the concept of Linear Regression for Multiple variables as the output is dependent on multiple features. 

<p align="center">
$\ y = m_{1}*x_{1} + m_{2}*x_{2}+ m_{3}*x_{3}+...+b$ 
</p>
Where:

- $\ y$ - output variable
- $\ x_{1}, x_{2}, x_{3}...$ - input variables
- $\ m_{1}, m_{2}, m_{3}...$ - slopes/gradients/regression-coefficients/weights
- $\ b$ - y_intercept/bias

### 3) Gradient Descent and Cost Function: ###

To find the best fit line with most minimized error, we use the method 'Mean Squared Error' i.e. minimize cost function. Gradient Descent is an algorithm that finds best fit line for given training dataset.
In Gradient Descent algorithms it is necessary that we find the right learning rate and number of iterations to get the best minimum cost. The formula of Mean Squared Error is given by

<p align="center">
MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$
</p>

Where:
- MSE stands for Mean Squared Error.
- $\ n$ represents the number of data points in your dataset.
- $\ y$ is the actual target (ground truth) for the \(i\)-th data point.
- $\ \hat{y}$ is the predicted target (the output of your model) for the \(i\)-th data point.

Partial Derivative of MSE with respect to slope m and intercept b is as follows:

<p align="center">
$\ \frac{\partial MSE}{\partial m}$ = $\frac{-2}{n} \Sigma_{i=1}^n(x_{i}({y_{i}}-(m*x_{i}+b))) $
</p>
<p align="center">
$\ \frac{\partial MSE}{\partial b}$ = $\frac{-2}{n} \Sigma_{i=1}^n({y_{i}}-(m*x_{i}+b)) $
</p>

To find the new slope and new intercept -
<p align="center">
$\ m_{new} = m - Learning Rate * \frac{\partial MSE}{\partial m} $
</p>
<p align="center">
  $\ b_{new} = b - Learning Rate * \frac{\partial MSE}{\partial b} $
</p>

These equations together make the Gradient Descent Algorithm.

### 4) Save Model using JobLib and Pickle ###

Joblib provides two functions for saving and loading models: dump and load. To save a model using Joblib, you need to import the dump function from the joblib library and call the dump function with the model and the file name. While the “pickle” module provides a way to serialize and deserialize Python objects, including trained machine learning models. By saving a trained model using the pickle module, you can reuse the model for making predictions on new data, without having to retrain the model from scratch.

### 5) One Hot Encoding ###

One Hot Encoding can be defined as a process of transforming categorical variables into numerical format before fitting and training a Machine Learning algorithm. For each categorical variable, One Hot Encoding produces a numeric vector with a length equal to the number of categories present in the feature. We also eliminate Dummy variable Trap by discarding one of the dummy variable. 

### 6) Training and Testing Data ###

We can train and test the model by splitting the given diverse data set into 2 parts test data and train data with pre-defined proportions. This can be achieved by scikit library

### 7) Logistic Regression Binary Classification ###

The basis of logistic regression is the logistic function, also called the sigmoid function, which takes in any real valued number and maps it to a value between 0 and 1. Logistic regression model takes a linear equation as input and use logistic function and log odds to perform a binary classification task. Here, we take HR_survey based dataset from Kaggle and perform binary classification.

### 8) Logistic Regression MultiClass Classification ###

The multinomial logistic regression algorithm is an extension to the logistic regression model that involves changing the loss function to cross-entropy loss and predict probability distribution to a multinomial probability distribution to natively support multi-class classification problems. Here, we try to identify the hand written digits (0,1,2,.....,9 -> multiclass) and recognize thw digits using Logistic Regression.

**Confusion Matrix** - We also use confusion Matrix to understand where our model fails to predict and where it predicts accurately

### 9) Decision Tree ###

It is a non-parametric supervised machine learning algorithm which is utilized for both classification and regression tasks. It is used when the target variables are categorical and with branching happening through binary partitioning. We need to make sure the order of the categorical variables to create a decision tree model should be such that it has less entropy and High Information Gain. We also focus of 
**Gini Impurity** which is used to measure the dataset impurity level.


## Acknowledgements

 - [CodeBasics](https://www.youtube.com/@codebasics)



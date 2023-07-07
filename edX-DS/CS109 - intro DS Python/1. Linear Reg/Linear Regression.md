# Linear Regression

In An Introduction to Statistical Learning External link, 2nd edition, read the following sections:

Introduction - Pages 1 to 14
Simple Linear Regression - Pages 61 to 64

# 1.1 KNN

## Response vs. Predictor Variables
* Variables whose values we use to **make our prediction**. These are known as **predictors**, **features**, or **covariates**.
* Variables whose values we **want to predict**. These are known as o**utcomes**, **response variables**, or **dependent variables**.

## nomenclature
A vector is defined by lowercase variavle (x), a Matrix is defined by a uppercase variable (X).

![Nomemclature](/images/nomenclatures.png)

## Pandas

### Shape
![Shape](/images/shape.png)

### Series or DataFrame
![Series or dataframe](/images/series%20or%20dataframe.png)


---

KNN is a way to predict using the mean value between some k nearest neighboors
![KNN](/images/KNN.png)

# 1.2 Error Evaluation and Model Comparison

## Train Validation Split
To say a model is good we need to evaluate its errors. To do that, we need to separate our dataset in train and validation sets.

![train validation split](/images/train-validation-split.png)

Residual is error between the true value and the predicted value.
If my model predicts 3, and the true value is 3.4, the residual for this observation is 0.4.

![Residual](/images/residual.png)

## Error evaluation
the way we use to calculate the error measure is called **loss function**

* Mean Squared Error
* Max Absolute Error
* Mean Absolute Error

With the loss function defined, we can calculate it for all K-nearest-neighboors used and pick the K one whose MSE is the lowest!

## Scale for compare
Is 5 a good MSE? and 5000? We cannot say that if we don't compare it with other values!

we will assume the naive model (predicting just the mean) as the worst model, and the best model as predicting the exactly value.

### R-squared
![R-Squared](/images/R-squared.png)

Now we have a sacle, where:
* 0 is bad, means the model is good as the mean. 
* 1 is perfect, means the model predicts all true values exactly
* negative is worse, means the model is worst than the mean os alll values

# 1.3 Linear Regression
to get the best linear model for a dataset, we need to find what is the f(x) that will have the lowest loss function.

this process is called fitting or training.

Linear regressions are sensitive to outliers

## using SkLearn
![SkLearn](/images/sklearn.png)

the fit function will try to minimize the loss function. This is calculated using Derivatives and partial Derivatives.

# Kaggle
Projects that I looked over to see these topics in practice

* [KNN Classifier Tutorial](https://www.kaggle.com/code/prashant111/knn-classifier-tutorial)
  * > A good KNN first implementation. It has explanations and uses a short dataset that facilitates understanding

* [Car Price Prediction (Linear Regression - RFE)](https://www.kaggle.com/code/goyalshalini93/car-price-prediction-linear-regression-rfe)
  * > This notebook presents well done Data Analisys of the dataset variables before introducing the models.
  * > It has inferences for all analisis. Its more direct to the point, without much data analitics theorical explanation

* [A Detailed Regression Guide with House-pricing](https://www.kaggle.com/code/masumrumi/a-detailed-regression-guide-with-house-pricing)
  * > Very weel explained and detailed Linear Regression model to predict house pricing

* [Linear Regression Tutorial](https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial)
  * > The best math explained Linear Regression implementation of all these codes. 

* [Practical Introduction to 10 Regression Algorithm](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm)
  * > 10 regression models explained and applied to the same dataset


## My KNN Project
Predict team ranking based on last seasons data of each teams victory

## My Linear Regression Project
import time
from preprocess import *
from algorithm import *
import math
# ----------------------------------
# Pre-processing: Prepare the data for the algorithm.
# ----------------------------------
import pandas
import numpy as np

data = pandas.read_csv("train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
answer = "Survived"

# Replace values of Male with 0 & values of Female with 1
#data.loc[data["Sex"] == "male","Sex"] = 0
#data.loc[data["Sex"] == "female","Sex"] = 1
data["Sex"] = genderToBinary(data["Sex"])

# Use the average age for rows without an age
data["Age"] = fillAgeGaps(data["Age"])

# Normalize
for f in features:
	data[f] = normalize(data[f])

X = data[features]
Y = data[answer]

#Divide the data into training and cross validation sets.
from sklearn.cross_validation import KFold

folds = KFold(data.shape[0], n_folds=4, random_state=1)
Xtrain = Ytrain = Xval = Yval = []
for train, test in folds:
	Xtrain = X.iloc[train,:]
	Ytrain = Y.iloc[train]
	Xval = X.iloc[test,:]
	Yval = Y.iloc[test]

# ----------------------------------
# Train Algorithm: Take the data and compute thetas.
# ----------------------------------
from sklearn import linear_model

'''
best_C = tweakLogreg(Xtrain, Ytrain, Xval, Yval, 0, 1.5, 25)
print(best_C)
'''

alg = linear_model.LogisticRegression(C=7.5)
alg.fit(Xtrain, Ytrain)

# ----------------------------------
# Cross Validation: Test trained algorithm on the cross validation set and further tweak it.
# ----------------------------------

accuracy = crossVal(alg, Xval, Yval)
print(accuracy)

# ----------------------------------
# Learning Curve: Plot a learning curve.
# ----------------------------------

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves"
# SVC is more expensive so we do a lower number of CV iterations:
plot_learning_curve(alg, title, X, Y, (0.5, 1.01), cv=folds)

plt.show()        

# ----------------------------------
# Test Predictions: Run the algorithm on the test set to generate predictions.
# ----------------------------------
test = pandas.read_csv("test.csv")
test["Sex"] = genderToBinary(test["Sex"])
test["Age"] = fillAgeGaps(data["Age"])

# Normalize
for f in features:
    test[f] = normalize(test[f])

Xtest = test[features]
testPredictions = alg.predict(Xtest)

pandas.DataFrame({
        	"PassengerId": test["PassengerId"],
        	"Survived": testPredictions
    		}).to_csv('submission' + str(math.floor(time.time())) + '.csv',index=False)

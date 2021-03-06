import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import pandas
import numpy as np

def splitSet(X, Y, n_folds, random_state = 1):
	folds = cross_validation.KFold(X.shape[0], n_folds=n_folds, shuffle=True)
	X1 = Y1 = X2 = Y2 = []
	for train, test in folds:
		X1 = X.iloc[train,:]
		Y1 = Y.iloc[train]
		X2 = X.iloc[test,:]
		Y2 = Y.iloc[test]
	return [X1, Y1, X2, Y2, folds]

def learningCurve(alg, X, Y, folds):
	plotLCurve = None

	while (type(plotLCurve) != int or plotLCurve not in range(1,3)):
		try:
			plotLCurve = int(input("\nPlease choose whether to plot a learning curve or not:\n1 - to plot\n2 - to not plot\n\n"))
		except Exception:
			print("Incorrect input. Please enter either 1 or 2.")

	if plotLCurve == 1:
		plot_learning_curve(alg, "Learning Curves", X, Y, (0.5, 1.01), cv=folds)
		plt.show() 

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

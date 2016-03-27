from preprocess import *
from algorithm import *
from helper import *
import time
import math
import sys
# ----------------------------------
# Pre-processing: Prepare the data for the algorithm.
# ----------------------------------
import pandas
import numpy as np

data = pandas.read_csv("train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
answer = "Survived"

# Replace values of Male with 0 & values of Female with 1
data["Sex"] = genderToBinary(data["Sex"])

# Use the average age for rows without an age
data["Age"] = fillAgeGaps(data["Age"])

# Normalize
for f in features:
	data[f] = normalize(data[f])

X = data[features]
Y = data[answer]

#Divide the data into training and cross validation sets.
from sklearn import cross_validation

[Xtrainval, Ytrainval, Xtest, Ytest, folds1] = splitSet(X, Y, 5)
[Xtrain, Ytrain, Xval, Yval, folds2] = splitSet(Xtrainval, Ytrainval, 4)

# ----------------------------------
# Algorithm: Run the algorithm.
# ----------------------------------
algChoice = None

while (type(algChoice) != int or algChoice not in range(1,3)):
	try:
		algChoice = int(input("\nPlease choose an algorithm:\n1 - for Logistic Regression\n2 - for SVM with Sigmoid Kernel\n\n"))
	except Exception:
		print("Incorrect input. Please enter either 1 or 2.")

if algChoice == 1:
	# Logistic Regression
	[acc, alg] = logReg(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
elif algChoice == 2:
	# SVM with Sigmoid Kernel
	[acc, alg] = SVMSigmoid(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)

print("Accuracy = " + str(acc))

# ----------------------------------
# Learning Curve: Plot a learning curve.
# ----------------------------------
learningCurve(alg, Xtrainval, Ytrainval, folds2)


# ----------------------------------
# Test Predictions: Run the algorithm on the test set to generate predictions.
# ----------------------------------
createSub = None

while (type(createSub) != int or createSub not in range(1,3)):
	try:
		createSub = int(input("\nPlease choose whether to generate a submission file or not:\n1 - to generate\n2 - to not generate\n\n"))
	except Exception:
		print("Incorrect input. Please enter either 1 or 2.")

if createSub == 1:
	submSet = pandas.read_csv("test.csv")
	submSet["Sex"] = genderToBinary(submSet["Sex"])
	submSet["Age"] = fillAgeGaps(data["Age"])

	# Normalize
	for f in features:
		submSet[f] = normalize(submSet[f])

	Xsubm = submSet[features]
	submPredictions = alg.predict(Xsubm)

	pandas.DataFrame({
	        	"PassengerId": submSet["PassengerId"],
	          	"Survived": submPredictions
	   		}).to_csv('submission' + str(math.floor(time.time())) + '.csv',index=False)

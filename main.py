import time
import preprocess
import math
# ----------------------------------
# Pre-processing: Prepare the data for the algorithm.
# ----------------------------------
import pandas
import numpy as np

data = pandas.read_csv("train.csv")

features = ["Pclass", "Sex", "Age"]
answer = "Survived"

# Replace values of Male with 0 & values of Female with 1
#data.loc[data["Sex"] == "male","Sex"] = 0
#data.loc[data["Sex"] == "female","Sex"] = 1
data["Sex"] = preprocess.genderToBinary(data["Sex"])

# Use the average age for rows without an age
data["Age"] = preprocess.fillAgeGaps(data["Age"])

X = data[features]
Y = data[answer]

#Divide the data into training and cross validation sets.
from sklearn.cross_validation import KFold

folds = KFold(data.shape[0], n_folds=4, random_state=1)
Xtrain = Ytrain = Xval = Yval = []
for train, test in folds:
	Xtrain = X.iloc[train,:]
	YTrain = Y.iloc[train]
	Xval = X.iloc[test,:]
	Yval = Y.iloc[test]

# ----------------------------------
# Train Algorithm: Take the data and compute thetas.
# ----------------------------------
from sklearn import linear_model

logreg = linear_model.LogisticRegression()
logreg.fit(X, Y)

# ----------------------------------
# Cross Validation: Test trained algorithm on the cross validation set and further tweak it.
# ----------------------------------

crossvalPreds = logreg.predict(Xval)
crossvalResults = (crossvalPreds == Yval)
hits = 0
for hit in crossvalResults:
	if hit:
		hits += 1
accuracy = hits / len(Yval)
print(accuracy)
        

# ----------------------------------
# Test Predictions: Run the algorithm on the test set to generate predictions.
# ----------------------------------
test = pandas.read_csv("test.csv")
test["Sex"] = preprocess.genderToBinary(test["Sex"])
test["Age"] = preprocess.fillAgeGaps(data["Age"])

Xtest = test[features]
testPredictions = logreg.predict(Xtest)

pandas.DataFrame({
        	"PassengerId": test["PassengerId"],
        	"Survived": testPredictions
    		}).to_csv('submission' + str(math.floor(time.time())) + '.csv',index=False)

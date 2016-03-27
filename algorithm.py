from sklearn import linear_model
from sklearn import svm

def logReg(Xtrain, Ytrain, Xval, Yval):
	'''
	best_C = tweakLogreg(Xtrain, Ytrain, Xval, Yval, 0, 1.5, 25)
	print(best_C)
	'''

	alg = linear_model.LogisticRegression(C=7.5)
	alg.fit(Xtrain, Ytrain)

	return [crossVal(alg, Xval, Yval), alg]

def SVMSigmoid(Xtrain, Ytrain, Xval, Yval):
	'''
	best = tweakSVM(Xtrain, Ytrain, Xval, Yval, 0, 2, 5)
	best_C = best[0]
	best_gamma = best[1]
	print(best_C)
	print(best_gamma)
	'''

	alg = svm.SVC(C=4, gamma=8)
	alg.fit(Xtrain, Ytrain)

	return [crossVal(alg, Xval, Yval), alg]
 
def crossVal(alg, Xval, Yval):
	crossvalPreds = alg.predict(Xval)
	crossvalResults = (crossvalPreds == Yval)
	hits = 0
	for hit in crossvalResults:
		if hit:
			hits += 1
	return hits / len(Yval)


def tweakLogreg(Xtrain, Ytrain, Xval, Yval, bottom = 1, interval = 3, its = 5):
	C = []
	for i in range(1,its + 1):
		C.append(1 / (10 ** bottom) * (interval ** i))

	best_accuracy = 0
	best_C = None

	for i in C:
		alg = linear_model.LogisticRegression(C=i)
		alg.fit(Xtrain, Ytrain)

		accuracy = crossVal(alg, Xval, Yval)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_C = i

	return best_C

def tweakSVM(Xtrain, Ytrain, Xval, Yval, bottom = 1, interval = 3, its = 5):
	C = []
	for i in range(1,its + 1):
		C.append(1 / (10 ** bottom) * (interval ** i))
	gamma = C

	best_accuracy = 0
	best_C = None
	best_gamma = None

	for i in C:
		for j in gamma:

			alg = svm.SVC(C=i, gamma=j)
			alg.fit(Xtrain, Ytrain)

			accuracy = crossVal(alg, Xval, Yval)

			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_C = i
				best_gamma = j

	return [best_C, best_gamma]
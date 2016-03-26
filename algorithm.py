from sklearn import linear_model
from sklearn import svm

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
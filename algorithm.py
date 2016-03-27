from sklearn import linear_model
from sklearn import svm

def logReg(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
	best_C = tweakLogreg(Xtrain, Ytrain, Xval, Yval, 2, 3, 15)
	print("Best C = " + str(best_C))

	alg = linear_model.LogisticRegression(C=best_C)
	alg.fit(Xtrain, Ytrain)

	return [accuracy(alg, Xval, Yval), alg]

def SVMSigmoid(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
	best = tweakSVM(Xtrain, Ytrain, Xval, Yval, 2, 3, 6)
	best_C = best[0]
	best_gamma = best[1]
	print("Best C = " + str(best_C))
	print("Best gamma = " + str(best_gamma))

	alg = svm.SVC(C=best_C, gamma=best_gamma)
	alg.fit(Xtrain, Ytrain)

	return [accuracy(alg, Xtest, Ytest), alg]
 
def accuracy(alg, X, Y):
	preds = alg.predict(X)
	results = (preds == Y)
	hits = 0
	for hit in results:
		if hit:
			hits += 1
	return hits / len(Y)


def tweakLogreg(Xtrain, Ytrain, Xval, Yval, bottom = 1, interval = 3, its = 5):
	C = []
	for i in range(1,its + 1):
		C.append(1 / (10 ** bottom) * (interval ** i))

	best_accuracy = 0
	best_C = None

	for i in C:
		alg = linear_model.LogisticRegression(C=i)
		alg.fit(Xtrain, Ytrain)

		acc = accuracy(alg, Xval, Yval)

		if acc > best_accuracy:
			best_accuracy = acc
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

			acc = accuracy(alg, Xval, Yval)

			if acc > best_accuracy:
				best_accuracy = acc
				best_C = i
				best_gamma = j

	return [best_C, best_gamma]
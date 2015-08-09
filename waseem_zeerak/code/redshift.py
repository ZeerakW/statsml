import numpy as np
from prep import meanFree
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

class Regression(object):
    def __init__(self, train, test):
        self.trainX = train[:,:-1]
        self.trainY = train[:,-1]
        self.testX  = test[:,:-1]
        self.testY  = test[:,-1]

    def classify(self, clf, params=None):
        if params != None:
            # Normalise in case of non-linear regression and cross validate
            clf = GridSearchCV(clf, param_grid = params, cv = 5)
            
            clf.fit(self.trainX, self.trainY)
            testPred = clf.predict(self.testX)
            trainPred = clf.predict(self.trainX)
        else:
            clf.fit(self.trainX, self.trainY)
            testPred = clf.predict(self.testX)
            trainPred = clf.predict(self.trainX)

        return testPred, trainPred, clf

    def scorer(self, score, pred, labels):
        err = score(labels, pred)

        return err

    def design_matrix(self, matrix):
        col = np.ones((len(matrix), 1))
        return np.hstack((col, matrix))

    # Remember the bias after calculating the inverse
    def calc_inv(self, design, labels):
        inv = np.linalg.pinv(np.dot(design.T, design))
        lab = np.dot(design.T, labels.reshape(-1, 1))

        return np.dot(inv, lab)

    def run(self):
        ### Linear regression (Q1)
        design = self.design_matrix(self.trainX)
        inv    = self.calc_inv(design, self.trainY)
        print  "Parameters of the model\n%s\n" % str(inv)

        # Get the model
        lrTestPred, lrTrainPred, _ = self.classify(LinearRegression())

        # Get the errors
        print "Errors for the linear regression"
        lrTrainErr = self.scorer(mean_squared_error, lrTrainPred, self.trainY)
        print "Mean Squared training error is %f" % lrTrainErr
        lrTestErr = self.scorer(mean_squared_error, lrTestPred, self.testY)
        print  "Mean Squared test error is %f\n" % lrTestErr
        
        print "Non Linear regressions. Normalised training and test set\n"

        ### Non-linear Regression (Q2)

        ## KNN Regression
        # Get the model
        print "Nearest Neighbour Regression\n"
        knnTestPred, knnTrainPred, clf = self.classify(KNeighborsRegressor(), 
               {'n_neighbors': [5, 7, 9, 11, 13]})
        

        # Get the errors
        print "Errors using Nearest Neighbours"
        knnTrainErr = self.scorer(mean_squared_error, knnTrainPred, self.trainY)
        print "Mean Squared training error is %f" % knnTrainErr
        knnTestErr = self.scorer(mean_squared_error, knnTestPred, self.testY)
        print  "Mean Squared test error is %f\n" % knnTestErr

        ## Tree Regression
        # Create the model
        print "Tree Regression"

        scores = []
        for i in range(10):
            treeTestPred, treeTrainPred, clf = self.classify(DecisionTreeRegressor(), 
                {'max_features':['auto', 'sqrt', 'log2']})
            print clf.best_params_
            # Calculate errors the errors
            treeErrTrain = self.scorer(mean_squared_error, treeTrainPred, self.trainY)
            print "Mean Squared training error is %f" % treeErrTrain
            treeErrTest = self.scorer(mean_squared_error, treeTestPred, self.testY)
            print "Mean Squared test error is %f" % treeErrTest
            scores.append(treeErrTest)
        print "\nMean of ten iterations of scores: %s" % str(sum(scores) / float(len(scores)))

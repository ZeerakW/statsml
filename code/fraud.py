import prep
import numpy as np
import pylab as plt
from sklearn.cluster import KMeans, k_means
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, zero_one_loss

class Classification(object):
    def __init__(self, train, test, linear, nonlinear, multi = False):
        self.trainX    = train[:,:-1]
        self.trainY    = train[:,-1]
        self.testX     = test[:,:-1]
        self.testY     = test[:,-1]
        self.linear    = linear
        self.nonlinear = nonlinear
        self.multi     = multi
        plt.rc('text', usetex=True)

    def model_select(self, clf):
        gclf = GridSearchCV(clf, param_grid = {'C': [0.001,0.01,0.1,1,10], 
            'gamma': [0.0001, 0.001,0.01,0.1,1]}, cv = 5)
        gclf.fit(self.trainX, self.trainY)

        return gclf
    
    def classify(self, model, test_y, test_x):
        pred = model.predict(test_x)
        if not self.multi:
            rec, spec, acc = self.score(pred, test_y)
            return rec, spec, acc
        else:
            return 1 - zero_one_loss(test_y, pred)

    def score(self, pred, y_true):
        # Sensitivity: TP / TP + FN == Recall
        rec  = recall_score(y_true, pred)
        spec = self.specificity(y_true, pred)
        acc  = accuracy_score(y_true, pred)

        return rec, spec, acc

    def specificity(self, y_true, pred):
        # Specificity: TN / TN + FP
        neg, tn = 0, 0
        for i in range(len(y_true)):
            if float(y_true[i]) == 0.0:
                neg += 1
                if pred[i] == y_true[i]:
                    tn += 1

        return tn / float(neg)

    def calc_means(self, data):
        mean = np.zeros(21)
        for elem in data:
            mean += elem

        return mean / len(data)

    def calc_cov(self, data, mean):
        out = 0
        for elem in data:
            d = elem - mean
            out += np.outer(d, d)

        return 1.0 / len(data) * out

    def do_sort(self, eigw, eigv):
        tmp = [(abs(eigw[i]), eigv[:,i]) for i in range(len(eigw))]
        return sorted(tmp, reverse = True, key = lambda tuple: tuple[0])


    def project(self, data, component):
        x = [np.dot(component[0], elem.reshape(-1,1)) for elem in data]
        y = [np.dot(component[1], elem.reshape(-1,1)) for elem in data]
        
        return x, y

    def prep_pca(self, data):
        norm, _ = prep.meanFree(data, data)
        mean = self.calc_means(norm)
        cov = self.calc_cov(norm, mean)
        
        eigw, eigv = np.linalg.eig(cov)
        eigens = self.do_sort(eigw, eigv)
        proj_pc = [eigens[0][1], eigens[1][1]]
        x, y = self.project(norm, proj_pc)
        pc = np.hstack((eigens[0][1].reshape(-1,1), eigens[1][1].reshape(-1,1)))

        return eigw, x, y, pc, norm

    def get_clusters(self, clf, normX):
        clf.fit_transform(normX, self.trainY)
        return clf.cluster_centers_

    def run(self):
        ### Q3
        # Select the models
        lin_mod       = self.model_select(self.linear)
        non_lin_mod   = self.model_select(self.nonlinear)
        
        # Get the scores (recall, specificity, accuracy) for test set
        lin_score     = self.classify(lin_mod, self.testY, self.testX)
        non_lin_score = self.classify(non_lin_mod, self.testY, self.testX)
        train_lin_score = self.classify(lin_mod, self.trainY, self.trainX)
        train_non_lin_score = self.classify(non_lin_mod, self.trainY, self.trainX)

        print "Scores on prediction on test set\n"
        print "Scores for the linear model"
        if not self.multi:
            print "Recall: %f\nSpecificity: %f\nAccuracy: %f\n" % (lin_score[0], 
                    lin_score[1], lin_score[2])

            print "Scores for the non-linear model"
            print "Recall: %f\nSpecificity: %f\nAccuracy: %f\n" % (non_lin_score[0], 
                    non_lin_score[1], non_lin_score[2])
        
            print "Scores on prediction on the training set"

            print "Scores for the linear model"
            print "Recall: %f\nSpecificity: %f\nAccuracy: %f\n" % (train_lin_score[0], 
                    train_lin_score[1], train_lin_score[2])

            print "Scores for the non-linear model"
            print "Recall: %f\nSpecificity: %f\nAccuracy: %f\n" % (train_non_lin_score[0], 
                    train_non_lin_score[1], train_non_lin_score[2])
            
            ### Q4
            eig, x, y, vects, normX = self.prep_pca(self.trainX)
            plt.plot(range(1, len(eig) + 1), eig)
            plt.title("Eigenspectrum")
            plt.show()

            plt.plot(x, y, 'x')
            plt.xlabel("1st Principle Component")
            plt.ylabel("2nd Principle Component")
            plt.title("Principle Components")
            plt.show()

            ### Q5
            centers = self.get_clusters(KMeans(n_clusters=2), normX)
            print "Centers\n%s" % str(centers)
            
            proj_cent = np.dot(vects.T, centers.T)
            plt.plot(proj_cent[0], proj_cent[1], 'ro', label = 'centers')
            plt.plot(x, y, 'x', label = 'Principle components')
            plt.title('cluster centers projected on principle components')
            plt.legend(loc='best')
            plt.show()

        else:
            ### Q6
            print "Classification error %f\n" % lin_score
            
            print "Scores for non-linear model"
            print "Classification error %f\n" % non_lin_score

            print "Scores on prediction on the training set"
            print "Classification error on Linear model: %f\n" % train_lin_score
            print "CLassification error on non-linear model: %f\n" % train_non_lin_score

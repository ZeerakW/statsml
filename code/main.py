import sys
import fraud
import redshift
from prep import readFiles
from sklearn.svm import SVC

def main():
    # Read in files
    redshiftTrain, redshiftTest = readFiles('data/redshiftTrain.csv', 'data/redshiftTest.csv')
    binKeyTrain, binKeyTest = readFiles('data/keystrokesTrainTwoClass.csv', 
            'data/keystrokesTestTwoClass.csv', delim = ',')
    multKeyTrain, multKeyTest = readFiles('data/keystrokesTrainMulti.csv',
            'data/keystrokesTestMulti.csv', delim = ',')

    # Init regression
    rs = redshift.Regression(redshiftTrain, redshiftTest)
    
    # Init Binary classification
    binKS = fraud.Classification(binKeyTrain, binKeyTest,\
            SVC(kernel='linear'), SVC(kernel='rbf'), multi = False)

    # Init Multiclass classification
    multKS = fraud.Classification(multKeyTrain, multKeyTest,\
            SVC(kernel='linear'), SVC(kernel='rbf'), multi = True)

    try:
        if sys.argv[1].lower() == 'regression':
            # Run regression (Q1, Q2)
            rs.run()

        elif sys.argv[1].lower() == 'binary':
            # Run binary classification and PCA (Q3 - Q5)
            print "\nBegin classification tasks\n"
            print "Binary classification\n"
            binKS.run()

        elif sys.argv[1].lower() == 'multiclass':
            # Run Multiclass classification (Q6)
            print "Multiclass classification\n"
            multKS.run()
    except:
        # Run all (Q1 - Q6)
        rs.run()
        binKS.run()
        multKS.run()

if __name__ == '__main__':
    main()

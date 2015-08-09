import numpy as np
import pandas as pd

def readFiles(train_f, test_f, delim=" "):
   train = pd.read_csv(train_f, sep = delim) 
   test = pd.read_csv(test_f, sep = delim)

   return np.array(train), np.array(test)

def calcMeanVar(data):
    mean = sum(data) / float(len(data))
    var = sum((data - mean) ** 2) / len(data)
    return mean, var

def meanFree(data, test):
    mean, var = calcMeanVar(data)
    train = (data - mean) / np.sqrt(var) 
    test = (test - mean) / np.sqrt(var)
    return train, test

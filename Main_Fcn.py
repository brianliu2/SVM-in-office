from numpy import *
import numpy as np
from pandas import *
import pandas as pd
from importlib import reload
from SVM_Fcn import *

dataFile = '/Users/xliu/Documents/MRC/Work/Program/Python/machineLearningInAction' \
           '/machinelearninginaction-master/Ch06/testSet.txt'

datMat, datLabels = createDataset(dataFile)
b, alphas = simpleSMO(datMat, datLabels, 0.6, 0.001, 40)
print(b)
print(alphas)
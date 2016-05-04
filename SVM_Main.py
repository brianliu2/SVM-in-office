from numpy import *
from SVM_Fcn import *
from FullSMO import *
import os

datPath = ('S:\program\Python\Machine Learning in Action\Ch06\\')
fileName = 'testSet.txt'
datMat, datLabel = creatDat(datPath, fileName)
const = 0.6
tolerance = 0.0001
MaxIter = 40
#b, alphas = simpleSMO(datMat, datLabel, const, tolerance, MaxIter)
b, alphas = SMOFull(datMat, datLabel, const, tolerance, MaxIter)
print(b)
print(shape(alphas[alphas>0]))

for i in range(100):
    if(alphas[i] > 0):
        print(datMat[i], datLabel[i])

ws = calcWs(alphas, datMat, datLabel)
print(ws)

datMat = mat(datMat)
datLabel = mat(datLabel)

rowNum = shape(datMat)[0]
#cmpVec = zeros((rowNum, 1))

correctCnt = 0
for n in range(rowNum):
    pred = datMat[n] * mat(ws) + b
    if(sign(pred.item(0)) == sign(datLabel.item(n))):
        correctCnt += 1
correctRate = correctCnt / rowNum
print(correctRate)

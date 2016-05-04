import os
from numpy import *


def find_nearest(dataMat, value):
    dim = shape(dataMat)[1]
    valAndIdx = zeros((dim, 2))
    for n in range(dim):
        idx = (abs(dataMat[:, n] - value[n])).argsort()[0]
        valAndIdx[n, :] = ([MCMCSamples[idx, n], int(idx)])
    return valAndIdx

def find_nearest2(dataMat, value, neighbourNum, stepSize):
    dim = shape(dataMat)[1]
    selctDat= zeros((neighbourNum, dim))
    for n in range(dim):
        idx = (abs(dataMat[:, n] - value[n])).argsort()
        for i in range(neighbourNum):
            selctDat[i, n] = dataMat[idx[i*stepSize], n]
    return selctDat


MCMCPost = ('H:\Document\Program\emulator\Modified Version\marian'
            '\JASA_2014_code\emulator_differentLogTransformation\\')
fileName = 'MCMC_Posterior.txt'

MCMCFile = open(MCMCPost+fileName, 'r')

rowNum = 10000
colNum = 6

MCMCSamples = zeros((rowNum, colNum))

rowIdx = 0

for line in MCMCFile.readlines():
    lineSplit = line.split('\t')
    lineSplitArr = array(lineSplit)
    for n in range(colNum):
        MCMCSamples[rowIdx, n] = float(lineSplitArr[n])
    rowIdx += 1

trueParVal = array([0.2868, 0.1057, 3.449, -13.41, 0.3333, 0.6667])
neighbourNum = 20
stepSize = 80

valAndIdx = find_nearest(MCMCSamples, trueParVal)
selctDat = find_nearest2(MCMCSamples, trueParVal, neighbourNum, stepSize)


savetxt(MCMCPost+'selectMCMC.txt', selctDat, delimiter=' ')

#fid = open(MCMCPost+'selectMCMC.bin', 'wb')
#selctDat.tofile(fid, sep=" ")
#fid.close()

print(selctDat)


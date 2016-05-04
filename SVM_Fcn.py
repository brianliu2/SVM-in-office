from numpy import *
import numpy as np
from pandas import *
import pandas as pd

def createDataset(dataFile):
    datMat = []
    datLabels = []
    datFile = open(dataFile, 'r')

    for line in datFile.readlines():
        dat = line.strip().split('\t')
        datMat.append([float(dat[0]), float(dat[1])])
        datLabels.append(float(dat[-1]))

    return datMat, datLabels

def drawIndexJ(i, m):
    j = i
    while(j == i):
        j = int(random.randint(0, m))
    return j

def clipAlphaVal(aj, low, upper):
    if aj > upper:
        aj = upper
    elif aj < low:
        aj = low
    return aj

def simpleSMO(datMat, datLabel, constant, tolerance, MaxIter):
    datMat = mat(datMat)
    datLabel = mat(datLabel).transpose()
    rowNum = shape(datMat)[0]
    colNum = shape(datMat)[1]
    alphas = zeros((rowNum, 1))
    b = 0
    alphaPairsChanged = 0
    iter = 0
    while iter < MaxIter:
        activeCnt = 0
        for i in range(rowNum):
            fxi = float(multiply(alphas, datLabel).T * (datMat * datMat[i, :].T)) + b
            Ei = fxi - float(datLabel[i])
            if ((datLabel[i] * Ei < -tolerance) and (alphas[i] < constant)) or \
                ((datLabel[i] * Ei > tolerance) and (alphas[i] > 0)):
                j = drawIndexJ(i, MaxIter)
                fxj = float(multiply(alphas, datLabel).T * (datMat * datMat[j, :].T)) + b
                Ej = fxi - float(datLabel[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (datLabel[i] != datLabel[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(constant, constant + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - constant)
                    H = min(constant, alphas[j] + alphas[i])
                if L == H:
                    print('Lower bound equals to Upper bound')
                    continue

                # calculate the eta using for updating alphas[i] and alphas[j]
                eta = 2.0 * datMat[i, :] * datMat[j, :].T - datMat[i, :] * datMat[i, :].T - datMat[j, :] * datMat[j, :].T
                if eta >= 0:
                    print('eta equals %.2f, and it is greater than 0 so jump over loop' % eta)
                    continue

                # Update alpha[i] and alpha[j]
                alphas[j] = alphas[j] - datLabel[j] * (Ei - Ej) / eta
                #alphas[j] -= datLabel[j] * (Ei - Ej) / eta
                alphas[j] = clipAlphaVal(alphas[j], L, H)

                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue

                # update i by the same amount as j
                # the update is in the opposite direction
                alphas[i] = alphas[i] + datLabel[j] * datLabel[i] * (alphaJold - alphas[j])
                #alphas[i] = alphas[i] + datMat[j] * datMat[i] * (alphaJold - alphas[j])

                # Update b through b1 and b2
                b1 = b - Ei - datLabel[i] * (alphas[i] - alphaIold) * \
                              datMat[i, :] * datMat[i, :].T - \
                     datLabel[j] * (alphas[j] - alphaJold) * datMat[i, :] * datMat[j, :].T
                b2 = b - Ej - datLabel[i] * (alphas[i] - alphaIold) * \
                              datMat[i, :] * datMat[j, :].T - \
                     datLabel[j] * (alphas[j] - alphaJold) * datMat[j, :] * datMat[j, :].T

                if (0 < alphas[i]) and (constant > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (constant > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0

                alphaPairsChanged += 1
                print('iter: {0:d} i:{1:d}, pairs changed {2:d}'.format(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas














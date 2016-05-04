from numpy import *
import numpy as np
from pandas import *
import pandas as pd


class datStrue:
    def __init__(self, dataIn, dataLabelIn, const, tolerance):
        self.data = dataIn
        self.label = dataLabelIn
        self.c = const
        self.tol= tolerance
        self.rowNum = shape(dataIn)[0]
        self.alphas = mat(zeros((self.rowNum, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.rowNum, 2)))

def calcError(datStruct, idx):
    fxk = float(multiply(datStruct.alphas, datStruct.label).T * (datStruct.data * datStruct.data[idx, :].T)) + datStruct.b
    Ek = fxk - float(datStruct.label[idx])
    return Ek

def drawIndexJrand(i, m):
    j = i
    while(j == i):
        j = int(random.randint(0, m))
    return j

def drawIndexJ(i, Ei, datStruct):
    j = 0; MaxEr = 0
    datStrue.eCache[i] = [1, Ei]
    # .A is converting matrix to array
    validEcacheList = nonzero(datStrue.eCache[:, 0].A)[0]
    if(len(validEcacheList) != 0):
        for k in validEcacheList:
            if(k == i):
                continue
            Ek = calcError(datStrue, k)
            deltaK = abs(Ei - Ek)
            if(deltaK > MaxEr):
                j = k
                Ej = Ek
                MaxEr = deltaK
    else:
        j = drawIndexJrand(i, datStrue.rowNum)
        Ej = calcError(datStrue, j)
    return j, Ej

def updateEk(datStrue, k):
    Ek = calcError(datStrue, k)
    datStrue.eCache[k, :] = [1, Ek]







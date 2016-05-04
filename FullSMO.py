from numpy import *

# construct a SVM data structure
class SVMdata:
    def __init__(self, dataIn, dataLabel, const, tolerance):
        self.data = dataIn
        self.label = dataLabel
        self.const = const
        self.tol = tolerance
        self.rowNum = shape(dataIn)[0]
        self.alphas = mat(zeros((self.rowNum, 1)))
        self.eCache = mat(zeros((self.rowNum, 2)))
        self.b = 0

# this is the code for calculating error
def calcErr(SVMData, idx):
    fxk = float(multiply(SVMData.alphas, SVMData.label).T * (SVMData.data * SVMData.data[idx, :].T)) + SVMData.b
    Ek = fxk - float(SVMData.label[idx])
    return Ek

# a function to pick up index as j
def Jrand(i, rowNum):
    j = i
    while(j == i):
        j = random.randint(0, rowNum)
    return j

# a function to clip the alpha value based on the lower and upper bound
def clipAlphaVal(alpha, L, H):
    if(alpha < L):
        alpha = L
    elif(alpha > H):
        alpha = H
    return alpha

# this is the function for selecting index j by using the maximum error in cache
def selectJ(SVMData, i):
    # initialize the index j, the error Ej and maximum difference between Ej and Ei
    j = 0; Ej = 0; MaxDeltaErr = 0

    # calculate the Ei with inserting index i
    Ei = calcErr(SVMData, i)

    # store the Ei in the cache and set the indicator to active status
    SVMData.eCache[i, :] = [1, Ei]

    # search through the eCache, and create a list that compose all alphas corresponding to nonzero
    validEcacheList = nonzero(SVMData.eCache[:, 0].A)[0]

    # if all errors are zero, then we pick up index j in random way, otherwise we select the j
    # based on the maximum difference between Ei and Ej
    if(len(validEcacheList) != 0):
        for k in validEcacheList:
            if(k == i):
                continue
            Ek = calcErr(SVMData, k)
            deltaErr = abs(Ek - Ei)
            if(deltaErr > MaxDeltaErr):
                Ej = Ek
                MaxDeltaErr = deltaErr
                j = k
    else:
        j = Jrand(i, SVMData.rowNum)
        Ej = calcErr(SVMData, j)
    return j, Ej

def updateEcache(SVMData, k):
    Ek = calcErr(SVMData, k)
    SVMData.eCache[k, :] = [1, Ek]

# this is the inner loop for SMO
def innerLoop(SVMData, i):
    # we first calculate the Ej to judge if the condition of doing optimization is satisfied
    Ei = calcErr(SVMData, i)

    # Judge the condition is satisfied
    if((Ei*SVMData.label[i] < -SVMData.tol) and (SVMData.alphas[i] < SVMData.const)) or \
      ((Ei*SVMData.label[i] > SVMData.tol) and (SVMData.alphas[i] > 0)):
        # if satisfied, then we select the j index
        j, Ej = selectJ(SVMData, i)

        # store the current alpha[i] and alpha[j]
        alphaJold = SVMData.alphas[j]
        alphaIold = SVMData.alphas[i]

        # update the L and H values depending on if label[i] = label[j]
        if(SVMData.label[i] != SVMData.label[j]):
            L = max(0, SVMData.alphas[j] - SVMData.alphas[i])
            H = min(SVMData.const, SVMData.const + SVMData.alphas[j] - SVMData.alphas[i])
        else:
            L = max(0, SVMData.alphas[j] + SVMData.alphas[i] - SVMData.const)
            H = min(SVMData.const, SVMData.alphas[j] + SVMData.alphas[i])

        # if L value equals to H value, then we escape the current for-loop
        if(L == H):
            print('L == H')
            return 0

        # calculate the step size (i.e. eta) for updating alpha[j]
        eta = 2.0 * SVMData.data[i, :] * SVMData.data[j, :].T - SVMData.data[i, :] * SVMData.data[i, :].T - SVMData.data[j, :] * SVMData.data[j, :].T

        # if the step size is greater than zero then escape
        if(eta >= 0):
            print('eta is greater than zero')
            return 0

        # update alpha[j] by using eta
        SVMData.alphas[j] = SVMData.alphas[j] - SVMData.label[j] * (Ei - Ej) / eta

        # clip the alpha[j] using L and H values
        SVMData.alphas[j] = clipAlphaVal(SVMData.alphas[j], L, H)

        # update the eCache along the new alpha[j] value which is stored in SVMData strcture
        updateEcache(SVMData, j)

        ## if the alpha[j] has not been substantially updated, then we omit the b update as which depends on latest alpha[i] and alpha[j]
        #if(abs(alphaJold - SVMData.alphas[j]) < 0.00001):
        #    print('J makes a tiny move')
        #    #print('%.2f' % abs(alphaJold - SVMData.alphas[j]))
        #    print('alphaJold: %.2f; alphas[j]: %.2f' % (alphaJold, SVMData.alphas[j]))
        #    return 0

        # If the movement of alpha[j] is large enough, then we continue to update b
        b1 = SVMData.b - Ei - SVMData.label[i] * (SVMData.alphas[i] - alphaIold) * SVMData.data[i, :] * SVMData.data[i, :].T - SVMData.label[j] * (SVMData.alphas[j] - alphaJold) * SVMData.data[i, :] * SVMData.data[j, :].T
        b2 = SVMData.b - Ej - SVMData.label[i] * (SVMData.alphas[i] - alphaIold) * SVMData.data[i, :] * SVMData.data[j, :].T - SVMData.label[j] * (SVMData.alphas[j] - alphaJold) * SVMData.data[j, :] * SVMData.data[j, :].T

        # If alpha[i] is within the [L, H], then b1 is set to be b
        if(SVMData.alphas[i] > 0) and (SVMData.alphas[i] < SVMData.const):
            SVMData.b = b1
        elif(SVMData.alphas[j] > 0) and (SVMData.alphas[j] < SVMData.const):
            SVMData.b = b2
        else:
            SVMData.b = (b1 + b2)/2
        return 1
    else:
        return 0

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w

def SMOFull(dataIn, dataLabel, const, tolerance, MaxIter):
    # create a SVMData class object
    SVMData = SVMdata(mat(dataIn), mat(dataLabel).transpose(), const, tolerance)

    # create a iteration counter
    iter = 0

    # create a flag to specify whether the data has been entirely visited
    entireSet = True

    # create a variable to specify whether the alphas have been updated in pair-wise
    alphaPairsChanged = 0

    # Main loop for Full Platt SMO
    while(iter < MaxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        # In every iteration, the alphaPairsChanged variable is reset
        alphaPairsChanged = 0

        # go over all data, i.e. data[i, :]
        if entireSet:
            for i in range(SVMData.rowNum):
                alphaPairsChanged += innerLoop(SVMData, i)
                print('fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((SVMData.alphas.A > 0) * (SVMData.alphas.A < SVMData.const))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLoop(SVMData, i)
                print('non-bound, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        # toggle entire set loop
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' % iter)

        return SVMData.b, SVMData.alphas













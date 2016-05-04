from numpy import *
import os

# Load the data and corresponding labels in Python
def creatDat(datPath, fileName):
    #os.chdir(datPath)
    data = open(datPath+fileName, 'r')
    datMat = []
    datLabel = []
    for line in data.readlines():
        data = line.strip().split('\t')
        datMat.append([float(data[0]), float(data[1])])
        datLabel.append(float(data[-1]))
    return datMat, datLabel

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

# This is the simplified version of SMO, meaning we go through all alphas without selective scheme
def simpleSMO(datMatIn, datLabelIn, const, tolerance, MaxIter):
    # convert datMat and datLabel into numpy.Matrix for multiply convenience
    datMat = mat(datMatIn)

    # convert the data label vector into matrix format and transpose it,
    # this is due to we want to make a element-wise multiplication between
    # data label and alphas which has dimension rowNum * 1
    datLabel = mat(datLabelIn).T

    # retrieve the dimension info of data
    rowNum = shape(datMat)[0]
    colNum = shape(datMat)[1]

    # initialize alphas as zero
    alphas = mat(zeros((rowNum, 1)))

    # initialize b as zero
    b = 0

    # create the iter counter
    iterSVM = 0

    # create the counter for alpha changing in pair
    alphaChangePair = 0

    # Main loop while the loop number is greater than MaxIter, then stop the SMO
    while(iterSVM < MaxIter):
        # In this simplified version, all alphas need to be visited
        for i in range(rowNum):
            # calculate the fxi by incorporating alpha[i] and data[i]
            fxi = float(multiply(alphas, datLabel).T * (datMat * datMat[i, :].T)) + b

            # calculate the error between fxi and label[i]
            Ei = fxi - float(datLabel[i])

            #### judge the availability of updating alpha[j]
            if ((datLabel[i] * Ei < -tolerance) and (alphas[i] < const)) or ((datLabel[i]*Ei > tolerance) and (alphas[i] > const)):
                # if meet the condition of updating alpha[j],
                # then we pick up index j randomly through our helper function
                j = Jrand(i, rowNum)

                # calculate the fxj by incorporating alpha[j] and data[j]
                fxj = float(multiply(alphas, datLabel).T * (datMat * datMat[j, :].T)) + b

                # calculate the error between fxj and label[j]
                Ej = fxj - float(datLabel[j])

                # record the current values of alpha[i] and alpha[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                ### calculate the values of Lower and upper bounds depending on various conditions
                if(datLabel[i] != datLabel[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(const, const + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - const)
                    H = min(const, alphas[j] + alphas[i])

                ### escape the for-loop if the L = H
                if(L == H):
                    print('L == H')
                    continue

                ### calculate the step size (eta) for moving alpha[j]
                eta = 2.0 * datMat[i, :] * datMat[j, :].T - datMat[i, :] * datMat[i, :].T - datMat[j, :] * datMat[j, :].T

                # if eta is greater or equals to zero, then escape the for-loop
                if(eta >= 0):
                    print('eta >=')
                    continue
                # update alpha according to eta value
                alphas[j] = alphas[j] - datLabel[j] * (Ei - Ej)/eta

                # clip alpha[j] value through L and H values
                alphas[j] = clipAlphaVal(alphas[j], L, H)

                # determine if the step move ahead of alpha[j] is too small
                # if so, we escape the current for-loop with out updating alpha[i]
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('J move is too small')
                    continue
                # if the step move ahead of alpha[j] is large enough,
                # we need to update alpha[i], as they are in pair-wise
                alphas[i] = alphas[i] + datLabel[i] * datLabel[j] * (alphaJold - alphas[j])

                #### we eventually update the b component through b1 and b2
                b1 = b - Ei - datLabel[i] * (alphas[i] - alphaIold) * datMat[i, :] * datMat[i, :].T - datLabel[j] * (alphas[j] - alphaJold) * datMat[i, :] * datMat[j, :].T
                b2 = b - Ej - datLabel[i] * (alphas[i] - alphaIold) * datMat[i, :] * datMat[j, :].T - datLabel[j] * (alphas[j] - alphaJold) * datMat[j, :] * datMat[j, :].T

                # assign b1 or b2 or its average to b, conditioning on if alpha[i] is within interval
                # if not, we determine by looking if alpha[j] is within interval
                if(0 < alphas[i]) and (alphas[i] < const):
                    b = b1
                elif(0 < alphas[j]) and (alphas[j] < const):
                    b = b2
                else:
                    b = (b1+b2)/2

                # if we successfully updated alpha[i] and alpha[j] in pair
                # we add the counter by one
                alphaChangePair += 1
                print('iter: %d i:%d, pairs changed %d' % (iterSVM, i, alphaChangePair))

        if(alphaChangePair == 0):
            iterSVM += 1
        else:
            iterSVM = 0
        print('iteration number: %d' % iterSVM)
    return b, alphas

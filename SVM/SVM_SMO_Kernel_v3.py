#-*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
import random

def loadDataset(fileName):
    '''
    说明：读入数据;并转化为np.array

    Arguments：
    fileName - 数据文件名

    Returns：
    dataMat - 数据集和：X
    labelMat - 数据集合：Y
    '''
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    dataArr = np.array(dataMat)
    m = np.shape(labelMat)
    labelArr = np.array(labelMat)
    labelArr = labelArr.reshape((m[0],1))

    return dataArr, labelArr

def showDataSet(dataMat, labelMat):
    '''
    Declaration:
    画出正负样本图形

    :param dataMat: 样本X
    :param labelMat: 样本Y
    :return: 无
    '''
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

def clipAlpha(alpha, H, L):
    '''
    修剪alpha的数值
    :param alpha: 输入alpha
    :param H: 上限
    :param L: 下限
    :return: 修剪后的 alpha_clipped
    '''
    if alpha>H :
        alpha = H
    if alpha<L :
        alpha = L
    return alpha

def selectJ(i, m):
    '''
    随机选择alpha的index：i以外的j， 范围（0，m ）
    :param i: index for alpha
    :param m: 范围
    :return: index J
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def get_w(oS):
    '''
    根据结果，求得w
    :param alphas:
    :param dataArr:
    :param labelArr:
    :return:  w
    '''
    w = np.dot((oS.alphas*oS.labelArr).T, oS.X)
    return w

def get_b(oS):
    '''
    根据所有支持向量，求b的平均值
    :param alphas:
    :param dataArr:
    :param labelArr:
    :return: b
    '''
    index_set = np.where( oS.alphas > 1e-8)
    _,num = np.shape(index_set)
    w = get_w(oS)

    b_sum = 0
    for i in range(num):
        index = index_set[0][i]
        b_sum += oS.labelArr[index][0] - np.dot(w, oS.X[index].T)
    b = b_sum/num
    return b

def kernelTrans(X, A, kTup):
    m,_ = np.shape(X)
    K = np.zeros((m,1))
    if kTup[0] == 'linear':
        K = np.dot(X, A.T)
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = np.dot(deltaRow, deltaRow.T)
        K = np.exp(K/(-2.0*np.power(kTup[1], 2)))
    else: raise NameError('Kernel is not recongnized')
    return K

class  optStruct :
    def __init__(self, dataArr, labelArr, C, toler, kTup):
        self.X = dataArr
        self.labelArr = labelArr
        self.C = C
        self.toler = toler
        self.m, self.n = np.shape(dataArr)
        self.eCache = np.zeros((self.m, 2))
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.K = np.zeros((self.m, self.m))
        self.fx = np.zeros((self.m, 1))
        self.kTup = kTup
        for i in range(self.m):
            for j in range(self.m):
                self.K[j,i] = kernelTrans(self.X, self.X[i,:], kTup)[j]

def fX(oS, k):
    '''
    计算f(xk)
    :param oS:
    :param k:
    :return:
    '''
    fXk = np.dot((oS.alphas*oS.labelArr).T, oS.K[:,k]) + oS.b
    return np.float64(fXk)

def calcEk(oS, k):
    '''
    计算误差 Ek
    :param oS:
    :param k:
    :return:
    '''
    fXk = fX(oS, k)
    Ek = fXk - oS.labelArr[k]
    return np.float64(Ek)

def update(oS, k):
    '''
    更新误差缓存 oS.eCache
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def findJ(i, oS, Ei):
    '''
    应用启发式方法，获得第二个alpha：j
    选择两个alpha的误差差值最大对应的i，j
    :param i:
    :param oS:
    :param Ei:
    :return:  j Ej
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0

    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList > 1):
        for k in validEcacheList:
            if k==i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        j = maxK
    else:
        j = selectJ(i, oS.m)
        #Ej = calcEk(oS, j)
    return j

def smoAlg_Kernel(dataArr, labelArr, C, toler, kTup, maxIter):
    '''
    分割超平面wx + b = 0, 求该超平面的w b, 启发式选择alpha对
    :param dataArr: 训练样本X
    :param labelArr: 训练样本标签Y
    :param C: 为一个给定常数，通过调节不同参数，获得不同结果
    :param toler: 松弛变量toler>=0
    :param maxIter: 最大迭代次数
    :return:
    '''
    oS = optStruct(dataArr, labelArr, C, toler, kTup)

    iter = 0
    iter_done = 0
    while (iter<maxIter):
        alphaPairsChanged = 0 #每次循环（1，m）中，alpha对的更新数量
        for i in range(oS.m):
            yi = oS.labelArr[i, 0]
            fxi = fX(oS, i)

            #若违背KKT条件， alpha值进行更新
            bool1 = (oS.alphas[i] < 0) or (oS.alphas[i] > oS.C)
            bool2 = 1 - yi*fxi > oS.toler
            bool3 = abs(oS.alphas[i]*(1 - yi*fxi - oS.toler)) > 1e-8
            if bool1 or bool2 or bool3:
                Ei = fxi - yi

                j = findJ(i, oS, Ei)
                yj = oS.labelArr[j, 0]
                Ej = fX(oS, j) - yj

                alphaI_old = oS.alphas[i,0]
                alphaJ_old = oS.alphas[j,0]

                if abs(yi - yj)< 1e-8:
                    L = max(0.0, alphaI_old + alphaJ_old -oS.C)
                    H = min(oS.C, alphaI_old + alphaJ_old)
                else:
                    L = max(0.0, alphaJ_old - alphaI_old)
                    H = min(oS.C, oS.C + alphaJ_old - alphaI_old)

                xi = oS.X[i]
                xj = oS.X[j]

                eta = oS.K[i,i] + oS.K[j,j] - 2*oS.K[i,j]
                if abs(eta)<1e-8: continue #eta == 0

                alphaJ_new_temp = alphaJ_old + yj*(Ei - Ej)/eta
                alphaJ_new = clipAlpha(alphaJ_new_temp, H, L)
                oS.alphas[j] = alphaJ_new
                update(oS, j)

                alphaI_new = alphaI_old + yi*yj*(alphaJ_old - alphaJ_new)
                oS.alphas[i] = alphaI_new
                update(oS, i)

                #若alpha对，更新较小，则不计更新次数alphaPairsChange
                if (abs(alphaI_new - alphaI_old) < 1e-10) and \
                        (abs(alphaJ_new - alphaJ_old) < 1e-10):
                    continue

                b1 = oS.b - Ei - yi*oS.K[i,i]*(alphaI_new - alphaI_old) - \
                    yj*oS.K[i,j]*(alphaJ_new - alphaJ_old)
                b2 = oS.b - Ej - yi*oS.K[i,j]*(alphaI_new - alphaI_old) - \
                    yj*oS.K[j,j]*(alphaJ_new - alphaJ_old)
                if (alphaI_new>0) and (alphaI_new<oS.C): oS.b = b1
                elif (alphaJ_new>0) and (alphaJ_new<oS.C): oS.b = b2
                else: oS.b = 0.5*(b1 + b2)

                alphaPairsChanged += 1
        iter += 1
        print('[iter {0:2}: alphaPairsChanged = {1:3}]'.format(iter, alphaPairsChanged))

        if alphaPairsChanged != 0:
            iter_done = 0

        #连续5次迭代，没有任何alpha更新，则退出迭代
        if alphaPairsChanged != 0:
            iter_done = 0
        if alphaPairsChanged <= 0:
            iter_done += 1
        if iter_done >= 5 :
            print('last_iter:', iter)
            break

    if iter >= maxIter: print('last_iter', iter)

    oS.b = get_b(oS)

    return oS

def fX_LIN(oS, testX):
    '''
    线性核，求测试集的fX
    :param oS:
    :param testX:
    :return:
    '''
    m_t, _ = testX.shape
    w = get_w(oS)
    fX_LIN = (np.dot(w, testX.T) + oS.b).T
    return fX_LIN

def fX_RBF(oS, testX):
    '''
    RBF 高斯核 求fX
    :param oS:
    :param testX:
    :return:
    '''
    m_t, _ = testX.shape
    fX_RBF = []
    for i in range(m_t):
        deltaRow = oS.X - testX[i,:]
        deltaX1 = deltaRow[:,0].reshape(oS.m, 1)
        deltaX2 = deltaRow[:,1].reshape(oS.m, 1)
        norm_i2 = np.power(deltaX1,2) + np.power(deltaX2, 2)
        Ki = np.exp(-norm_i2/(2.0*np.power(oS.kTup[1], 2)))
        fXi = np.sum(oS.alphas*oS.labelArr*Ki) + oS.b
        fX_RBF.append(fXi[0])
    fX_RBF = np.array(fX_RBF).reshape(m_t, 1)
    return fX_RBF

def testPerformance(oS, testX, testY):
    '''
    测量训练好的SVM模型，在测试集的表现
    :param oS:
    :param testX:
    :param testY:
    :return:
    '''
    m_t, _ = testX.shape
    if oS.kTup[0] == 'linear':
        fX_test = fX_LIN(oS, testX)
    elif oS.kTup[0] == 'rbf':
        fX_test = fX_RBF(oS, testX)
    else:
        raise NameError('Kernel is not recongnized')

    ErrorLocations = np.where(fX_test*testY < -1e-8)
    ErrorNum = len(ErrorLocations[0])
    ErrorRate = ErrorNum/m_t
    print('Error rate in this set is: {0:.2%}'.format(ErrorRate))

    ErrorPoints = []
    for k in range(ErrorNum):
        ErrorPoints.append(testX[ErrorLocations[0][k]])
    ErrorPoints = np.array(ErrorPoints)
    showTestSet(testX, testY, ErrorPoints)

def showTestSet(testX, testY, ErrorPoints):
    '''
    Declaration:
    画出正负样本图形
    画出错误分类的点

    :param dataMat: 样本X
    :param labelMat: 样本Y
    :return: 无
    '''
    data_plus = []
    data_minus = []
    for i in range(len(testX)):
        if testY[i] > 0:
            data_plus.append(testX[i])
        else:
            data_minus.append(testX[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c=(1,0.5,0))
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c=(0,0.5,0.8))

    if len(ErrorPoints)>=1:
        plt.scatter(ErrorPoints.T[0], ErrorPoints.T[1], s=100, c='r', marker='x', \
                    alpha=0.7, linewidth=3, edgecolor='red')

    plt.xlabel("x1")
    plt.ylabel('x2')
    plt.legend('PNW')

    plt.show()

if __name__ == '__main__':
    dataArr,labelArr = loadDataset('TrainingSet_RBF.txt')      #线性不可分训练集
    #dataArr,labelArr = loadDataset('TrainingSet_Linear.txt')  #线性可分训练集
    #showDataSet(dataArr, labelArr)

    time_start = time.time()
    #kTup = ('linear',)
    kTup = ('rbf', 0.263)#0.263
    oS = smoAlg_Kernel(dataArr, labelArr, 200, 1e-4, kTup,800)
    time_end = time.time()
    print('Training time: {0:.3}s'.format(time_end - time_start))

    #训练集表现
    print('{0:-^34}'.format('Training Set'))
    testPerformance(oS, dataArr, labelArr)

    #测试集表现
    testX,testY = loadDataset('TestSet_RBF.txt')
    print('{0:-^34}'.format('Test Set'))
    testPerformance(oS, testX, testY)

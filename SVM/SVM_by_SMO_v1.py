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

def fx(dataArr, labelArr, index, alphas, b):
    '''
    计算 f(xi)
    :param dataArr: 样本X (m,n)
    :param labelArr: Y (m,1)
    :param index:  index
    :param alphas: alpha (m,1)
    :param b: b float
    :return: f(xi)
    '''
    w = get_w(alphas, dataArr, labelArr)
    fx_index = np.dot(w, dataArr[index].T) + b
    return np.float64(fx_index)

def get_w(alphas, dataArr, labelArr):
    '''
    根据结果，求得w
    :param alphas:
    :param dataArr:
    :param labelArr:
    :return:  w
    '''
    ayi = alphas*labelArr
    w = np.dot(ayi.T, dataArr)
    return w

def get_b(alphas, dataArr, labelArr):
    '''
    根据所有支持向量，求b的平均值
    :param alphas:
    :param dataArr:
    :param labelArr:
    :return: b
    '''
    index_set = np.where( alphas > 1e-8)
    _,num = np.shape(index_set)
    w = get_w(alphas, dataArr, labelArr)

    b_sum = 0
    for i in range(num):
        index = index_set[0][i]
        b_sum += labelArr[index][0] - np.dot(w, dataArr[index].T)
    b = b_sum/num
    return b

def showClassifer(dataMat, classLabels, w, b):
    """
    分类结果可视化
    Parameters:
        dataMat - 数据矩阵
        w - 直线法向量
        b - 直线解决
    Returns:
        无
    """
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat[:, 0])
    x2 = min(dataMat[:, 0])
    a1, a2 = w[0][0], w[0][1]
    b = float(b)

    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-8:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

'''
def showClassifer(dataMat, w, b):
    """
    分类结果可视化
    Parameters:
        dataMat - 数据矩阵
        w - 直线法向量
        b - 直线解决
    Returns:
        无
    """
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        predict = np.dot(w, dataMat[i].T) + b
        if predict > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat[:, 0])
    x2 = min(dataMat[:, 0])
    a1, a2 = w[0][0], w[0][1]
    b = float(b)

    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-8:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()
'''

def smoSimple(dataArr, labelArr, C, toler, maxIter):
    '''
    简化版的smo算法，分割超平面wx + b = 0, 求该超平面的w b
    :param dataArr: 训练样本X
    :param labelArr: 训练样本标签Y
    :param C: 为一个给定常数，通过调节不同参数，获得不同结果
    :param toler: 松弛变量toler>=0
    :param maxIter: 最大迭代次数
    :return:
    '''
    m, _ = np.shape(dataArr)
    b = 0
    alphas = np.zeros((m,1))

    iter = 0
    iter_done = 0
    while (iter<maxIter):
        alphaPairsChanged = 0 #每次循环（1，m）中，alpha对的更新数量
        for i in range(m):
            yi = labelArr[i,0]
            fxi = fx(dataArr, labelArr, i, alphas, b)
            Ei = fxi - yi

            #若违背KKT条件， alpha值进行更新
            bool1 = (alphas[i] < 0) or (alphas[i] > C)
            bool2 = 1 - yi*fxi > toler
            bool3 = abs(alphas[i]*(1 - yi*fxi - toler)) > 1e-8
            if bool1 or bool2 or bool3:
                j = selectJ(i, m)
                yj = labelArr[j,0]
                fxj = fx(dataArr, labelArr, j, alphas, b)
                Ej = fxj - yj

                alphaI_old = alphas[i,0]
                alphaJ_old = alphas[j,0]

                if abs(yi - yj)< 1e-8:
                    L = max(0.0, alphaI_old + alphaJ_old -C)
                    H = min(C, alphaI_old + alphaJ_old)
                else:
                    L = max(0.0, alphaJ_old - alphaI_old)
                    H = min(C, C + alphaJ_old - alphaI_old)

                xi = dataArr[i]
                xj = dataArr[j]

                eta = np.dot(xi, xi.T) + np.dot(xj, xj.T) - 2*np.dot(xi, xj.T)
                if abs(eta)<1e-8: continue #eta == 0

                alphaJ_new_temp = alphaJ_old + yj*(Ei - Ej)/eta
                alphaJ_new = clipAlpha(alphaJ_new_temp, H, L)

                alphaI_new = alphaI_old + yi*yj*(alphaJ_old - alphaJ_new)

                alphas[i] = alphaI_new
                alphas[j] = alphaJ_new

                #若alpha对，更新较小，则不计更新次数alphaPairsChange
                if (abs(alphaI_new - alphaI_old) < 1e-10) and \
                        (abs(alphaJ_new - alphaJ_old) < 1e-10):
                    continue

                b1 = b - Ei - yi*np.dot(xi, xi.T)*(alphaI_new - alphaI_old) - \
                    yj*np.dot(xi, xj.T)*(alphaJ_new - alphaJ_old)
                b2 = b - Ej - yi*np.dot(xi, xj.T)*(alphaI_new - alphaI_old) - \
                    yj*np.dot(xj, xj.T)*(alphaJ_new - alphaJ_old)
                if (alphaI_new>0) and (alphaI_new<C): b = b1
                elif (alphaJ_new>0) and (alphaJ_new<C): b = b2
                else: b = 0.5*(b1 + b2)

                alphaPairsChanged += 1
        iter += 1
        #print('[iter', iter, ': alphaPairsChanged =', alphaPairsChanged, ']')

        if alphaPairsChanged != 0:
            iter_done = 0

        #连续20次迭代，没有任何alpha更新，则退出迭代
        if alphaPairsChanged != 0:
            iter_done = 0
        if alphaPairsChanged <= 0:
            iter_done += 1
        if iter_done >= 20 :
            print('last_iter:', iter)
            break

    if iter >= maxIter: print('last_iter', iter)
    w = get_w(alphas, dataArr, labelArr)
    b = get_b(alphas, dataArr, labelArr)

    return alphas, w, b



if __name__ == '__main__':
    time_start = time.time()
    dataArr,labelArr = loadDataset('SVM_InputData.txt')
    alphas, w, b = smoSimple(dataArr, labelArr, 0.6, 1e-3, 800)
    print('w: ', w)
    print('b: ', b)
    time_end = time.time()
    print('time: ', time_end - time_start, 's')
    #showDataSet(dataArr, labelArr)
    #showClassifer(dataArr, w, b)
    showClassifer(dataArr, labelArr, w, b)

import matplotlib.pyplot as plt
import numpy as np
import time
import random
#import types

def loadDataset(fileName):
    '''
    说明：读入数据

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
    return dataMat, labelMat

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

def showClassifer(dataMat, labelMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]

    b = float(b)
    a1 = float(w[0][0])
    a2 = float(w[0][1])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def clipAlpha(alpha, H, L):
    '''
    Declaration: 修剪alpha
    :param alpha: 更新后的alpha
    :param H: 上限
    :param L: 下限
    :return: 修剪后的alpha
    '''
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha

def gX(dataMat, labelMat, alphas, index):
    '''
    Declaration:
    计算g(x)
    f(x) = g(x) + b
    :param dataMat: 样本X
    :param labelMat: 样本Y
    :param alphas: alpha
    :param index: 索引[index]
    :return: g(x[index])
    '''
    ay = alphas*labelMat
    ayx = np.dot(ay.T, dataMat)
    g_index = np.dot(ayx, (dataMat[index]).T)
    return  g_index

def smoSimple(dataMatIn, classLabels, C, maxIter):
    '''
    Declaration:
    简化版SMO(Sequential Minimal Optimization)

    :param dataMatIn: 数据X
    :param classLabels: 数据Y
    :param C: alpha限定值 - alpha<=C
    :param maxIter: 最大迭代次数
    :return: 超平面参数 w,b, 解为 wx + b = 0
    '''
    dataMat = np.array(dataMatIn)
    labelMat = np.array(classLabels)

    m, _ = np.shape(dataMat)
    alphas = np.zeros((m, 1))

    labelMat = labelMat.reshape(m,1)
    b = 0

    iter_num = 0
    while(iter_num < maxIter):
        for i in range(m):
            #随机确定 j
            j = i
            while( j==i):
                j = int(random.uniform(0,m))

            xi = dataMat[i]; xj = dataMat[j]
            yi = labelMat[i]; yj = labelMat[j]
            alphai_old = alphas[i]; alphaj_old = alphas[j]

            gi = gX(dataMat, labelMat, alphas, i)
            gj = gX(dataMat, labelMat, alphas, j)
            eta = np.dot(xi, xi.T) + np.dot(xj, xj.T) - 2*np.dot(xi, xj.T)

            epsilon = 1e-8
            if abs(yi - yj) < epsilon:
                H = min(C, alphai_old + alphaj_old)
                L = max(0, alphai_old + alphaj_old -C)
            else:
                H = min(C, C + alphaj_old - alphai_old)
                L = max(0, alphaj_old - alphai_old)

            alphaj_new = alphaj_old + yj*((gi - yi) - (gj - yj))/eta
            alphaj_new = clipAlpha(alphaj_new, H, L)

            alphai_new = alphai_old + yi*yj*(alphaj_old - alphaj_new)

            alphas[i] = alphai_new
            alphas[j] = alphaj_new

        iter_num +=1
        #print('end', iter_num)

    w = get_w(alphas, dataMat, labelMat)
    b = get_b(alphas, dataMat, labelMat)

    return alphas,w,b

def get_w(alphas, data, label):
    '''
    Declaration:得到w
    :param alphas: alpha数组
    :param data: 样本X
    :param label: 样本Y
    :return: 超平面方程参数w
    '''
    ay_temp = alphas*label
    w = np.dot(ay_temp.T, data)
    return w

def get_b(alphas, data, label):
    '''
    得到 b
    :param alphas: alpha数组
    :param data:  样本X
    :param label: 样本Y
    :return: 超平面参数 b
    '''
    w = get_w(alphas, data, label)
    index = np.where(abs(alphas) < 1e-8)
    sv_num = len(index[0])
    sum_sv = 0.0
    for i in range(sv_num):
        sv_index = index[0][i]
        yi_sv = label[sv_index]
        wxi_sv = np.dot(w, data[sv_index].T)
        sum_temp = yi_sv - wxi_sv
        sum_sv += sum_temp
    b = sum_sv/sv_num
    return b

if __name__ == '__main__':
    time_start = time.time()
    dataMat,labelMat = loadDataset('SVM_InputData.txt')
    alphas, w,b = smoSimple(dataMat, labelMat, 0.6, 400)
    print('w', w)
    print('b', b)
    time_end = time.time()
    print('time:', time_end - time_start, ' s')
    showClassifer(dataMat, labelMat, w, b)

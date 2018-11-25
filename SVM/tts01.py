import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random
import types

'''
dataMat = np.mat([[1,2],[3,4], [5,6]])
labelMat = np.mat([[1],[-1],[1]])
alphas = np.mat([[1],[-1],[1]])
'''
dataMat = np.array([[1,2],[3,4], [5,6]])
labelMat = np.array([[1],[-1],[1]])
alphas = np.array([[1],[2],[3]])

index = 0
'''
print(dataMat)
print(np.shape(dataMat))
print(labelMat)
print(np.shape(labelMat))
print(alphas)
print(np.shape(alphas))
print(dataMat.shape)
print(type(dataMat))
'''

ay = alphas*labelMat
ayx = np.dot(ay.T, dataMat)
g_index = np.dot(ayx, (dataMat[index]).T)
''''
print(g_index)
print(type(np.shape(alphas)))
print(type(np.zeros((11, 1))))
print(dataMat[2])
print(np.shape(dataMat[2]))
print(np.shape(np.array(dataMat[2])))
print(dataMat[2,:])
print(dataMat[:,0])
print(np.shape(dataMat[:,0]))

m, _ = np.shape(dataMat)
print(m)
print(dataMat.shape)
#print(n)
'''
#print('dataMat:', dataMat)
#a = np.where(dataMat >3)
#print(a)
#print(type(a), np.shape(a))
#print(a[0][0])
print('dataMat',dataMat, dataMat.shape)
#norm_dataMat = np.linalg.norm(dataMat)
#print('normdataMat', norm_dataMat)
#print((91)**0.5)
#print(np.power(dataMat,2))

#ii1,ii2= max(dataMat)
#print(max(dataMat))
#print(ii1,ii2)

i = 1
dataArr = dataMat
m, _ = dataArr.shape
print('m', m)
mr = dataArr.shape
type('mr', mr, type(mr))
print('---------------')
dataArr_minus = dataArr - dataArr[i]
dataArr_m1 = dataArr_minus[:, 0].reshape(m, 1)
dataArr_m2 = dataArr_minus[:, 1].reshape(m, 1)
norm_sqare = np.power(dataArr_m1, 2) + np.power(dataArr_m2, 2)
indexJ = np.where(norm_sqare == np.max(norm_sqare))
j = indexJ[0][0]
#print('dataArr_minus',dataArr_minus, dataArr_minus.shape)
#print('[:,0]', dataArr_minus[:,0], type(dataArr_minus[:,0]), dataArr_minus[:,0].shape)
print('norm_square', norm_sqare, type(norm_sqare),norm_sqare.shape)
print('index', indexJ)
print('j', j,type(j))

def findJ(dataArr, i):
    m, _ = dataArr.shape
    dataArr_minus = dataArr - dataArr[i]
    dataArr_m1 = dataArr_minus[:, 0].reshape(m[0], 1)
    dataArr_m2 = dataArr_minus[:, 1].reshape(m[0], 1)
    norm_sqare = np.power(dataArr_m1, 2) + np.power(dataArr_m2, 2)
    indexJ = np.where(norm_sqare == np.max(norm_sqare, axis = 1))
    j = indexJ[0][0]
    return j



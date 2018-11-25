# -*- coding: utf-8 -*-
###神经网络模型-正则化：L2 Regularization to prevent overfitting
import numpy as np
import matplotlib.pyplot as plt
import time

def load_nn_data():
    '''
    加载数据
    :return: train_X ndarray二维 (n_x, m)； train_Y ndarray二维 (1,m)
    '''
    train_X = np.load('DataSet/train_X.npy')
    train_Y = np.load('DataSet/train_Y.npy')
    test_X = np.load('DataSet/test_X.npy')
    test_Y = np.load('DataSet/test_Y.npy')
    return train_X, train_Y, test_X, test_Y

def plot_data(X,Y):
    '''
    作图数据，观看其样本分布
    :param X: Data
    :param Y: Label
    :return:
    '''
    Y = Y.reshape(X[0,:].shape)

    plt.scatter(X[0,:], X[1,:], c= Y, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Points of Positive and Negative Samples')
    plt.show()


def relu(z):
    a = np.maximum(z, 0)
    return a

def d_relu(z):
    dz = z >= 0
    return dz

def sigmoid(z):
    a = 1.0/(1 + np.exp(-z))
    return a

def d_sigmoid(z):
    a = sigmoid(z)
    dz = a*(1 - a)
    return dz

def initialize_paras(layer_dims):
    '''
    "He" initialization
    :param layer_dims: python list- [n_x, n_1, ... , n_L]
    :return: python dictionary- W1 b1, ... , WL bL
    '''
    np.random.seed(3) #给定随机种子（不指定也可）
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])\
                                   *np.sqrt(2.0/layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def normalize_input(X):
    '''
    归一化输入：输入X转化为均值为0，方差为1的标准输入
    训练集与测试集都用-训练集-的-均值和方差
    :param X: 二维np数组， 维度（n_x，m）
    :return:  归一化后的二维np数组，维度（n_x，m）
    '''
    m = np.shape(X)[1]
    miu = 1.0/m*np.sum(X, axis = 1, keepdims = True)
    x = X - miu
    sigma = np.power(1.0/m*np.sum(np.power(x, 2), axis = 1, keepdims= True), 0.5)
    norm_X = x/sigma
    return norm_X

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    Compute one layer: linear and activation
    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return: A: activation; cache tuple ((A_prev, W, b), Z)
    '''
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'relu':
        A = relu(Z)
    if activation == 'sigmoid':
        A = sigmoid(Z)
    cache = (linear_cache, Z)

    return A, cache

def L_model_forward(X, parameters):
    '''
    Compute forward propagation
    :param X: data set 2d ndarray (n_x, m)
    :param parameters: dictionary {"W1":W1, "b1":b1, ...}
    :return:AL-Y predict: 2d ndarray (1,m);  caches- list of cache
    '''
    L = len(parameters)//2
    caches = []
    A = X
    for l in range(1,L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        activation = 'relu'
        A, cache = linear_activation_forward(A_prev, W, b, activation)
        caches.append(cache)

    A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    activation = 'sigmoid'
    AL, cache = linear_activation_forward(A_prev, W, b, activation)
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1.0/m*np.sum(Y*np.log(AL) + (1 - Y)*np.log(1-AL), axis = 1, keepdims= True)
    cost = np.squeeze(cost)
    return cost

def compute_cost_L2(AL, Y, lambd, parameters):
    cost_entropy = compute_cost(AL, Y)

    m = Y.shape[1]
    L = len(parameters)//2
    cost_L2 = 0
    for l in range(1, L+1):
        cost_L2 += 0.5*lambd/m*np.sum(np.square(parameters["W" + str(l)]))
    cost = cost_entropy + cost_L2
    return cost

def linear_backward(dZ, cache):
    m = dZ.shape[1]
    A_prev, W, _ = cache # _ is b
    dW = 1.0/m*np.dot(dZ, A_prev.T)
    db = 1.0/m*np.sum(dZ, axis = 1, keepdims= True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, caches, activation):
    linear_cache, Z = caches
    if activation == 'relu':
        dZ = dA*d_relu(Z)
    if activation == 'sigmoid':
        dZ = dA*d_sigmoid(Z)
    dA_prev,dW, db  = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    '''
    Compute back propagation
    :param AL:
    :param Y:
    :param caches:
    :return: grads- dictionary {"dW1":dW1, "db1":db1, ...}
    '''
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    linear_cache, ZL = caches[-1]
    dZL = (AL - Y)
    dA_prev, dWL, dbL = linear_backward(dZL, linear_cache)
    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL
    #为何这两段代码所产生的结果不同
    '''
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    dA_prev, grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[-1], "sigmoid")
    '''

    for l in reversed(range(L-1)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[l], activation = 'relu')
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
    gradient descent update
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    '''
    L = len(parameters)//2
    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*grads['db' + str(l+1)]
    return parameters

def update_parameters_L2(parameters, grads, learning_rate, lambd, m):
    L = len(parameters)//2
    for l in range(L):
        parameters['W' + str(l+1)] = (1 - learning_rate*lambd/m)*parameters['W' + str(l+1)] - \
                                     learning_rate*grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*grads['db' + str(l+1)]
    return parameters


def model(X, Y, layer_dims, learning_rate, max_iter_num, print_cost = False):
    X = normalize_input(X)
    parameters = initialize_paras(layer_dims)
    costs = []
    for i in range(max_iter_num):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print('[Iter {0:^6} Cost = {1:8.6}]'.format(i, cost))



    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations (per hundred)')
    plt.ylabel('cost')
    plt.title('Learning rate = {0:^5}'.format(learning_rate))
    plt.show()

    return parameters

def model_L2(X, Y, layer_dims, learning_rate, max_iter_num, lambd, print_cost = False):
    X = normalize_input(X)
    m = X.shape[1]
    parameters = initialize_paras(layer_dims)
    costs = []
    for i in range(max_iter_num):
        AL, caches = L_model_forward(X, parameters)
        #cost = compute_cost(AL, Y)
        cost = compute_cost_L2(AL, Y, lambd, parameters)
        grads = L_model_backward(AL, Y, caches)
        #parameters = update_parameters(parameters, grads, learning_rate)
        parameters = update_parameters_L2(parameters, grads, learning_rate, lambd, m)

        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print('[Iter {0:^6} Cost = {1:8.6}]'.format(i, cost))



    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations (per hundred)')
    plt.ylabel('cost')
    plt.title('Learning rate = {0:^5}'.format(learning_rate))
    plt.show()

    return parameters

def predict(X, parameters):
    '''
    结构：X->LINEAR->RELU->LINEAR->RELU->...->LINEAR->SIGMOID->Y
    :param X:
    :param parameters:
    :return:
    '''
    X = normalize_input(X)
    L = len(parameters)//2
    A = X
    for l in range(1,L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z)

    A_prev = A
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = np.dot(WL, A_prev) + bL
    Y_predict_temp = sigmoid(ZL)
    Y_predict = Y_predict_temp > 0.5
    return Y_predict

def plot_decision_boundary(X, parameters):
    #x_min, x_max = X[0, :].min() - .5, X[0, :].max() + .5
    #y_min, y_max = X[1, :].min() - .5, X[1, :].max() + .5
    x_min, x_max = X[0, :].min() - 0, X[0, :].max() + 0
    y_min, y_max = X[1, :].min() - 0, X[1, :].max() + 0

    h = 0.01
    X_meshed = np.array(np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)))

    X_mesh = X_meshed.reshape(2,-1)
    Y_bound_pred = predict(X_mesh, parameters)
    Y_bound_pred = Y_bound_pred.reshape(X_meshed[0].shape)

    plt.title('Model of Neural Networks')
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    plt.contourf(X_meshed[0], X_meshed[1], Y_bound_pred, cmap=plt.cm.Spectral)

    Y_predict = predict(X, parameters)
    Y_predict = Y_predict.reshape(X[0,:].shape)
    plt.scatter(X[0, :], X[1, :], c=Y_predict, cmap=plt.cm.Spectral)

    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    train_X, train_Y, test_X, test_Y = load_nn_data()
    #print(type(train_X), train_X.shape)
    #print(type(train_Y), train_Y.shape)
    #print(type(test_X), test_X.shape)
    #print(type(test_Y), test_Y.shape)

    #plot_data(train_X, train_Y)
    #plot_data(test_X, test_Y)

    X = train_X
    Y = train_Y
    learning_rate = 0.01
    layer_dims = [X.shape[0],8,4,3,1]
    #layer_dims = [2,1] #L = 1 为逻辑回归
    max_iter_num = 30000#50000
    lambd = 0.5
    parameters = model_L2(X, Y, layer_dims, learning_rate, max_iter_num, lambd, print_cost = True)

    Y_predict_train = predict(train_X, parameters)
    error_rate_train = np.mean(np.abs(Y_predict_train - train_Y))
    print('Train error rate is {0:.2%}'.format(error_rate_train))

    Y_predict_test = predict(test_X, parameters)
    error_rate_test = np.mean(np.abs(Y_predict_test - test_Y))
    print('Test error rate is {0:.2%}'.format(error_rate_test))

    plot_decision_boundary(train_X, parameters)
    #plot_decision_boundary(test_X, parameters)

    end_time = time.time()
    #print('Total time is {0:.2f} s'.format(end_time - start_time))

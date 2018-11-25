# -*- coding: utf-8 -*-
###神经网络模型
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
    '''
    #linear_cache, ZL = caches[-1]
    dZL = (AL - Y)
    dA_prev, dWL, dbL = linear_backward(dZL, linear_cache)
    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL
    #为何这两段代码所产生的结果不同
    '''
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    dA_prev, grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[-1], "sigmoid")

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
    #print(train_X)
    #x = train_X.shape
    #print(type(x), len(x))
    #print(type(train_Y), train_Y.shape)
    #print(type(test_X), test_X.shape)
    #print(type(test_Y), test_Y.shape)

    #plot_data(train_X, train_Y)
    #plot_data(test_X, test_Y)



    '''
    x = np.array([[-1,-2,3], [4,5,6]])
    x = np.array([-1,-2,3])
    print(x.shape)

    sa = d_sigmoid(x)
   
    print(type(sa), np.shape(sa), sa)
    '''
    '''
    layer_dims = [2,4,1]
    parameters = initialize_paras(layer_dims)
    print(parameters['W1'], type(parameters['W1']), np.shape(parameters['W1']))
    print(parameters['b1'], type(parameters['b1']), np.shape(parameters['b1']))
    print(parameters['W2'])
    print(parameters['b2'])
    print('------------------')
    pp = len(parameters)//2
    print(type(parameters), type(pp), np.shape(pp))
    '''
    #A_prev = np.array([[-0.41675785 -0.05626683] [-2.1361961 1.64027081] [-1.79343559 -0.84174737]])
    '''
    A_prev = np.array([[-0.41675785, -0.05626683], [-2.1361961, 1.64027081], [-1.79343559, -0.84174737]])
    W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
    b = np.array([[-0.90900761]])
    print(A_prev, type(A_prev), A_prev.shape)
    print(W, type(W), W.shape)
    print(b, type(b), b.shape)
    print('------------------')
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    print(np.shape(linear_activation_cache))
    print(linear_activation_cache)
    '''

    '''
    X = np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],\
                  [-2.48678065,  0.91325152,  1.12706373, -1.51409323],\
         [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],\
                  [-0.33588161,  1.23773784,  0.11112817,  0.12915125],\
                  [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
    parameters = {'b2': np.array([[ 1.50278553],
       [-0.59545972],
       [ 0.52834106]]), 'W1': np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
       [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
       [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
       [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]), 'W2': np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
       [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
       [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]), 'b1': np.array([[ 1.38503523],
       [-0.51962709],
       [-0.78015214],
       [ 0.95560959]]), 'b3': np.array([[-0.16236698]]), 'W3': np.array([[ 0.9398248 ,  0.42628539, -0.75815703]])}
    print(X,type(X),X.shape)
    print(len(parameters))
    print(parameters['W1'].shape,type(parameters['W1']))
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))
    '''
    '''
    Y = np.array([[1, 1, 1]])
    AL = np.array([[ 0.8,  0.9,  0.4]])
    cost = compute_cost(AL, Y)
    print("cost = " + str(compute_cost(AL, Y)))
    print(type(cost))
    '''
    '''

    dAL = np.array([[-0.41675785, -0.05626683]])
    caches = ((np.array([[-2.1361961 ,  1.64027081],
       [-1.79343559, -0.84174737],
       [ 0.50288142, -1.24528809]]), np.array([[-1.05795222, -0.90900761,  0.55145404]]), np.array([[ 2.29220801]])), np.array([[ 0.04153939, -1.11792545]]))
    dA_prev, dW, db = linear_activation_backward(dAL, caches, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")

    dA_prev, dW, db = linear_activation_backward(dAL, caches, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    '''

    '''
    AL = np.array([[ 1.78862847,  0.43650985]])
    Y = np.array([[1, 0]])
    caches = (((np.array([[ 0.09649747, -1.8634927 ],
       [-0.2773882 , -0.35475898],
       [-0.08274148, -0.62700068],
       [-0.04381817, -0.47721803]]), np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
       [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
       [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]), np.array([[ 1.48614836],
       [ 0.23671627],
       [-1.02378514]])), np.array([[-0.7129932 ,  0.62524497],
       [-0.16051336, -0.76883635],
       [-0.23003072,  0.74505627]])), ((np.array([[ 1.97611078, -1.24412333],
       [-0.62641691, -0.80376609],
       [-2.41908317, -0.92379202]]), np.array([[-1.02387576,  1.12397796, -0.13191423]]), np.array([[-1.62328545]])), np.array([[ 0.64667545, -0.35627076]])))
    print(AL,type(AL), AL.shape)
    print(Y, type(Y), Y.shape)
    print(len(caches))
    print('--------------------------------------------------------')
    grads = L_model_backward(AL, Y, caches)
    print(grads)
    '''
    '''
    parameters = {'W2': np.array([[-0.5961597 , -0.0191305 ,  1.17500122]]), 'b2': np.array([[-0.74787095]]), 'b1': np.array([[ 0.04153939],
       [-1.11792545],
       [ 0.53905832]]), 'W1': np.array([[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
       [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
       [-1.05795222, -0.90900761,  0.55145404,  2.29220801]])}
    grads = {'db1': np.array([[ 0.88131804],
       [ 1.70957306],
       [ 0.05003364]]), 'dW1': np.array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
       [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
       [-0.04381817, -0.47721803, -1.31386475,  0.88462238]]), 'dW2': np.array([[-0.40467741, -0.54535995, -1.54647732]]), 'db2': np.array([[ 0.98236743]])}
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
    '''


    X = train_X
    Y = train_Y
    learning_rate = 0.01
    layer_dims = [X.shape[0],8,4,3,1]
    #layer_dims = [2,1] #L = 1 为逻辑回归
    max_iter_num = 30000#50000
    parameters = model(X, Y, layer_dims, learning_rate, max_iter_num, print_cost = True)

    Y_predict_train = predict(train_X, parameters)
    error_rate_train = np.mean(np.abs(Y_predict_train - train_Y))
    print('Train error rate is {0:.2%}'.format(error_rate_train))

    Y_predict_test = predict(test_X, parameters)
    error_rate_test = np.mean(np.abs(Y_predict_test - test_Y))
    print('Test error rate is {0:.2%}'.format(error_rate_test))

    #plot_decision_boundary(train_X, parameters)
    #plot_decision_boundary(test_X, parameters)

    end_time = time.time()
    #print('Total time is {0:.2f} s'.format(end_time - start_time))







    '''
    b = np.array([[[1,2,3], [4,5,6],[7,8,9]], [[2,3,4], [5,6,7],[8,9,10]]])
    print(b, b.shape, type(b))
    print(b[1,1,2])

    sum0 = np.sum(b, axis = 0, keepdims=False)
    sum1 = np.sum(b, axis = 1, keepdims=True)
    sum2 = np.sum(b, axis = 2, keepdims=True)
    #sum3 = np.sum(b, axis = 3, keepdims=True)
    print('------------------')
    print(sum0, sum0.shape)
    print('------------------')
    print(sum1)
    print('------------------')
    print(sum2)
    print('------------------')
    sum_1 = np.sum(b, axis = None , keepdims=True)
    print(sum_1, type(sum_1), sum_1.shape)
    print('------------------')
    sum_2 = np.sum(b, axis = None , keepdims=False)
    print(sum_2, type(sum_2), sum_2.shape)
    '''
    '''
    b = np.array([[1,2,3], [4,5,6]])
    norm_b = normalize_input(b)
    print(norm_b, type(norm_b), norm_b.shape)
    '''



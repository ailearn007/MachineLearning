# -*- coding: utf-8 -*-
import numpy as np

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

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z/e_z.sum(axis= 0)


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
    if activation == 'softmax':
        A = softmax(Z)
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
    #softmax
    m = Y.shape[1]
    cost = -1.0/m*np.sum(Y*np.log(AL))
    cost = np.squeeze(cost)
    return cost

# if __name__ == '__main__':
#     AL = np.abs(np.random.randn(2,4))
#     Y = np.abs(np.random.randn(2,4))
#     cost = compute_cost(AL,Y)
#     print(cost.shape, type(cost), cost)

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

    #也可用下面两行代码
    #dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    #dA_prev, grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, caches[-1], "sigmoid")

    for l in reversed(range(L-1)):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[l], activation = 'relu')
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
    return grads

def initialize_ADAM(parameters):
    Vd = {}
    Sd = {}
    for item in parameters.items():
        Vd['V_d' + item[0]] = np.zeros(item[1].shape)
        Sd['S_d' + item[0]] = np.zeros(item[1].shape)
    return Vd, Sd

def update_parameters_ADAM(parameters, grads, learning_rate, iter_num, Vd, Sd, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

    L = len(parameters)//2
    for l in range(L):
        Vd['V_dW' + str(l+1)] = beta_1*Vd['V_dW' + str(l+1)] + (1-beta_1)*grads['dW' + str(l+1)]
        Vd['V_db' + str(l+1)] = beta_1*Vd['V_db' + str(l+1)] + (1-beta_1)*grads['db' + str(l+1)]
        Sd['S_dW' + str(l+1)] = beta_2*Sd['S_dW' + str(l+1)] + (1-beta_2)*np.square(grads['dW' + str(l+1)])
        Sd['S_db' + str(l+1)] = beta_2*Sd['S_db' + str(l+1)] + (1-beta_2)*np.square(grads['db' + str(l+1)])

        V_dW_corrected = Vd['V_dW' + str(l+1)]/(1 - beta_1**iter_num)
        V_db_corrected = Vd['V_db' + str(l+1)]/(1 - beta_1**iter_num)
        S_dW_corrected = Sd['S_dW' + str(l+1)]/(1 - beta_2**iter_num)
        S_db_corrected = Sd['S_db' + str(l+1)]/(1 - beta_2**iter_num)

        parameters['W' + str(l+1)] -= learning_rate*V_dW_corrected/(np.sqrt(S_dW_corrected) + epsilon)
        parameters['b' + str(l+1)] -= learning_rate*V_db_corrected/(np.sqrt(S_db_corrected) + epsilon)
    return parameters, Vd, Sd

def compute_mini_batches(shuffled_X, shuffled_Y, batch_size = 128):
    #shuffled_X.shape = (m, 64, 64, 3)
    #shuffled_Y.shape = (6, m)
    m = shuffled_X.shape[0]
    batches = []
    batch_num = m//batch_size
    for k in range(batch_num):
        mini_batch_X = shuffled_X[k*batch_size:(k+1)*batch_size, :,:,:]
        mini_batch_Y = shuffled_Y[:, k*batch_size:(k+1)*batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)

    if m%batch_size != 0:
        mini_batch_X = shuffled_X[batch_num*batch_size:, :,:,:]
        mini_batch_Y = shuffled_Y[:, batch_num*batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)
    return batches

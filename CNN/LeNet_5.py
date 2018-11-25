#coding = utf-8
import time
import datetime
import math
import numpy as np
import h5py
from FC_nets import *
import matplotlib.pyplot as plt
# 模拟LeNet-5 的网络结构
# X -> CONV1 -> MAX-POOL -> CONV2 -> MAX-POOL -> FC3 -> FC4 -> FC5(Softmax) -> Y

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def some_data(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, \
              train_num = 360, test_num = 100):
    # 将X归一化，将Y转换为one-hot vector
    # 从训练集和测试集中选取部分数据，进行模型训练和测试
    X_train = train_set_x_orig/255
    X_test = test_set_x_orig/255
    Y_train = convert_to_one_hot(train_set_y_orig, 6).T
    Y_test = convert_to_one_hot(test_set_y_orig, 6).T

    np.random.seed(0)
    state_1 = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state_1)
    np.random.shuffle(Y_train)

    state_2 = np.random.get_state()
    np.random.shuffle(X_test)
    np.random.set_state(state_2)
    np.random.shuffle(Y_test)

    X_train_some = X_train[0:train_num]
    Y_train_some = Y_train[0:train_num]
    X_test_some = X_test[0:test_num]
    Y_test_some = Y_test[0:test_num]
    return X_train_some, Y_train_some, X_test_some, Y_test_some

# def relu(z):
#     a = np.maximum(z, 0)
#     return a

def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0), (pad,pad), (pad,pad), (0,0)), \
                   'constant', constant_values= 0)
    return X_pad

def conv_sigle_pad(a_slice_prev, W, b):
    s = a_slice_prev*W
    Z = np.sum(s)
    Z = float(Z + b)
    return Z

def conv_linear_forward(A_prev, W, b, layer_para):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    s = layer_para['stride']
    p = layer_para['pad']

    n_H = math.floor((n_H_prev + 2*p - f)/s + 1)
    n_W = math.floor((n_W_prev + 2*p - f)/s + 1)
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, p)

    for i in range(m):
        for h in range(n_H):

            slice_start_h = h*s
            slice_end_h = slice_start_h + f

            for w in range(n_W):

                slice_start_w = w*s
                slice_end_w = slice_start_w + f

                for c in range(n_C):
                    a_slice_prev = A_prev_pad[i, \
                                   slice_start_h:slice_end_h, slice_start_w:slice_end_w, :]
                    Z[i, h, w, c] = conv_sigle_pad(a_slice_prev, W[:,:,:,c], b[0,0,0,c])

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, layer_para)
    return Z,cache

def conv_avtivate_forward(Z):
    A = relu(Z)
    return A

def conv_step_forward(A_prev, W, b, layer_para):
    Z, cache = conv_linear_forward(A_prev, W, b, layer_para)
    A = conv_avtivate_forward(Z)
    cache_conv = (Z, cache)
    return A, cache_conv

def conv_linear_backward(dZ, cache):
    (A_prev, W, b, layer_para) = cache
    (m, n_H, n_W, n_C) = dZ.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    s = layer_para['stride']
    p = layer_para['pad']

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    A_prev_pad = zero_pad(A_prev, p)
    dA_prev_pad = zero_pad(dA_prev, p)

    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    slice_start_h = h*s
                    slice_end_h = slice_start_h + f
                    slice_start_w = w*s
                    slice_end_w = slice_start_w + f
                    a_slice = a_prev_pad[slice_start_h:slice_end_h, slice_start_w:slice_end_w,:]

                    da_prev_pad[slice_start_h:slice_end_h, slice_start_w:slice_end_w, :] += \
                    W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice*dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]

        #dA_prev[i,:,:,:] = da_prev_pad[p:-p, p:-p, :]
        if p != 0:
            dA_prev[i,:,:,:] = da_prev_pad[p:-p, p:-p, :]
        if p == 0:
            dA_prev[i,:,:,:] = da_prev_pad[:, :, :]

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

# def d_relu(z):
#     dz = z >= 0
#     return dz

def conv_step_backward(dA, cache_conv):
    (Z, cache) = cache_conv
    dZ = d_relu(Z)*dA
    dA_prev, dW, db = conv_linear_backward(dZ, cache)
    return dA_prev, dW, db



def pool_forward(A_prev, pool_layer_para):
    f = pool_layer_para['filter_size']
    s = pool_layer_para['stride']
    mode = pool_layer_para['mode']
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    n_H = math.floor((n_H_prev  - f)/s + 1)
    n_W = math.floor((n_W_prev  - f)/s + 1)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            slice_start_h = h*s
            slice_end_h = slice_start_h + f
            for w in range(n_W):
                slice_start_w = w*s
                slice_end_w = slice_start_w + f
                for c in range(n_C):
                    if mode == 'max':
                        A[i,h,w,c] = np.max(A_prev[i, slice_start_h:slice_end_h, \
                                            slice_start_w:slice_end_w, c])
                    if mode == 'average':
                        A[i,h,w,c] = np.mean(A_prev[i, slice_start_h:slice_end_h, \
                                            slice_start_w:slice_end_w, c])
    assert (A.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, pool_layer_para)
    return A, cache

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz/(n_H*n_W)
    a = np.ones(shape)*average
    return a

def pool_backward(dA, cache):
    (A_prev, pool_layer_para) = cache
    s = pool_layer_para["stride"]
    f = pool_layer_para["filter_size"]
    mode = pool_layer_para['mode']

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    slice_start_h = h*s
                    slice_end_h = slice_start_h + f
                    slice_start_w = w*s
                    slice_end_w = slice_start_w + f

                    if mode == 'max':
                        a_prev_slice = a_prev[slice_start_h:slice_end_h, \
                                       slice_start_w:slice_end_w, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, slice_start_h:slice_end_h, slice_start_w:slice_end_w, c] += \
                            dA[i,h,w,c]*mask

                    if mode == 'average':
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, slice_start_h:slice_end_h, slice_start_w:slice_end_w, c] += \
                            distribute_value(da, shape)

    return dA_prev

def flatten(A):
    m = A.shape[0]
    flat_A = A.reshape(-1, m)
    return flat_A, A.shape

def flatten_back(dA, A_shape):
    flat_back_dA = dA.reshape(A_shape)
    return flat_back_dA

def define_hparas():
    conv_para_1 = {'stride':2, 'pad':0}
    pool_para_1 = {'stride':2, 'filter_size':2, 'mode':'max'}
    conv_para_2 = {'stride':1, 'pad':1}
    pool_para_2 = {'stride':2, 'filter_size':2, 'mode':'max'}
    hparas = (conv_para_1, pool_para_1, conv_para_2, pool_para_2)
    return hparas

def init_CNN_paras():
    parameters = {}
    parameters['W1'] = np.random.randn(5,5,3,6)*0.01
    parameters['W2'] = np.random.randn(5,5,6,16)*0.01
    parameters['W3'] = np.random.randn(120,576)*0.01
    parameters['W4'] = np.random.randn(84,120)*0.01
    parameters['W5'] = np.random.randn(6,84)*0.01
    parameters['b1'] = np.zeros((1,1,1,6))
    parameters['b2'] = np.zeros((1,1,1,16))
    parameters['b3'] = np.zeros((120,1))
    parameters['b4'] = np.zeros((84,1))
    parameters['b5'] = np.zeros((6,1))
    return parameters

def write_parameters(parameters):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = \
        parameters['W1'], \
        parameters['b1'], \
        parameters['W2'], \
        parameters['b2'], \
        parameters['W3'], \
        parameters['b3'], \
        parameters['W4'], \
        parameters['b4'], \
        parameters['W5'], \
        parameters['b5']

    # W1 = parameters['W1']
    # b1 = parameters['b1']
    # W2 = parameters['W2']
    # b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    # W4 = parameters['W4']
    # b4 = parameters['b4']
    # W5 = parameters['W5']
    # b5 = parameters['b5']
    return  W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

def forward_model(X, parameters):
    A0 = X
    (conv_para_1, pool_para_1, conv_para_2, pool_para_2) = define_hparas()
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = write_parameters(parameters)

    A1, cache_conv1 = conv_step_forward(A0, W1, b1, conv_para_1)
    A1_pool,cache_pool1 = pool_forward(A1, pool_para_1)

    A2, cache_conv2 = conv_step_forward(A1_pool, W2, b2, conv_para_2)
    A2_pool,cache_pool2 = pool_forward(A2, pool_para_2)

    A2P_flat, cache_flat = flatten(A2_pool)

    A3, cache_fc3 = linear_activation_forward(A2P_flat, W3, b3, activation = 'relu')
    A4, cache_fc4 = linear_activation_forward(A3, W4, b4, activation = 'relu')
    A5, cache_fc5 = linear_activation_forward(A4, W5, b5, activation = 'softmax')

    caches = (cache_conv1, cache_pool1, cache_conv2, cache_pool2, cache_flat, \
              cache_fc3, cache_fc4, cache_fc5)
    return A5, caches

def backward_model(A5, Y, caches):
    #Y.shape = (6, m)
    grads = {}
    (cache_conv1, cache_pool1, cache_conv2, cache_pool2, cache_flat, \
     cache_fc3, cache_fc4, cache_fc5) = caches

    dZ5 = A5 - Y
    linear5, _ = cache_fc5
    dA4, grads['dW5'], grads['db5'] = linear_backward(dZ5, linear5)

    dA3, grads['dW4'], grads['db4'] = \
        linear_activation_backward(dA4, cache_fc4, activation = 'relu')

    dA2_fc, grads['dW3'], grads['db3'] = \
        linear_activation_backward(dA3, cache_fc3, activation = 'relu')

    dA2_flatback = flatten_back(dA2_fc, cache_flat)
    dA2 = pool_backward(dA2_flatback, cache_pool2)

    dA1_pool, grads['dW2'], grads['db2'] = conv_step_backward(dA2, cache_conv2)
    dA1 = pool_backward(dA1_pool, cache_pool1)

    dA0, grads['dW1'], grads['db1'] = conv_step_backward(dA1, cache_conv1)
    return grads


def CNN_model(X, Y, learning_rate = 1e-2, max_epoch = 1e5, batch_size = 128, print_cost = False):
    Y = Y.T
    parameters = init_CNN_paras()
    Vd, Sd = initialize_ADAM(parameters)
    costs = []

    batches = compute_mini_batches(X, Y, batch_size)
    for epoch_num in range(1, 1+max_epoch):
        for mini_batch in batches:
            mini_X, mini_Y = mini_batch
            A5, caches = forward_model(mini_X, parameters)
            cost = compute_cost(A5, mini_Y)
            grads = backward_model(A5, mini_Y, caches)
            parameters, Vd, Sd = update_parameters_ADAM(parameters, grads, \
                                                        learning_rate, epoch_num, Vd, Sd, \
                                                        beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        if epoch_num%10 == 0:
            costs.append(cost)
            if print_cost:
                print('[ Epoch {0:>4},  Cost = {1:<8.6} ]'.format(epoch_num, cost))

    # plt.plot(costs)
    # plt.xlabel('epochs (per hundred)')
    # plt.ylabel('costs')
    # plt.title('CNN(learning rate = {0:^5})'.format(learning_rate))
    # plt.show()
    return parameters, costs

def CNN_predict(parameters, X, Y):
    Y = Y.T
    m = Y.shape[1]
    Y_hat, _ = forward_model(X, parameters)
    Y_pred = np.float64(Y_hat == np.max(Y_hat, axis= 0))
    precision = 1.0/m*np.sum(Y*Y_pred)
    print('Precision = {0:>6.2%}'.format(precision))

    Time = datetime.datetime.now().strftime('[%Y-%m-%d]-[%H-%M-%S]')
    Y_pred_name = 'Y_pred' + Time + str(time.time())[-3:] + '.npy'
    np.save(Y_pred_name, Y_pred)
    Y_predict = np.load(Y_pred_name)
    return Y_predict

def plot_costs(costs):
    plt.plot(costs)
    plt.xlabel('epochs (per ten)')
    plt.ylabel('costs')
    plt.title('CNN')
    plt.show()
    return None

def save_and_load(parameters, costs):
    Time = datetime.datetime.now().strftime('[%Y-%m-%d]-[%H-%M-%S]')
    para_name = 'parameters--' + Time + '.npy'
    costs_name = 'costs--' + Time + '.npy'
    np.save(para_name, parameters)
    np.save(costs_name, costs)
    parameters_load = np.load(para_name).item()
    costs_load = list(np.load(costs_name))
    return parameters_load, costs_load

def show_results(X, Y, Y_pred, index = 0):
    Y = Y.T
    print('REAL: ', Y[:, index])
    print('PRED: ', Y_pred[:, index])
    plt.imshow(X[index])
    plt.show()
    return None



if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train, Y_train, X_test, Y_test = \
        some_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig,
                  train_num= 360, test_num= 100)


    start_time = time.time()
    parameters, costs = CNN_model(X_train, Y_train, \
                           learning_rate = 0.009, max_epoch = 80, \
                           batch_size = 8, print_cost = True)
    end_time = time.time()
    print('Training time is {0:<8.1f} s'.format(end_time - start_time))

    parameters, costs = save_and_load(parameters, costs)
    plot_costs(costs)

    Y_predict_train = CNN_predict(parameters, X_train, Y_train)
    show_results(X_train, Y_train, Y_predict_train, index = 0)

    Y_predict_test = CNN_predict(parameters, X_test, Y_test)
    show_results(X_test, Y_test, Y_predict_test, index = 0)








    # index = 80
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(Y_train[index, :])
    # plt.imshow(X_train[index])
    #plt.show()

    # parameters = init_CNN_paras()
    # X = X_train[0:2,:,:,:]
    # print(X.shape)
    # A5, caches = forward_model(X, parameters)
    #print(A5, A5.shape, type(A5))

    # Y = Y_train[0:2,:].T
    # grads = backward_model(A5, Y, caches)
    # print(grads['dW1'].shape)
    # print(grads['db1'])

    # pad = 0
    # X_pad = zero_pad(X, pad)
    #
    # print(X_pad.shape)




    #
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(Y_test[index, :])
    # plt.imshow(X_test[index])
    # plt.show()

    # np.random.seed(1)
    # A_prev = np.random.randn(10,4,4,3)
    # print('A_prev', A_prev[0,0,0,0])
    # W = np.random.randn(2,2,3,8)
    # b = np.random.randn(1,1,1,8)
    # layer_para  = {"pad" : 2,"stride": 2}
    # #
    # Z, cache_conv = conv_linear_forward(A_prev, W, b, layer_para)
    # print("Z's mean =", np.mean(Z))
    # print("Z[3,2,1] =", Z[3,2,1])
    # print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    # np.random.seed(1)
    #dA, dW, db = conv_backward(Z, cache_conv)
    # dA, dW, db = conv_linear_backward(Z, cache_conv)
    # print("dA_mean =", np.mean(dA))
    # print("dW_mean =", np.mean(dW))
    # print("db_mean =", np.mean(db))

    # np.random.seed(1)
    # A_prev = np.random.randn(2, 4, 4, 3)
    # hparameters = {"stride" : 2, "filter_size": 3}
    #
    # A, cache = pool_forward(A_prev, hparameters)
    # print("mode = max")
    # print("A =", A)
    # print()
    # A, cache = pool_forward(A_prev, hparameters, mode = "average")
    # print("mode = average")
    # print("A =", A)

    # np.random.seed(1)
    # A_prev = np.random.randn(5, 5, 3, 2)
    # pool_layer_para = {"stride" : 1, "filter_size": 2}
    # A, cache = pool_forward(A_prev, pool_layer_para)
    # dA = np.random.randn(5, 4, 2, 2)
    #
    # dA_prev = pool_backward(dA, cache, mode = "max")
    # print("mode = max")
    # print('mean of dA = ', np.mean(dA))
    # print('dA_prev[1,1] = ', dA_prev[1,1])
    # print()
    # dA_prev = pool_backward(dA, cache, mode = "average")
    # print("mode = average")
    # print('mean of dA = ', np.mean(dA))
    # print('dA_prev[1,1] = ', dA_prev[1,1])

    # np.random.seed(1)
    # B = np.random.randn(2,3,4,5)
    # A = np.random.randn(2,3,4,5)
    # print(B == A)
    # flat_A, A_shape = flatten(A)
    # dA = A
    # A_back = flatten_back(flat_A, A_shape)
    # print(A.shape)
    # print(A_back.shape)
    # print(flat_A.shape)
    # print(A_back == A)














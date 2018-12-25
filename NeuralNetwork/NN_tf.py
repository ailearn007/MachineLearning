#coding:utf-8
import numpy as np
import tensorflow as tf


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

def add_layer(X_input, in_size, out_size, keep_prob=1.0, activate = tf.nn.relu):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Z = tf.matmul(X_input, W) + b
    Z = tf.nn.dropout(Z, keep_prob)
    outputs = activate(Z)
    return outputs

if __name__ =='__main__':
    train_X, train_Y, test_X, test_Y = load_nn_data()

    xs = tf.placeholder(tf.float32, [None, 2])
    ys = tf.placeholder(tf.float32, [None, 1])

    L1 = add_layer(xs, 2, 20)
    L2 = add_layer(L1, 20, 8)
    L3 = add_layer(L2, 8, 3)
    outs = add_layer(L3, 3, 1, 1.0, activate= tf.nn.sigmoid)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(outs) + (1-ys)*tf.log(1-outs),
                                                  reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(5000):
        sess.run(train_step, feed_dict={xs: train_X.T, ys: train_Y.T})

        if i%20 == 0:
            print('----------------')
            print('Train: ', sess.run(cross_entropy, feed_dict={xs: train_X.T, ys: train_Y.T}))
            print('Test:  ', sess.run(cross_entropy, feed_dict={xs: test_X.T, ys: test_Y.T}))


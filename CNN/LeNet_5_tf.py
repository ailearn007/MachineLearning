#coding = utf-8
import tensorflow as tf
import numpy as np
import h5py
from keras.utils import np_utils

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

def normlize_dataset():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes =\
    load_dataset()
    X_train = train_set_x_orig/255
    Y_train = np.squeeze(np_utils.to_categorical(train_set_y_orig.T, num_classes= 6))

    X_test = test_set_x_orig/255
    Y_test = np.squeeze(np_utils.to_categorical(test_set_y_orig.T, num_classes= 6))
    return X_train, Y_train, X_test, Y_test

def add_conv2d(Input, W_shape, b_shape, strides, padding, name, activate = tf.nn.relu):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), dtype=tf.float32, name= name + '_filter')
        b = tf.Variable(tf.constant(0.1, shape= b_shape), dtype=tf.float32, name= name + '_bias')
        conv_out = tf.nn.conv2d(Input, W, strides, padding, name= name + '_conv2d')
        Output = activate((conv_out + b), name)
    return Output

def add_max_pool(Input, ksize, strides, padding, name):
    with tf.name_scope(name):
        pool_out = tf.nn.max_pool(Input, ksize, strides, padding, data_format= 'NHWC', name= name)
    return pool_out

def add_fc(Input, W_shape, b_shape, name, activate = tf.nn.relu):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), dtype=tf.float32, name= name + '_Weight')
        b = tf.Variable(tf.constant(0.1, shape= b_shape), dtype=tf.float32, name= name + '_bia')
        if activate == tf.nn.softmax:
            Output = tf.nn.softmax((tf.matmul(Input, W) + b), axis = 1,name= name)
        else:
            Output = activate((tf.matmul(Input, W) + b), name)
    return Output

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = normlize_dataset()
    print(X_train.shape, Y_train.shape)
    print(Y_train[0])

    X_input = tf.placeholder(dtype= tf.float32, shape= [None, 64, 64, 3], name= 'X_in')
    Y_label = tf.placeholder(dtype= tf.float32, shape= [None, 6], name= 'Y_out')

    CONV1 = add_conv2d(X_input, W_shape = [5,5,3,6], b_shape = [1,1,1,6], strides = [1,2,2,1],
                        padding = 'VALID', name = 'CONV_1', activate = tf.nn.relu)

    MAX1 = add_max_pool(CONV1, ksize = (1,2,2,1), strides = [1,2,2,1],
                         padding = 'VALID',name = 'MAX_1')

    with tf.name_scope('Pad'):
        CONV2Pad = tf.pad(MAX1, paddings= tf.constant([[0,0], [1,1], [1,1], [0,0]]), name='CONV2_pad')
    CONV2 = add_conv2d(CONV2Pad, [5,5,6,16], [1,1,1,16], [1,1,1,1], 'VALID', 'CONV_2')
    MAX2 = add_max_pool(CONV2, [1,2,2,1], [1,2,2,1], 'VALID', 'MAX_2')

    with tf.name_scope('Flat'):
        Flatten = tf.reshape(MAX2, [-1, 576])

    FC3 = add_fc(Flatten, [576, 120], [1, 120], 'FC_3')
    FC4 = add_fc(FC3, [120, 84], [1, 84], 'FC_4')
    FC5 = add_fc(FC4, [84, 6], [1, 6], 'FC_5', activate= tf.nn.softmax)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_label*tf.log(FC5), reduction_indices=[1]))

    with tf.name_scope('Train'):
        train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)

    tf.summary.histogram('loss', cross_entropy)
    tf.summary.scalar('loss', cross_entropy)

    merged = tf.summary.merge_all()


    saver = tf.train.Saver()
    # with tf.name_scope('init'):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter('logs/', sess.graph)
        saver.save(sess, 'LeNet_5_tf/Le5', global_step=505)

        for i in range(500):
            sess.run(train_step, feed_dict={X_input:X_train, Y_label:Y_train})

            if i%20 == 0:
                print('-------------------')
                print('Train: ',sess.run(cross_entropy, feed_dict={X_input:X_train, Y_label:Y_train}))
                print('Test: ',sess.run(cross_entropy, feed_dict={X_input:X_test, Y_label:Y_test}))
                result = sess.run(merged, feed_dict={X_input:X_train, Y_label:Y_train})
                writer.add_summary(result, i)



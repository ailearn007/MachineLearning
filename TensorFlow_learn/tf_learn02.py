#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, keep_prob, activate = None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])+ 0.1)
    Z = tf.matmul(inputs, W) + b

    Z = tf.nn.dropout(Z, keep_prob)

    if activate is None:
        outputs = Z
    else:
        outputs = activate(Z)
    return outputs

def compute_acc(v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict= {xs: v_xs, keep_prob:1.0})
    y_real = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(v_ys, 1))
    acc = tf.reduce_mean(tf.cast(y_real, tf.float32))
    result = sess.run(acc, feed_dict={xs: v_xs, ys: v_ys, keep_prob:1.0})
    return result

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot= True)
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('Layer__1'):
        L1 = add_layer(xs, 784, 200, keep_prob, activate=tf.nn.tanh)
    with tf.name_scope('Layer__2'):
        L2 = add_layer(L1, 200, 100, keep_prob, activate=tf.nn.tanh)
    with tf.name_scope('OUT'):
        prediction = add_layer(L2, 100, 10, keep_prob, activate= tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs2/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs2/test', sess.graph)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict= {xs:batch_xs, ys:batch_ys, keep_prob:0.5})
        if i%50 ==0:
            print(compute_acc(mnist.test.images, mnist.test.labels))

            train_result = sess.run(merged, feed_dict={xs: mnist.train.images, ys:mnist.train.labels, keep_prob:1.0})
            test_result = sess.run(merged, feed_dict={xs: mnist.test.images, ys:mnist.test.labels, keep_prob:1.0})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)


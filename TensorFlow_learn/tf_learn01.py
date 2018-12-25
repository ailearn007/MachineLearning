#coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorboard

def add_layer(inputs, in_size, out_size, n_layer, activation = None):
    layer_name = 'layer__{0}'.format(n_layer)
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name= 'W')
            tf.summary.histogram(layer_name + 'Weighttt', W)
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name= 'b')
            tf.summary.histogram(layer_name + 'bia', b)
        with tf.name_scope('linear_Z'):
            Z = tf.matmul(inputs, W) + b
        if activation == None:
            outputs = Z
        else:
            outputs = activation(Z)
        tf.summary.histogram(layer_name + '/outputs', outputs)

    return outputs

if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    with tf.name_scope('Inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name= 'X_inputs')
        ys = tf.placeholder(tf.float32, [None, 1], name= 'Y_inputs')

    Layer_1 = add_layer(xs, 1, 10, 1, tf.nn.relu)
    prediction = add_layer(Layer_1, 10, 1, 2)

    with tf.name_scope('losses'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                              reduction_indices= [1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('logs/', sess.graph)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for i in range(1000):
        sess.run(train_step, feed_dict= {xs: x_data, ys:y_data})
        if i%20 == 0:
            print(sess.run(loss, feed_dict= {xs:x_data, ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value = sess.run(prediction, feed_dict= {xs: x_data, ys:y_data})
            lines = ax.plot(x_data, prediction_value, 'r-')
            plt.pause(0.2)

            result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
            writer.add_summary(result, i)


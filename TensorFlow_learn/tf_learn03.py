from __future__ import print_function
import tensorflow as tf
import numpy as np

W = tf.Variable([[1,2,3], [4,5,6]], dtype= tf.float32, name= 'Weight')
b = tf.Variable([[1,2,3]], dtype= tf.float32, name= 'bia')

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'my_net/save_net', global_step=0)
    print('save to path:', save_path)
    print(sess.run(W))
# saver = tf.train.Saver()
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#
#
# with tf.Session() as sess:
#     saver.restore(sess, 'my_net/save_net')
#     print("weight: ", sess.run(W))
#     print("bia: ", sess.run(b))

# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_net/save_net-0.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint('./my_net/'))
    print(sess.run(W))



# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, "my_net/")
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))

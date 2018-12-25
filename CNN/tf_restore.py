#coding = utf-8
import tensorflow as tf
import numpy as np

sess = tf.Session()
new_saver = tf.train.import_meta_graph('LeNet_5_tf/Le5-505.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('LeNet_5_tf/'))
sess.run(W)

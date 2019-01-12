# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf
x = [[1.,1.], [5.,2.]]
a = tf.reduce_mean(x, axis = 1)

sess = tf.Session()

print(sess.run(a))
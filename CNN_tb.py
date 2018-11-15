import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

# mnist = input_data.read_data_sets('./mnist', one_hot=True) # they has been normalized to range (0,1)
# 如果网络连接有问题，可以提前下载好minist数据集，使用下面这种方法导入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

with tf.name_scope('Inputs'):
     tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
     image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
     tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )

with tf.name_scope('Outputs'):
     output = tf.layers.dense(flat, 10)              # output layer

with tf.name_scope('loss'):
     loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
with tf.name_scope('Train'):
     train_op = tf.train.AdamOptimizer(LR).minimize(loss)
with tf.name_scope('accuracy'):
     accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
writer = tf.summary.FileWriter('./log', sess.graph)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

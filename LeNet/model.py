import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

class LeNet(object):
	def __init__(self, sess, x, y_):
		self.sess = sess
		self.x = x
		self.y_ = y_
		self.weight_variable = weight_variable
		self.bias_variable = bias_variable
		self.max_pool_2x2 = max_pool_2x2
		self.conv2d = conv2d

		self._build_model()

	def _build_model(self):
		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

		# Conv1 layer
		self.W_conv1 = self.weight_variable([5, 5, 1, 32])
		self.b_conv1 = self.bias_variable([32])
		self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = self.max_pool_2x2(self.h_conv1)

		# Conv2 layer
		self.W_conv2 = self.weight_variable([5, 5, 32, 64])
		self.b_conv2 = self.bias_variable([64])
		self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = self.max_pool_2x2(self.h_conv2)

		# FC layer
		self.W_fc1 = self.weight_variable([7*7*64, 1024])
		self.b_fc1 = self.bias_variable([1024])
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

		# Drop layer
		self.keep_prob = tf.placeholder(tf.float32)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# Output layer
		self.W_fc2 = self.weight_variable([1024, 10])
		self.b_fc2 = self.bias_variable([10])
		self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

		# Loss
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
		# Adam
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		# Accuracy
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
	
	def train(self):
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		tf.global_variables_initializer().run()
		for i in range(20000):
			batch = self.mnist.train.next_batch(50)
			if i % 100 == 0:
				self.train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
				print("step %d, train_accuracy %g" % (i, self.train_accuracy))
				self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
		print("test accuracy %g" % self.accuracy.eval(feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels, self.keep_prob: 1.0}))
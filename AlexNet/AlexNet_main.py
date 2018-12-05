import tensorflow as tf 
import numpy as np
import pickle
from CIFARHelper import CifarHelper
from AlexNet_model import AlexNet

def main():

	CIFAR_DIR = '/home/lyc-zc/AlexNet/cifar-10-batches-py/'
	dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
	all_data = [0,1,2,3,4,5,6]

	for i,direc in zip(all_data,dirs):
		all_data[i] = pickle.load(open(CIFAR_DIR+direc, 'rb'))

	#placeholder for input and dropout rate
	x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
	y_true = tf.placeholder(tf.float32, shape = [None, 10])
	keep_prob = tf.placeholder(tf.float32)

	# Create the AlexNet model
	model = AlexNet(x = x, keep_prob = keep_prob, num_classes = 10)

	#define activation of last layer as score
	score = model.fc8
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = score))

	# The optimiser used in this implementation is different
	# to that used in the paper.
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
	train = optimizer.minimize(cross_entropy)

	# Initialize all global variables
	init = tf.global_variables_initializer()

	# steps = 10,000 will create 20 epochs.
	# There are a total of 50,000 images in the training set.
	# (10,000 * 100) / 50,000 = 20
	steps = 10001

	ch = CifarHelper(all_data)
	# pre-processes the data.
	ch.set_up_images()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(steps):
			# get next batch of data.
			batch = ch.next_batch(100)
			# On training set.
			sess.run(train, feed_dict = {x : batch[0], y_true : batch[1], keep_prob : 0.5})
			# Print accuracy after every epoch.
			# 500 * 100 = 50,000 which is one complete batch of data.
			if i%500 == 0:
				
				print("EPOCH: {}".format(i / 500))
				print("ACCURACY ")
				
				matches = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
				acc = tf.reduce_mean(tf.cast(matches, tf.float32))
				
				# On valid/test set.
				print(sess.run(acc, feed_dict = {x : ch.test_images, y_true : ch.test_labels, keep_prob : 1.0}))
				print('\n')

if __name__ == '__main__':

	main()
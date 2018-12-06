import argparse
import tensorflow as tf 
import numpy as np
import pickle
from CIFARHelper import CifarHelper
from LeNet_model import LeNet
from AlexNet_model import AlexNet
from VGG16_model import VGG16

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", dest='model_type', default='LeNet', help='type of model')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='cifar-10-batches-py/', help='path of the dataset')

args = parser.parse_args()

def main():

	# Get data.
	CIFAR_DIR = args.dataset_dir
	dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
	all_data = [0,1,2,3,4,5,6]

	for i,direc in zip(all_data,dirs):
		all_data[i] = pickle.load(open(CIFAR_DIR+direc, 'rb'))

	x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
	y_true = tf.placeholder(tf.float32, shape = [None, 10])
	keep_prob = tf.placeholder(tf.float32)

	if args.model_type == 'LeNet':
		model = LeNet(x = x, keep_prob = keep_prob, num_classes = 10)
		score = model.y_conv
	elif args.model_type == 'AlexNet':
		model = AlexNet(x = x, keep_prob = keep_prob, num_classes = 10)
		score = model.fc8
	elif args.model_type == 'VGG16':
		model = VGG16(x = x, keep_prob = keep_prob, num_classes = 10)
		score = model.fc3

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = score))

	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
	train = optimizer.minimize(cross_entropy)

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
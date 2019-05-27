import argparse
import tensorflow as tf 
import numpy as np
import pickle
from CIFARHelper import CifarHelper
from LeNet_model import LeNet
from AlexNet_model import AlexNet
from VGG16_model import VGG16
from GoogLeNet_model import GoogLeNet
from ResNet_model import ResNet50

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", dest='model_type', default='LeNet', help='type of model')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='cifar-10-batches-py/', help='path of the dataset')
parser.add_argument('--model_dir', dest='model_dir', default='model/', help='path to saving/restoring model')

args = parser.parse_args()

def unpickle(file):

	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def main():

	# Path to loading model.
	if args.model_dir[-5:] != '.ckpt':
		MODEL_DIR = args.model_dir + args.model_type + '.ckpt'
		PRE_TRAINED = False
	else:
		# Train from pretrained model.
		MODEL_DIR = args.model_dir
		PRE_TRAINED = True
		print('Training based on {}'.format(MODEL_DIR))

	# Load data.
	CIFAR_DIR = args.dataset_dir
	dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
	all_data = [0,1,2,3,4,5,6]
	for i,direc in zip(all_data, dirs):
		all_data[i] = unpickle(CIFAR_DIR + direc)
	print("CIFAR10 has loaded!")

	x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
	y_true = tf.placeholder(tf.float32, shape = [None, 10])
	keep_prob = tf.placeholder(tf.float32)

	if args.model_type == 'LeNet':
		model = LeNet(x = x, keep_prob = keep_prob, num_classes = 10)
	elif args.model_type == 'AlexNet':
		model = AlexNet(x = x, keep_prob = keep_prob, num_classes = 10)
	elif args.model_type == 'VGG16':
		model = VGG16(x = x, keep_prob = keep_prob, num_classes = 10)
	elif args.model_type == 'GoogLeNet':
		model = GoogLeNet(x = x, keep_prob = keep_prob, num_classes = 10)
	elif args.model_type == 'ResNet50':
		model = ResNet50(x = x, keep_prob = keep_prob, num_classes = 10)
	else:
		print("'{}' is not recognized. "
			"Use 'LeNet' / 'AlexNet' / 'VGG16' / 'GoogLeNet' / 'ResNet50'. ".format(args.command))

	y = model.output

	with tf.name_scope('cross_entropy'): 
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y))
	tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
		train = optimizer.minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
		acc = tf.reduce_mean(tf.cast(matches, tf.float32))
	tf.summary.scalar('accuracy', acc)

	# Add an op to initialize the variables.
	init = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	# steps = 10,000 will create 20 epochs.
	# There are a total of 50,000 images in the training set.
	# (10,000 * 100) / 50,000 = 20
	steps = 10000

	ch = CifarHelper(all_data)
	# pre-processes the data.
	ch.set_up_images()

	with tf.Session() as sess:

		if PRE_TRAINED:
			# Restore variables from disk.
			saver.restore(sess, MODEL_DIR)
			print("Model restored.")
		else:
			# Initialize the variables.
			sess.run(init)

		# Write Logs
		merged_summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter('logs/', sess.graph)

		for i in range(steps):
			# Get next batch of data.
			batch = ch.next_batch(100)
			# On training set.
			sess.run(train, feed_dict = {x : batch[0], y_true : batch[1], keep_prob : 0.5})
			# Print accuracy after every epoch.
			# 500 * 100 = 50,000 which is one complete batch of data.
			if i % 500 == 0:
				print("EPOCH: {}".format(i / 500))
				print("ACCURACY ")
				# On valid/test set.
				print(sess.run(acc, feed_dict = {x : ch.test_images, y_true : ch.test_labels, keep_prob : 1.0}))
				summary_writer.add_summary(sess.run(merged_summary_op, feed_dict = {x : ch.test_images, y_true : ch.test_labels, keep_prob : 1.0}), i)
		
		# Save the variables to disk.
		save_path = saver.save(sess, MODEL_DIR)
		print("Model saved in path: %s" % save_path)

		summary_writer.close()

if __name__ == '__main__':

	main()
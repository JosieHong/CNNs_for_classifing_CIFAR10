import tensorflow as tf 
from model import LeNet

def main():
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	sess = tf.InteractiveSession()
	model = LeNet(sess, x, y_)
	model.train()

if __name__ == '__main__':
    main()
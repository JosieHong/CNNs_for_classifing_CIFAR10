import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from LeNet_model import leNet

def main():
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	sess = tf.InteractiveSession()
	lenet_model = LeNet(sess, x, y_)

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	tf.global_variables_initializer().run()
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = lenet_model.accuracy.eval(feed_dict={lenet_model.x: batch[0], lenet_model.y_: batch[1], lenet_model.keep_prob: 1.0})
			print("step %d, train_accuracy %g" % (i, train_accuracy))
			lenet_model.train_step.run(feed_dict={lenet_model.x: batch[0], lenet_model.y_: batch[1], lenet_model.keep_prob: 0.5})
	print("test accuracy %g" % lenet_model.accuracy.eval(feed_dict={lenet_model.x: mnist.test.images, lenet_model.y_: mnist.test.labels, lenet_model.keep_prob: 1.0}))

if __name__ == '__main__':
    main()
import tensorflow as tf
import numpy as np

class alexNet(object):
    def __init__(self, x, keepPro, classNum, skip, modelPath = "bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath

        self._build_model()

    def _build_model(self):
    	# Layer1
        self.conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        self.lrn1 = LRN(self.conv1, 2, 2e-05, 0.75, "norm1")
        self.pool1 = maxPoolLayer(self.lrn1, 3, 3, 2, 2, "pool1", "VALID")

        # Layer2
        self.conv2 = convLayer(self.pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        self.lrn2 = LRN(self.conv2, 2, 2e-05, 0.75, "lrn2")
        self.pool2 = maxPoolLayer(self.lrn2, 3, 3, 2, 2, "pool2", "VALID")

        # Layer3
        self.conv3 = convLayer(self.pool2, 3, 3, 1, 1, 384, "conv3")

        # Layer4
        self.conv4 = convLayer(self.conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        # Layer5
        self.conv5 = convLayer(self.conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        self.pool5 = maxPoolLayer(self.conv5, 3, 3, 2, 2, "pool5", "VALID")

        # Layer6
        self.fcIn = tf.reshape(self.pool5, [-1, 256 * 6 * 6])
        self.fc1 = fcLayer(self.fcIn, 256 * 6 * 6, 4096, True, "fc6")
        self.dropout1 = dropout(self.fc1, self.KEEPPRO)

        # Layer7
        self.fc2 = fcLayer(self.dropout1, 4096, 4096, True, "fc7")
        self.dropout2 = dropout(self.fc2, self.KEEPPRO)

        # Output
        self.fc3 = fcLayer(self.dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            sess.run(tf.get_variable('w', trainable = False).assign(p))

	def maxPoolLayer(self, x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
	    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
	                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

	def dropout(self, x, keepPro, name = None):
	    return tf.nn.dropout(x, keepPro, name)

	def LRN(self, x, R, alpha, beta, name = None, bias = 1.0):
	    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
	                                              beta = beta, bias = bias, name = name)	

	def fc(self, x, inputD, outputD, reluFlag, name):
	    with tf.variable_scope(name) as scope:
	        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
	        b = tf.get_variable("b", [outputD], dtype = "float")
	        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
	        if reluFlag:
	            return tf.nn.relu(out)
	        else:
	            return out

	def convLayer(self, x, kHeight, kWidth, strideX, strideY,
	              featureNum, name, padding = "SAME", groups = 1):
	    channel = int(x.get_shape()[-1])
	    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
	    with tf.variable_scope(name) as scope:
	        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
	        b = tf.get_variable("b", shape = [featureNum])

	        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
	        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)	

	        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
	        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
	        out = tf.nn.bias_add(mergeFeatureMap, b)

	        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)
import argparse
import sys
import tempfile

# Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data

# Deep MNIST CNN to improve accuracy
import tensorflow as tf

def deepnn(x):
	"""deepnn builds the graph for a deep net for classifying digits.

	Args:
		x: an input tensor with the dimensions (N_example, 784), where 784 is the number of pixels

	Returns:
		A tuple (y, keep_prob). y is a tensor of shape(N_example, 10), with values equal to the digits of classifying the digit into one of the 10 classes. keep_prob is a scalar placeholder for the probability of dropout.
	
	"""
	# To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])

	with tf.name_scope('conv1'):
		## First Conv Layer
		# Consist of convolution, followed by max pooling. The convolution will compute 32 features for each 5x5 patch. Its weight tensor will ahve a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, the the last is the number of output channels. We will also have a bias vector with a component for each output channel.
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])

		# We then convolve x_image with the weight tensor and add the bias, apply the ReLU function and finally max pool. The max_pool_2x2 method will reduce the image size to 14x14
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)

	with tf.name_scope('conv2'):
		## Second Conv Layer
		# In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])

		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)

	with tf.name_scope('fc1'):
		## Fully Connected Layer
		# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling payer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU
		W_fc1 = weight_variable([7*7*64, 1024])
		b_fc1 = bias_varialbe([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	with tf.name_scope('dropout'):
		## Dropout Layer
		# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neruon output in addition to masking them, so dropout just works output any additional scaling.
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	with tf.name_scope('fc2'):
		## Readout Layer
		# Finally, a layer just like for the one layer softmax repression above
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])

		y_conv = tf.matmul(h_fc_drop, W_fc2) + b_fc2

	return y_conv, keep_prob



# Weight Initialization. 
# To create this model, we need to create a lot of weights and biases. Since we are using ReLU neurons, it is also good practice to initialize then with a slightly positive initial bias to avoid "dead neurons".
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and Pooling.
# TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is out stride size? We will do convolution useing a stride on one and are zero padded so that the output is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks.
def conv2d(x, W):
	return tf.nn.conv2d(x, W, stides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')



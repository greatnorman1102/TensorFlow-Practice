# Deep MNIST

# Load MNIST Data
from tensorflow.example.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start TensorFlow Interactive Session
import tensorflow as tf
sess = tf.InteractiveSession()

# Build a Soltmax Regression Model
x = tf.placeholder(tf.float32, shape=[None, 784]) # input images x will consist of a 2d tensor of floating point numbers 28*28=784
y_ = tf.placeholder(tf.float32, shape[None, 10]) # output classes is a one-hot 10-dimensional vector indicating which digit class the corresponding MNIST image belongs to

# Set Variables
W = tf.Variable(tf.zeros([784,10])) # W is a 784x10 matrix
b = tf.Variable(tf.zeros([10])) # b is a 10 dimensional vector

sess.run(tf.global_variables_initializer())

# Predicted Class and Loss Function
y = tf.matmul(x,W) + b # out regression model

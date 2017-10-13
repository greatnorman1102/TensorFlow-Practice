# Tensorflow Introduction

import tensorflow as tf

# The computational Graph

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
# Final print statement: Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run([node1, node2]))
# Final print statement: [3.0, 4.0]

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
# Final print statement: node3: Tensor("Add:0", shape=(), dtype=float32) sess.run(node3): 7.0

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
# Final print statement: 7.5 \n [3. 7.]

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
# Expect output 22.5

## Create a Simple model
# In machine learning, Variable allow us to add trainable parameters to a graph
# They are contructed with a type and initial value
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# init is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we call sess.run, the variables are uninitialized.
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# A loss function measures how far apart the current model is from the provided data. We'll use a stanard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data.
y = tf.placeholder(tf.float32)
square_deltas = tf. square(linear_model - y)
loss = tf.reduce_sum(square_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# A variable is initialized to the value provided to tf.Variable but can be changed using operatios like tf.assign. Now we set W=-1 and b=1, which is the optimal parameters for the above model. Now the loss should be 0
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Tensorflow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent. It modefies each variable according to the magnitude of the derivative of loss w.r.t. that variable. 
optimizer = tf. train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrent defaults.
for i in range(1000):
	sess.run(train, {x: [1, 2, 3,4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))

# Estimator in TensorFlow is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following: 1)running training loops, 2)running evaluation loops, 3)managing datasets. tf.estimator defines many common models

import tensorflow as tf
#NumPy is often used to load, manipulate and prepocess data.
import numpy as np

# Declare list of features
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation (inference). There are many predefined types like linear repression, linear classification, and many neural network classifiers and regressors. (For example, a linear regresssion is shown belwo)
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up datasets.
# Below we use one training set and one evaluation set, we have to tell the function how many batches of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])
x_eval = np.array([2, 5, 8, 1])
y_eval = np.array([-1.01, -4.1, -7, 0])
input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the method and passing the training dataset
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well out model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

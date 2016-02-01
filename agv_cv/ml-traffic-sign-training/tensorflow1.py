########################################################################################
# Description: Machine learning based on the TensorFlow tutorial "MNIST For ML Beginners"
# Input: "dataset.mat" with X and Y of the dataset          
# Output: Training accuracy and prediction on few data of the given dataset                
# Dataset Format: X is a (a,b) matrix. a is the number of images, while b 
#                   is the total number of pixels in the image.
#                 Y is a (a,1) matrix that defines the class each image in X belongs to.
# Author: Manash Pratim Das (mpdmanash@gmail.com)
########################################################################################

#Usage: python tensorflow1.py

import tensorflow as tf
import scipy.misc, scipy.io, scipy.optimize, scipy.special
import numpy as np

'''mat = scipy.io.loadmat( "/home/manash/Codes/Courses/Machine Learning/machine-learning-ex4/ex4/ex4data2.mat" )
X, Y = mat['X'], mat['y']
x_size = 400
num_labels = 10'''

mat = scipy.io.loadmat( "dataset.mat" )
X, Y = np.double(mat['X']), mat['y']
Y = Y.transpose()
x_pix = 45
x_size = x_pix*x_pix
num_labels = 5




x = tf.placeholder(tf.float32, [None, x_size])

W = tf.Variable(tf.zeros([x_size, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, num_labels])
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

m = X.shape[0]

yz = np.zeros((m,num_labels))

for i in range(0,m):
    yz[i,Y[i]-1] = 1
print np.shape(X)
print np.shape(yz)

for i in range(1000):
	q = np.random.choice(X.shape[0], 100, replace=False)
	batch_xs = X[q,:]
	batch_ys = yz[q,:]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

test = X[1:2,:]

feed_dict = {x: test}
classification = sess.run(y, feed_dict)
print classification

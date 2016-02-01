########################################################################################
# Description: Machine learning based on the TensorFlow tutorial "Deep MNIST for Experts"
# Input: "dataset.mat" with X and Y of the dataset          
# Output: Training accuracy and prediction on few data of the given dataset                
# Dataset Format: X is a (a,b) matrix. a is the number of images, while b 
#                   is the total number of pixels in the image.
#                 Y is a (a,1) matrix that defines the class each image in X belongs to.
# Author: Manash Pratim Das (mpdmanash@gmail.com)
########################################################################################

#Usage: python tensorflow2.py

import tensorflow as tf
import scipy.misc, scipy.io, scipy.optimize, scipy.special
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



'''mat = scipy.io.loadmat( "/home/manash/Codes/Courses/Machine Learning/machine-learning-ex4/ex4/ex4data2.mat" )
X, Y = mat['X'], mat['y']
x_size = 400
num_labels = 10'''

mat = scipy.io.loadmat( "dataset.mat" )
X, Y = np.double(mat['X']), mat['y']
Y = Y.transpose()
x_pix = 28
x_size = x_pix*x_pix
num_labels = 6

print np.shape(X)
print np.shape(Y)


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, x_size])
y_ = tf.placeholder(tf.float32, [None, num_labels])

W = tf.Variable(tf.zeros([x_size, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)


#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

m = X.shape[0]

yz = np.zeros((m,num_labels))

for i in range(0,m):
    yz[i,Y[i]-1] = 1
print np.shape(X)
print np.shape(yz)


for i in range(1000):
	q = np.random.choice(X.shape[0], 50)
	batch_xs = X[q,:]
	batch_ys = yz[q,:]
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict={x: X, y_: yz})

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,x_pix,x_pix,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([x_pix * x_pix * 64 / 16, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, x_pix * x_pix * 64 / 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, num_labels])
b_fc2 = bias_variable([num_labels])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
	q = np.random.choice(X.shape[0], 50)
	batch_xs = X[q,:]
	batch_ys = yz[q,:]
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print "step %d, training accuracy %g"%(i, train_accuracy)
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: X, y_: yz, keep_prob: 1.0})

'''
test = X[1:2,:]

feed_dict = {x: test}
classification = sess.run(y, feed_dict)
print classification'''

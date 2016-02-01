########################################################################################
# Description: Train a neural network with a dataset, predict few random images and 
#              calculate accuracy of the training.                
# Input: "dataset.mat" with X and Y of the dataset          
# Output: Training accuracy and prediction on few data of the given dataset                
# Dataset Format: X is a (a,b) matrix. a is the number of images, while b 
#                   is the total number of pixels in the image.
#                 Y is a (a,1) matrix that defines the class each image in X belongs to.
# Author: Manash Pratim Das (mpdmanash@gmail.com)
########################################################################################

#Usage: python nn_train.py

import numpy as np
import scipy.optimize as scipy
import scipy.misc, scipy.io, scipy.optimize, scipy.special
from matplotlib import pyplot, cm

vlambda = 1

def sigmoid(z):
    g = 1.0 / (1.0 +  np.exp(-z) )
    return g

def sigmoidGradient(z):
    g = np.zeros(np.shape(z));
    g = np.multiply(sigmoid(z),(1-sigmoid(z)));
    return g

def paramUnroll( nn_params, input_layer_size, hidden_layer_size, num_labels ):
    theta1_elems = ( input_layer_size + 1 ) * hidden_layer_size
    theta1_size  = ( input_layer_size + 1, hidden_layer_size  )
    theta2_size  = ( hidden_layer_size + 1, num_labels )

    theta1 = nn_params[:theta1_elems].T.reshape( theta1_size ).T    
    theta2 = nn_params[theta1_elems:].T.reshape( theta2_size ).T

    return (theta1, theta2)

def feedForward( theta1, theta2, X, X_bias = None ):
    one_rows = np.ones((1, np.shape(X)[0] ))
    
    a1 = [one_rows, X.T]  if X_bias is None else X_bias
    z2 = theta1.dot( a1 )
    a2 = sigmoid(z2)
    a2 = [one_rows, a2] 
    z3 = theta2.dot( a2 )
    a3 = sigmoid( z3 )

    return (a1, a2, a3, z2, z3)

def nnCostFunction(nn_params,*args):
    input_layer_size,hidden_layer_size,num_labels,X,y,v_lambda = args
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)) ,order = 'F' )
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)) , order = 'F');
    m = X.shape[0];
    J = 0;    

    z2 = np.dot(X,Theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.concatenate( (np.ones((m,1)), a2) , axis=1 )
    z3 = np.dot(a2,Theta2.transpose())
    a3 = sigmoid(z3)
    Jz = np.zeros((m,num_labels));
    yz = np.zeros((m,num_labels));

    for i in range(0,m):
        yz[i,y[i]-1] = 1
        Jz[i,:] = np.multiply(-yz[i,:],np.log(a3[i,:])) - np.multiply((1-yz[i,:]) , np.log(1-a3[i,:]));

    J_reg = np.sum(np.power(Theta1[:,1:],2) ) + np.sum(np.power(Theta2[:,1:],2) );
    J_reg = (v_lambda/(2*m))*J_reg;
    J = (np.sum(Jz)/m) + J_reg;
    print J
    return J

def nnGradient(nn_params,*args):
    input_layer_size,hidden_layer_size,num_labels,X,y,v_lambda = args
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)) ,order = 'F' )
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)) , order = 'F');
    m = X.shape[0];
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

    z2 = np.dot(X,Theta1.transpose())

    a2 = sigmoid(z2)
    a2 = np.concatenate( (np.ones((m,1)), a2) , axis=1 )
    z3 = np.dot(a2,Theta2.transpose())
    a3 = sigmoid(z3)
    yz = np.zeros((m,num_labels));
    for i in range(0,m):
        yz[i,y[i]-1] = 1

    del3 = a3 - yz;
    temp = sigmoidGradient(z2);
    temp2 = np.dot(del3,Theta2)
    del2 = np.multiply(temp2[:,1:],temp);

    Theta1_grad = (np.dot(del2.transpose() , X))/m;
    Theta2_grad = (np.dot(del3.transpose() , a2))/m;
    Theta1_reg = Theta1*(v_lambda/m);
    Theta2_reg = Theta2*(v_lambda/m);
    Theta1_reg[:,0] = np.zeros(Theta1.shape[0]);
    Theta2_reg[:,0] = np.zeros(Theta2.shape[0]);

    Theta1_grad = Theta1_grad + Theta1_reg;
    Theta2_grad = Theta2_grad + Theta2_reg;

    grad = np.concatenate((Theta1_grad.ravel('F') , Theta2_grad.ravel('F')), axis=0)
    return grad

def randInitializeWeights(L_in, L_out):
    Einit = np.sqrt(6)/np.sqrt(L_in+L_out)
    W = np.random.rand(L_out,1+L_in)*2*Einit - Einit
    return W

def predict( X, theta1, theta2 ):
    n = np.shape(X)[0]
    X = np.reshape(X, (1, n) )
    m = X.shape[0];
    z2 = np.dot(X,theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.concatenate( (np.ones((m,1)), a2) , axis=1 )
    z3 = np.dot(a2,theta2.transpose())
    a3 = sigmoid(z3)
    return np.argmax(a3) + 1

def displayData( X, X1, theta1 = None, theta2 = None ):
    m, n = np.shape( X )
    width = np.sqrt( n )
    rows, cols = 5, 5

    out = np.zeros(( width * rows, width*cols ))

    rand_indices = np.random.permutation( m )[0:rows * cols]

    counter = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x * width
            start_y = y * width
            out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1

    img     = scipy.misc.toimage( out )
    figure  = pyplot.figure()
    axes    = figure.add_subplot(111)
    axes.imshow( img )

    if theta1 is not None and theta2 is not None:
        result_matrix   = []
        
        for idx in rand_indices:
            result = predict( X1[idx], theta1, theta2 )
            result_matrix.append( result )

        result_matrix = np.array( result_matrix ).reshape( rows, cols ).transpose()
        print result_matrix

    pyplot.show( )



# Implementation same as NN training taught at the Andrew Ng's Machine Learning Coursera course.



pixel_size = 28;
input_layer_size  = pixel_size*pixel_size; 
hidden_layer_size = 25; 
maxiter = 50 
num_labels = 6; 
v_lambda = 1.0 


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate((initial_Theta1.ravel('F') , initial_Theta2.ravel('F')), axis=0)

'''mat = scipy.io.loadmat( "/home/manash/Codes/Courses/Machine Learning/machine-learning-ex4/ex4/ex4data2.mat" )
X, Y = mat['X'], mat['y']'''


mat = scipy.io.loadmat( "dataset.mat" )
X, Y = mat['X'], mat['y']
Y = Y.transpose()

print np.shape(X)
print np.shape(Y)

m = X.shape[0];
X1 = np.concatenate( (np.ones((m,1)), X) , axis=1 )


args = (input_layer_size,hidden_layer_size,num_labels,X1,Y,v_lambda)
result = scipy.optimize.fmin_cg( nnCostFunction, x0=initial_nn_params, fprime=nnGradient, args=args, maxiter=maxiter, disp=True, full_output=True)
opt_theta = result[0]
Theta1 = np.reshape(opt_theta[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)) ,order = 'F' )
Theta2 = np.reshape(opt_theta[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)) , order = 'F');

displayData(X,X1,Theta1,Theta2)
counter = 0
for i in range(0, m):
    prediction = predict( X1[i], Theta1, Theta2 )
    actual = Y[i]
    if( prediction == actual ):
        counter+=1
print "Counter:"
print counter * 100 / m

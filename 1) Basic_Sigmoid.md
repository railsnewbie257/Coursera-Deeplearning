<h2>Basic Sigmoid</h2>

<pre>
# GRADED FUNCTION: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    <b>s = 1 / (1 + math.exp(-x))</b>
    ### END CODE HERE ###
    
    return s
 </pre>
 
 <h2>Sigmoid</h2>
 
 <pre>
 # GRADED FUNCTION: sigmoid

import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    <b>s = 1 / (1 + np.exp(-x))</b>
    ### END CODE HERE ###
    
    return s
 </pre>
 
 <h2>Sigmoid Derivative</h2>
 
 <pre>
 # GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    <b>s = sigmoid(x)
    ds = s * (1-s)</b>
    ### END CODE HERE ###
    
    return ds
 </pre>
 
 <h2>Normalize Rows</h2>
 
 <pre>
 # GRADED FUNCTION: normalizeRows

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    <b>x_norm = np.linalg.norm(x,axis=1,keepdims=True)</b>
    
    # Divide x by its norm.
    <b>x = x / x_norm</b>
    ### END CODE HERE ###

    return x
 </pre>
 
What you need to remember:

- np.exp(x) works for any np.array x and applies the exponential function to every coordinate
- the sigmoid function and its gradient
- image2vector is commonly used in deep learning
- np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs.
- numpy has efficient built-in functions
- broadcasting is extremely useful

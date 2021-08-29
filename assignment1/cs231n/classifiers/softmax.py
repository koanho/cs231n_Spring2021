from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dim = X.shape[1]
    n_samples = X.shape[0]
    n_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # loss = -log(e^-zc/ sum(e^-zi))

    
    for i, x in enumerate(X):
      scores = np.dot(x ,  W)
      softmax = np.exp(scores[y[i]]) / np.sum(np.exp(scores))
      loss += - np.log(softmax)

      coeff = np.zeros((1, n_classes))

      coeff = -(np.exp(scores) * np.exp(scores[y[i]]) / (np.sum(np.exp(scores)) **2))
      coeff[y[i]] = ( (np.exp(scores[y[i]]) * np.sum(np.exp(scores))) -  np.exp(2 * scores[y[i]])) / (np.sum(np.exp(scores)) ** 2)

      #print(x.shape, coeff.shape)
      dW += (np.dot(x.reshape(-1, 1), coeff.reshape(1, -1))) * (-(1/softmax))

  
    loss /= n_samples
    loss += np.sum(W*W)

    dW /= n_samples
    dW +=  2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dim = X.shape[1]
    n_samples = X.shape[0]
    n_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X, W)
    
    softmax_denomi = np.sum(np.exp(scores), axis = 1)
    softmax_nomi = np.exp(scores[range(n_samples), list(y)])
    softmax = softmax_nomi / softmax_denomi

    loss = np.sum( -np.log(softmax) ) / n_samples
    loss += reg * np.sum(W*W)

    coeff = np.zeros((n_samples, n_classes))
    intermediate = -  (1 / softmax)
    
    coeff = -(np.exp(scores) * (softmax_nomi.reshape(n_samples, 1)) ) / (softmax_denomi ** 2).reshape(n_samples, 1)
    coeff[range(n_samples), y] = (softmax_nomi * softmax_denomi - np.exp(2 * scores[range(n_samples), y]) ) / (softmax_denomi ** 2)
    coeff *= intermediate.reshape(n_samples, 1)

    dW = np.dot(X.T, coeff) / n_samples
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

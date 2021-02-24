import numpy as np
import tensorflow as tf
def gram(A):
    nh,nw,nc = A.shape
    A = A.reshape((nh*nw,nc))
    return np.matmul(np.transpose(A),A)

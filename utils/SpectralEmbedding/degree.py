import numpy as np
import scipy.sparse as sps

def degree(weights):
    """
    Computes the degree matrix from an adjacency matrix
    weights = matrix
    D = Diagoal degree matrix
    Dinv = Inverse Diagonal degree matrix
    """

    n = weights.shape[0]
    D = sps.spdiags(np.sum(weights,axis=0),0,n,n)
    Dinv = sps.spdiags(1/np.sum(weights,axis=0),0,n,n)
    return D,Dinv

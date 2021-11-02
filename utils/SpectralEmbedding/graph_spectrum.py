from scipy.sparse.linalg import eigs
import numpy as np

def eigen_values_spectrum(laplace,ne):
            
        """
        Computes the eigen values and vectors 

        Inputs: L Random walk/laplacian matrix
        ne : number of eigen values

        returns: Sorted eigen values and eigen vectors

        """
        Lambda, vectors = eigs(laplace,k = ne+1,sigma = 0, maxiter= 5000,tol = 1e-3)

        
        # Lambda = np.diag(Lambda)
        Lambda= Lambda.real
        vectors = vectors.real 

        #sorting
        idx = np.argsort(Lambda)
        Lambda.sort()
        vectors = vectors[:,idx]
        
        if sum(Lambda<1e-10)>1:  print('Multiple null vectors')

        signf = 1 - 2*(vectors[0,:]<0)

        vectors = vectors * signf
        
        return Lambda,vectors
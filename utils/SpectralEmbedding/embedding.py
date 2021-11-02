import numpy as np
import scipy.sparse as sps
import scipy
from scipy.sparse.linalg import eigs
import math

from utils.SpectralEmbedding.degree import degree
from utils.SpectralEmbedding.weight_adjaceny import weight_adjacency_matrix
from utils.SpectralEmbedding.graph_spectrum import eigen_values_spectrum


class Embedding:

    def __init__(self,obj):
        
        self.coords,self.faces = obj.coords, obj.faces
        self.coords_rows = obj.coords_rows
        self.n = obj.n
        self.depth = obj.depth
        self.thickness = obj.thickness
        self.mode = obj.mode
        self.ne = obj.ne
        self.P_corrected = obj.P_corrected
        self.weights = obj.weights


    def embedding(self):
        
            """
            Computes the spectral embedding of the graph

            self.weights: weight adjacency matrix
            self.D = diagonal degree matrix
            self.Dinv = D^-1
            self.Lambda = eigen values
            self.vectors = eigen vectors

            returns: self.X Spectral embedded (K* ne)
            
            """

            self.weights = weight_adjacency_matrix(self.coords, self.faces) #compute the weighted adjacency matrix (weight affinities)
            big = float('inf')
            assert self.weights!=None

            [r,c,v] = sps.find(self.weights) #find non zeros values
            v[v>big] = big #threshold/truncate big values

            self.D,self.Dinv = degree(self.weights) #Computes the degree matrix from an adjacency matrix

            self.Dsqrt = sps.spdiags(np.asarray(np.sum(self.weights,axis=0))**0.5,0,self.coords_rows,self.coords_rows)
            self.Dinvsqrt = sps.spdiags(np.asarray(np.sum(self.weights,axis=0))**(-0.5),0,self.coords_rows,self.coords_rows)

            if self.mode == 'normalized':
                self.laplace = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(self.weights, normed=True, return_diag=False, use_out_degree=False)) #get the graph laplacian

                Lambda, vectors = eigen_values_spectrum(self.laplace,self.ne)
                self.Lambda = Lambda[1:]
                self.vectors = vectors[:,1:]

                self.X =  np.dot(self.Dinvsqrt*self.vectors ,np.diag(self.Lambda ** (-0.5))) #normalized laplacian embedding (144559*144559), (144559*5) (5*5) = 144559*5

            elif self.mode == 'randomwalk':
                self.laplace = np.dot(self.Dinv,self.D-self.weights)

                Lambda, vectors = eigen_values_spectrum(self.laplace,self.ne)
                self.Lambda = Lambda[1:]
                self.vectors = vectors[:,1:]

                self.X = np.dot(self.vectors, np.diag(self.Lambda ** (-0.5))) #randomwalk embedding 


            else:
                print('Unown embedding mode ',self.mode)

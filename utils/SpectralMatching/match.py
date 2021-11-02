
import numpy as np
import scipy.spatial as spatial

def match(E1=None,E2=None):
        """
        Finds the correspondence between embeddings E1 and E2

        returns corr12: correspondence from E1 onto E2 (n1x1)
        corr21: correspondence from E2 onto E1 (n1x1)

        """

        #find correspondence between E1 onto E2

        corr12 = np.zeros((np.shape(E1)[0],1))
        tree =  spatial.KDTree(E2)  #nearest neigbor lookup 
        corr12 = tree.query(E1)[1]
        del(tree)

        #find correspondence between E2 and E1

        corr21 = np.zeros((np.shape(E2)[0],1))
        tree =  spatial.KDTree(E1)  #nearest neigbor lookup 
        corr21 = tree.query(E2)[1]
        
        # self.corr21 = corr21
        # self.corr12 = corr12
        del(tree)

        return corr12,corr21
import numpy as np
import scipy.sparse as sps

def weight_adjacency_matrix(coords,faces):
        """
    
        coords: vertex position (nx3)
        faces: traingulation (nx3) vetex indices for each triangle

        Returns n x n weight adjacency matrix
        """

        vertice_ax0_1 = np.sum([(coords[faces[:,0].T , :].T - coords[faces[:,1].T ,:].T) ** 2], #remove 1
            axis = 1).T
        vertice_ax0_2 = np.sum([(coords[faces[:,0].T , :].T - coords[faces[:,2].T ,:].T) ** 2],
            axis = 1).T
        vertice_ax1_0 = np.sum([(coords[faces[:,1].T , :].T - coords[faces[:,0].T ,:].T) ** 2],
            axis = 1).T
        vertice_ax1_2 = np.sum([(coords[faces[:,1].T , :].T - coords[faces[:,2].T ,:].T) ** 2],
            axis = 1).T
        vertice_ax2_0 = np.sum([(coords[faces[:,2].T , :].T - coords[faces[:,0].T ,:].T) ** 2],
            axis = 1).T
        vertice_ax2_1 = np.sum([(coords[faces[:,2].T , :].T - coords[faces[:,1].T ,:].T) ** 2], 
            axis = 1).T
        
        weights = np.vstack((vertice_ax0_1,vertice_ax0_2,vertice_ax1_0,vertice_ax1_2,vertice_ax2_0,vertice_ax2_1)) ** 0.5
        weights = 1/weights #inverse

        rows = np.vstack((np.array([faces[:, 0]]).T, np.array([faces[:, 0]]).T,
                        np.array([faces[:, 1]]).T, np.array([faces[:, 1]]).T,
                        np.array([faces[:, 2]]).T, np.array([faces[:, 2]]).T))
        cols = np.vstack((np.array([faces[:, 1]]).T, np.array([faces[:, 2]]).T,
                        np.array([faces[:, 0]]).T, np.array([faces[:, 2]]).T,
                        np.array([faces[:, 0]]).T, np.array([faces[:, 1]]).T))

        temp_array = np.append(rows, cols, axis=1)
        unique_arr, index = np.unique(temp_array, axis=0, return_index=True)
        weights = weights[index]

        # self.weights = 
        return(sps.csr_matrix((weights.flatten(), (unique_arr[:, 1].flatten(), unique_arr[:, 0].flatten())),
                                dtype=float))
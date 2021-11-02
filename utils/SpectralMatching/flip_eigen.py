
import numpy as np 

def flip_eigen_sign(M1,M2,ne):
        """
        Flip eigen vector signs of M2 such that it matches M1

        M1 = Reference embedding
        M2 = Embedding to flip sign
        ne = number of eigen vectors

        """
        for i in range(0,1):
            
            offset1 = np.mean(M1.coords[:,0:3],axis=0)
            offset2 = np.mean(M2.coords[:,0:3],axis=0)

            

            X1= M1.coords[:,0:3]-offset1  #center the mean (zero mean)
            X2 = M2.coords[:,0:3]-offset2

            
            #reference
            #weighted barycenters of poles
            w1 = np.sign(M1.X[:,i]) * (abs(M1.X[:,i]**3))
            
            #positive pole barycenter
            w = w1.copy()
            w[w<0] = 0
            w = w/np.sum(w,axis=0)
            w=np.expand_dims(w,1)      
            avgX1p = np.sum(X1*w,axis=0)

            del(w)
            #negative pole barycenter
            w = w1.copy()
            w[w>0] = 0
            w = w/np.sum(w,axis=0)
            w=np.expand_dims(w,1)
            avgX1m = np.sum(X1*w,axis=0)
        
            del(w)
            #to transform
            w2 = np.sign(M2.X[:,i]) * (abs(M2.X[:,i]**3))
            #positive pole barycenter
            w = w2.copy()
            w[w<0] = 0
            w = w/np.sum(w,axis=0)
            w=np.expand_dims(w,1)
            avgX2p = np.sum(X2*w,axis=0)

            del(w)
            #negative pole barycenter
            w=w2.copy()
            w[w>0] = 0
            w = w/np.sum(w,axis=0)
            w=np.expand_dims(w,1)
            avgX2m = np.sum(X2*w,axis=0)
            
            #distance betweem matched poles
            distp = np.sum((avgX1p - avgX2p)**2) + np.sum((avgX1m - avgX2m)**2)
            distm = np.sum((avgX1p - avgX2m)**2) + np.sum((avgX1p - avgX2p)**2)
            
            if distm < distp:
                if do_verbose:
                    print('Flip ',str(i))
                M2.vectors[:,i] = - M2.vectors[:,i]
                M2.X[:,i] = - M2.X[:,i]
            return M2
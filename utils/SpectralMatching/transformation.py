
import numpy as np
import random
import math

from utils.SpectralMatching.flip_eigen import flip_eigen_sign
from utils.SpectralMatching.match import match

class Matching:

  
        
    def get_transformation(self,ref,to_align,krot,matching_samples,w_sulcal,ki,matching_mode):
        """
        Performs spectral alignment of brain surfaces 

            ref: reference embedding
            ref.coords: vertices 
            ref.faces: mesh faces
            ref.depth: sulcal depth (surface data)


        to_align: embedding to be aligned to ref
        krot : number of eigen vectors used for transformation
        matching_samples : number of points used to find transformation
        matching_mode : partial or complete
        ki = number of eigen vectors used for matching
        w_sulcal : sulcal weights


        corr12 = map from mesh 1 to mesh 2
        corr21 = map from mesh 2 to mesh 1
        R12 = spectral transformation from E1 to E2
        R21 = spectral transformation from E2 to E1 


        Returns the aligned spectral embedding
        self.R12
        self.R21 
        self.Mw
        self.min_ssd
        """

        niter = 5*krot #50
        self.to_align = flip_eigen_sign(ref,to_align,krot)

        Mw = self.to_align

        best_R12 = []
        best_R21 = []

        R12 = np.eye(krot+1)
        R21 = np.eye(krot+1)

        U1 = np.column_stack((np.expand_dims(np.ones(ref.n),1).astype('int')/np.sqrt(ref.n), ref.vectors[:,0:krot+1]))
        U2 = np.column_stack((np.expand_dims(np.ones(self.to_align.n),1).astype('int')/np.sqrt(self.to_align.n), self.to_align.vectors[:,0:krot+1]))

        ssd_fwd = []
        ssd_bwd = []
        ssd = []
        min_ssd = 1e10

        for iter_match in range(niter):
    
            #Step1: truncate the eigen vectors
            trunc_id = min(5+1+np.ceil((iter_match-1)*0.65),krot+1)
            trunc_id = int(trunc_id)

            trunc = np.zeros((krot+1,krot+1))
            trunc[0:trunc_id+1,0:trunc_id+1] = np.eye(trunc_id)  #create an identity matrix
            trunc = trunc.astype('int')
            
            random.seed(1)

            ki_match = ki #number of eigen vectors to match
            if iter_match == 0:
                ki_match = 3 #start with 3, less ambiguous

            
            if matching_mode=='complete': #complete mesh
                
                c1,c2 = match(np.column_stack((w_sulcal*ref.depth, ref.X[:,0:ki_match])), np.column_stack((w_sulcal*self.to_align.depth, self.to_align.X[:,0:ki_match])))
                
                c = {'corr12':c1,'corr21':c2}
                
                
                #change base
                U12 = U1[c['corr21'],:]
                R12 = np.dot(U2.conjugate().T,U12)
                

                U21 = U2[c['corr12'],:]
                R21 = np.dot(U1.conjugate().T,U21)
                
            
            elif matching_mode=='partial': #partial mesh
                n = np.min(np.column_stack((matching_samples,ref.n,self.to_align.n)))
                idx1 = np.random.permutation(ref.n)
                idx1 = idx1[0:n]
                idx2 = np.random.permutation(self.to_align.n)
                idx2 = idx2[0:n]

                c1,c2 = match(np.column_stack((w_sulcal*ref.depth[idx1], ref.X[idx1,0:ki_match])) , np.column_stack((w_sulcal*self.to_align.depth[idx2], self.to_align.X[idx2,0:ki_match])))
                c = {'corr12':c1,'corr21':c2}

                Mw.c = c
                
                #change base
                U12 = U1[idx1[c['corr21']],:]
                U22 = U2[idx2,:]
                R12 = np.dot(U22.T,U12)

                U21 = U2[idx2[c['corr12']],:]
                U11 = U1[idx1,:]
                R21 = np.dot(U11.T,U21)

                #rescale weights based on downsampling

                R12 = R12 * (math.sqrt(ref.n*self.to_align.n)/n)
                R21 = R21 * (math.sqrt(ref.n*self.to_align.n)/n)
            
            #use first eigne vectors only 
            
            R12 = np.dot(np.dot( (np.dot(np.dot(trunc,R12),trunc) + np.eye(krot+1)-trunc), np.eye(krot+1)),self.to_align.n) / math.sqrt(self.to_align.n*ref.n)
            R21 = np.dot(np.dot( (np.dot(np.dot(trunc,R21),trunc) + np.eye(krot+1)-trunc), np.eye(krot+1)),self.to_align.n) / math.sqrt(self.to_align.n*ref.n)
    
            #make it symmetric
            nR12 = (R12 + R21.conjugate().T)/2
            nR21 = (R21 + R12.conjugate().T)/2
            del(R12)
            del(R21)
            R12 = nR12
            R21 = nR21

            #Transformed basis
            
            
            X2 = np.dot(np.dot(U2 ,R12), np.diag( np.insert(self.to_align.Lambda[0:krot],0,1,axis=0)**(-.5)))
            Mw.X[:,0:krot] = X2[:,1:]
            
            ki_ssd = 5

            #check energy
            if matching_mode == 'complete':
                ssd_fwd.append(np.sum( np.sum((np.column_stack((w_sulcal*ref.depth, ref.X[:,0:ki_ssd])) - np.column_stack((w_sulcal*Mw.depth[c['corr12']], Mw.X[c['corr12'],0:ki_ssd])) )**2,1)))
                ssd_bwd.append(np.sum( np.sum((np.column_stack((w_sulcal*ref.depth[c['corr21']], ref.X[c['corr21'],0:ki_ssd])) - np.column_stack((w_sulcal*Mw.depth, Mw.X[:,0:ki_ssd])) )**2,1)))

                
            elif matching_mode == 'partial':

                ssd_fwd.append(np.sum( np.sum((np.column_stack((w_sulcal*ref.depth[idx1], ref.X[idx1,0:ki_ssd])) - np.column_stack((w_sulcal*Mw.depth[idx2[c['corr12']]], Mw.X[idx2[c['corr12']],0:ki_ssd])) )**2,1)))
                ssd_bwd.append(np.sum( np.sum((np.column_stack((w_sulcal*ref.depth[idx1[c['corr21']]], ref.X[idx1[c['corr21']],0:ki_ssd])) - np.column_stack((w_sulcal*Mw.depth[idx2], Mw.X[idx2,0:ki_ssd])) )**2,1)))
           
            
            ssd.append(ssd_fwd[iter_match]+ssd_bwd[iter_match])

            #retain best transformation
            cur_ssd = ssd_fwd[iter_match]+ssd_bwd[iter_match]
            if cur_ssd < min_ssd:

                best_R12 = R12
                best_R21 = R21
                min_ssd = cur_ssd
            #check stopping criterion 
            if iter_match>4 and ssd[iter_match] > ssd[iter_match-2]:
                break
        
        self.R12 = best_R12
        self.R21 = best_R21

        Mw.c = c
        
        X2 = np.dot(np.dot(U2 ,R12), np.diag( np.insert(self.to_align.Lambda[0:krot],0,1,axis=0)**(-.5)))
        Mw.X[:,0:krot] = X2[:,1:]
        self.Mw = Mw 
        self.min_ssd = min_ssd

        # if self.plot:
        #         plot_graph(self.Mw.X[:,0:3],self.Mw.faces,self.Mw.vectors,num=0)
        #         plot_graph(self.Mw.vectors[:,0:3],self.Mw.faces,self.Mw.vectors,num=0)
                
                
        return R12,R21,Mw,min_ssd



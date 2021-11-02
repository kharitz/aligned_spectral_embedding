import os 
import glob
import pickle
import pandas as pd
import numpy as np
import time
import timeit

import scipy.spatial as spatial
import mne
from mne.transforms import apply_trans
from mne.io.constants import FIFF

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects 


import config
from utils.dice import dice
from utils.load_mesh import LoadMesh

from utils.SpectralEmbedding.embedding import Embedding
from utils.find_indices import find_indices
from utils.one_hot_encode import one_hot_ind
from utils.SpectralMatching.transformation import Matching
from utils.plot import plot,plot_graph


main_path = config.main_path #Path of scans
label_path = config.label_path #Path of manual labels
save_path = config.file_save_path

if not os.path.exists(save_path): #directory for saved files
    os.mkdir(save_path)


hemi = config.hemi 
ne = config.ne 
mode= config.mode 

krot = config.krot          #number of eigen vectos used for transformation
matching_samples = config.matching_samples  #  number of points used to find transformation #5000
matching_mode = config.matching_mode #partial or complete
niter = config.niter       #no iterations to refine basis transformation matrix (sign flip, reordering)
ki = config.ki             #eigen vectors used for matching
w_sulcal = config.w_sulcal #sulcal weights


patient_ids = sorted(os.listdir(label_path))
print('Number of scans: ',len(patient_ids))

ref = 'OASIS-TRT-20-1'

ref_data = LoadMesh()
ref_data.load_mesh(main_path,ref,hemi[0],ne,mode,label_path)

ref_data_embedded = Embedding(ref_data)
ref_data_embedded.embedding()


if ref in patient_ids:
    patient_ids.remove(ref)
    print('removed')
    print('After Removing:',len(patient_ids))

q = {}
q['X'] = ref_data_embedded.X; q['U'] = ref_data_embedded.vectors[:,0:3]; q['A'] = (ref_data_embedded.Dinvsqrt * (ref_data_embedded.D*ref_data_embedded.weights) * ref_data_embedded.Dinvsqrt); q['EUC'] = ref_data_embedded.coords; q['C'] = ref_data_embedded.depth
q['T'] = ref_data_embedded.thickness; q['F'] = ref_data_embedded.faces

ind = find_indices(ref_data_embedded.P_corrected, np.unique(ref_data_embedded.P_corrected))
man_one_hot = one_hot_ind(ind)
Y = man_one_hot
GT = man_one_hot 
q['Y']=Y; q['GT']=GT; 

file_handle = open(os.path.join(save_path,ref),'wb')
pickle.dump(q,file_handle)
file_handle.close()


del(q);del(ind);del(man_one_hot);del(Y);del(GT);

# spectral embedding and matching for other scans
start = timeit.default_timer()
st = start
for i in range(0,len(patient_ids)): #len(patient_ids)

    print('Patient ',patient_ids[i])
    mode = 'randomwalk'
    id_new = patient_ids[i]

    data_new = LoadMesh()
    data_new.load_mesh(main_path,id_new,hemi[0],ne,mode,label_path)
    
    data_new_embedded = Embedding(data_new)
    data_new_embedded.embedding()

    spec = Matching()
    
    R12,R21,Mw,min_ssd = spec.get_transformation(ref_data_embedded,data_new_embedded,krot,matching_samples,w_sulcal,ki,matching_mode='complete')

    X = Mw.X[:,0:3] #aligned spectral coordinates
    U = Mw.vectors[:,0:3] #unaligned spectral coordinates
    A = Mw.Dinvsqrt * (Mw.D+Mw.weights) * Mw.Dinvsqrt #weighted adjacency

    EUC = Mw.coords #euclidean corrdinates
    C = Mw.depth #sulcal depth
    T = Mw.thickness #cortical thickness
    F = Mw.faces #mesh face

    tree1 = spatial.KDTree(ref_data_embedded.X[:,0:3])  #nearest neigbor lookup 
    Idx = tree1.query(X[:,0:3])[1]

    if len(np.unique(Mw.P_corrected))==31:
        print('Adjusting Mw.P')
        a = ref_data_embedded.P_corrected[Idx]
        b = (a==0)
        Mw.P_corrected[b==True]=0
        del(a);del(b);

    dce = dice(ref_data_embedded.P_corrected[Idx],Mw.P_corrected)
    print('Dice ',np.mean(dce))
    
    ind = find_indices(Mw.P_corrected, np.unique(Mw.P_corrected))
    man_one_hot = one_hot_ind(ind)
    Y = man_one_hot
    GT = man_one_hot 

    q = {}
    q['X'] = X; q['U'] = U; q['A'] = A; q['EUC'] = EUC; q['C'] = C; q['T'] = T; q['F'] = F; q['Y']=Y; q['GT']=GT; 
    
    
    file_handle = open(os.path.join(save_path,id_new),'wb')
    pickle.dump(q,file_handle)
    file_handle.close()


    #ref mesh and spectral plot
    if config.plot==True:
        plot(ref_data_embedded.coords,ref_data_embedded.faces)
        plot_graph(ref_data_embedded.coords,ref_data_embedded.faces,ref_data_embedded.P_corrected,0)

        plot(Mw.coords,Mw.faces)
        plot_graph(Mw.coords[:,0:3],Mw.faces,ref_data_embedded.P_corrected[Idx],0)
        

    del(q);del(ind);del(man_one_hot);del(Y);del(GT);del(Mw);del(data_new);del(X);del(U);del(A);del(C);del(F);del(Idx);del(tree1);del(EUC);del(T)
    
    stop = timeit.default_timer()
    print('Time taken: ',(stop-start)/60,'m','  or ',(stop-start),' s')
    start = timeit.default_timer()

    print("########################################################")
    print("")
    
    
print('Total Time taken: ',(stop-st)/3600)






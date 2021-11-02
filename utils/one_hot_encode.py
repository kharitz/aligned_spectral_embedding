
import numpy as np
def one_hot_ind(a):
    one_h = np.zeros((max(a)+1,len(a)))

    for p,i in enumerate(a):
        one_h[i,p] = 1
    return np.transpose(one_h,axes=(1,0)).astype(int)
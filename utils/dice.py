import numpy as np

def dice(x,y):
    # smooth = 1e-4


    labels = np.unique(np.stack((np.unique(x),np.unique(y))))
    scores = np.zeros((np.size(labels)))
    for i in range(labels.size):
        label = labels[i]
        Ai = 1*(x==label)
        Bi = 1*(y==label)
       
        #get dice
        sc = 2* np.sum(np.logical_and(Ai,Bi)) / (np.sum(Ai) + np.sum(Bi))
        zr = (np.sum(x)+np.sum(y))
        if zr==0:
            sc = 0
        scores[i] = sc
    return scores
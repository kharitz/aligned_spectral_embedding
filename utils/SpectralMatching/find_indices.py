import numpy as np

def find_indices(a,b):
    q = []

    for i in a:
        if i not in b:
            q.append(0)
        else:
            q.append(np.where(b==i)[0][0])
    return q 


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import source as mycode

n = 100
n2 = 2*n
mh = np.random.normal(0,1,size = (n2, n2))
mh = 0.5j * (mh - mh.T) 
eig = np.linalg.eigvalsh(mh)
one_norm = 0
for i in range(n2):
    one_norm += abs(eig[i])
mh = mh/one_norm
mv = np.random.normal(0,1,size = (n,n))
np.save('mh.npy',mh)
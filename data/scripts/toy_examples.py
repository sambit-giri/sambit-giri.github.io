import numpy as np 
import matplotlib.pyplot as plt

import pickle
from glob import glob
from tqdm import tqdm 
import tools21cm as t2c


fig, axs = plt.subplots(1,1,figsize=(6,5))
xx = np.linspace(0,2*np.pi,200)
axs.plot(xx, np.sin(xx), lw=3, c='k')
axs.plot(xx, np.sin(xx)+0.5*(1.2+np.sin(xx))*np.sin(xx*10), label='positive')
axs.plot(xx, np.sin(xx)+0.5*(1.2-np.sin(xx))*np.sin(xx*10), label='negative')
axs.set_ylim(-2.2,2.2)
axs.legend()
plt.show()
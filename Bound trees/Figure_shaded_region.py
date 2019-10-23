#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: enfita, sdamrich
Indicates region of figure 1 that is used to compute the lower bound on the
number of forests that separate the trees
"""

import numpy as np
import numpy.ma as ma
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


raw=np.load('raw_CREMI25.npy')

seeds=np.load('seeds_CREMI25.npy')

mask = np.zeros_like(raw[:270, :])
mask[172:259, :] = 1

mask = ma.masked_where(mask == 0, mask)

plt.figure()
plt.axis('off')
plt.imshow(raw[:270],cmap='gray')
plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
color_scatter=np.concatenate((np.array([[1,0,0]]),np.array([[1,1,0]]*7),np.array([[127.5,214.5,247]])/255,np.array([[1,1,0]]*6)))
plt.scatter(np.where(seeds!=0)[1],np.where(seeds!=0)[0],s=40,c=color_scatter)


plt.savefig("shaded_region.pdf")

plt.show()

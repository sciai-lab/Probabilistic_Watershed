#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:45:55 2019

@author: enfita
Computation of the segmentation of the figure 1 of the NeurIPS 2019 paper
"Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning".

We use the Random Walker of sklearn to compute the Probabilistic Watershed due to its equivalence.

The boundary_p_maps (Probability boundary maps) were given by the authors of "End-to-End Learned Random Walkerfor
 Seeded Image Segmentation" [L. Cerrone et al., 2019]
"""

import numpy as np
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.morphology import watershed
from scipy.special import expit
from scipy.stats import entropy
import vigra 

def random_label_cmap(n=2**16):
   import matplotlib
   import colorsys
   # cols = np.random.rand(n,3)
   # cols = np.random.uniform(0.1,1.0,(n,3))
   h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
   cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
   cols[0] = 0
   return matplotlib.colors.ListedColormap(cols)

raw=np.load('raw_CREMI25.npy')
p_maps=np.load('boundary_p_maps_CREMI25.npy')
gt=np.load('gt_CREMI25.npy')
seeds=np.load('seeds_CREMI25.npy')




#%%







labels = random_walker(expit(p_maps), seeds, beta=1e3, mode='bf',return_full_prob=True)

seg_RW=np.argmax(labels,axis=0)
seg_WS, max_label = vigra.analysis.watersheds(-expit(p_maps), seeds=seeds.astype(np.uint32))
#seg_WS=  watershed(-expit(p_maps), markers=seeds.astype(np.uint32))




#%%
plots=(2,3)
num_colors=2**6
rmap=random_label_cmap(num_colors)
rmap.colors=np.load('colormap_64.npy')
rmap.colors[int((24-17)/(62-17)*num_colors)]=np.array([0,00,147])/255
plt.axis('off')

plt.subplot(plots[0],plots[1],1)
plt.imshow(raw,cmap='gray')
plt.title('Raw image')
plt.subplot(plots[0],plots[1],2)
plt.imshow(p_maps,cmap='gray')
plt.title('Boundary probability\nmap + Seeds')
color_scatter=np.concatenate((np.array([[1,0,0]]),np.array([[1,1,0]]*7),np.array([[127.5,214.5,247]])/255,np.array([[1,1,0]]*6)))
plt.scatter(np.where(seeds!=0)[1],np.where(seeds!=0)[0],s=40,c=color_scatter)

plt.axis('off')

plt.subplot(plots[0],plots[1],4)
plt.imshow(seg_WS,cmap=rmap)
plt.title('Segmentation WS')
plt.axis('off')


plt.subplot(plots[0],plots[1],5)
plt.imshow(seg_RW,cmap=rmap)
plt.title('Segmentation \nProbabilistic WS')

plt.axis('off')



plt.subplot(plots[0],plots[1],6)
plt.imshow(entropy(labels),cmap='gray')
plt.title('Entropy Prediction\n Probabilistic WS')

plt.axis('off')


plt.show()


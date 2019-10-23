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

save=True

raw=np.load('raw_CREMI25.npy')
p_maps=np.load('boundary_p_maps_CREMI25.npy')
gt=np.load('gt_CREMI25.npy')
seeds=np.load('seeds_CREMI25.npy')

if save:
	seeds=seeds[0:270]
	gt=gt[0:270]
	raw=raw[0:270]
	p_maps=p_maps[0:270]


#%%
###remove_small_seeds
seeds[137,250]=0
seeds[161,177]=0


#%%







labels = random_walker(expit(p_maps), seeds, beta=1e3, mode='bf',return_full_prob=True)

seg_RW=np.argmax(labels,axis=0)
seg_WS, max_label = vigra.analysis.watersheds(-expit(p_maps), seeds=seeds.astype(np.uint32))
#seg_WS=  watershed(-expit(p_maps), markers=seeds.astype(np.uint32))

for i,j in zip(np.where(seeds!=0)[0],np.where(seeds!=0)[1]):
	seg_RW[np.where(seg_RW==seg_RW[i,j])] = seg_WS[i,j]

k=1
for i,j in zip(np.where(seeds!=0)[0],np.where(seeds!=0)[1]):
	seg_RW[np.where(seg_RW==seg_RW[i,j])] = k
	seg_WS[np.where(seg_WS==seg_WS[i,j])] = k
	k=k+1




#%%%
seeds_coord=np.where(seeds!=0)[1],np.where(seeds!=0)[0]

num_seeds=len(seeds_coord[1])

plots=(2,3)
num_colors=num_seeds#2**6
rmap=random_label_cmap(num_colors)
rmap.colors=np.load('colormap_13.npy')
#np.save('colormap_13.npy',rmap.colors)
#rmap.colors[int((24-17)/(62-17)*num_colors)]=np.array([[127.5,214.5,247]])/255#np.array([0,0,147])/255
rmap.colors[0]=np.array([1,0,0])#17 min label and 62 max label
rmap.colors[8]=np.array([0,0,147])/255
plt.axis('off')

plt.subplot(plots[0],plots[1],1)
plt.imshow(raw,cmap='gray')
seeds_rest=np.array(range(len(seeds_coord[0])))
seeds_red_cyan=[0,8]
seeds_rest=np.delete(seeds_rest,seeds_red_cyan)
plt.scatter(seeds_coord[0][seeds_rest],seeds_coord[1][seeds_rest],s=100,c=rmap.colors[seeds_rest],alpha=1,edgecolor='black', linewidth='3')
plt.scatter(seeds_coord[0][np.array(seeds_red_cyan)],seeds_coord[1][np.array(seeds_red_cyan)],s=150,marker='v'
			,c=rmap.colors[seeds_red_cyan],edgecolor='black', linewidth='3',)
plt.title('Raw image')
plt.subplot(plots[0],plots[1],2)
plt.imshow(p_maps,cmap='gray')
plt.title('Boundary probability\nmap + Seeds')


plt.axis('off')

plt.subplot(plots[0],plots[1],4)
plt.imshow(seg_WS,cmap=rmap)
plt.title('Segmentation WS')
plt.axis('off')


plt.subplot(plots[0],plots[1],5)
plt.imshow(seg_RW,cmap=rmap)
plt.title('Segmentation \nProbabilistic WS')
plt.colorbar()
plt.axis('off')



plt.subplot(plots[0],plots[1],6)
plt.imshow(entropy(labels),cmap='gray')
plt.title('Entropy Prediction\n Probabilistic WS')


plt.axis('off')


plt.show()



#%%
if save:
	z=25
	size_scatter=3000
	plt.figure(figsize=(20,22))
	plt.imshow(raw,cmap='gray')
	seeds_yellow=np.array(range(len(seeds_coord[0])))
	seeds_red_cyan=[0,8]
	seeds_yellow=np.delete(seeds_yellow,seeds_red_cyan)
	plt.scatter(seeds_coord[0][seeds_yellow],seeds_coord[1][seeds_yellow],s=size_scatter,c=rmap.colors[seeds_yellow],alpha=1,
				edgecolor='black', linewidth='10',)
	plt.scatter(seeds_coord[0][np.array(seeds_red_cyan)],seeds_coord[1][np.array(seeds_red_cyan)],s=size_scatter+2000,marker='v'
				,c=rmap.colors[seeds_red_cyan],edgecolor='black', linewidth='10', )
	
	plt.axis('off')
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.savefig('raw_crop'+str(25)+'.png',dpi=200)
	
	
	
	plt.figure(figsize=(20,22))
	plt.imshow(seg_WS,cmap=rmap)
	plt.axis('off')
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.savefig('WS_CROP_'+str(z),dpi=200)
	
	plt.figure(figsize=(20,22))
	plt.imshow(seg_RW,cmap=rmap)
	plt.axis('off')
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.savefig('RW_CROP_'+str(z),dpi=200)
	
	
	
	plt.figure(figsize=(20,22))
	plt.imshow(entropy(labels),cmap='gray')
	plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
				hspace = 0, wspace = 0)
	plt.margins(0,0)
	#plt.colorbar()
	plt.axis('off')
	plt.savefig('entropy_'+str(z),dpi=200)
	
	plt.close('all')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:07:05 2019

@author: enfita
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils_figure2 import *

edges_3=[ 1 , 0.2 , 0.95 , 0.5 , 0.8 , 0.3 , 0.9 , 0.6 , 0.65 , 0.4 , 0.85 , 0.7]

A=np.zeros((9,9))
A[0,1]=edges_3[10]
A[0,3]=edges_3[6]

#A[0,4]=0.2
A[1,2]=edges_3[11]
A[1,4]=edges_3[8]
#A[2,4]=0.9
A[2,5]=edges_3[9]
A[3,4]=edges_3[5]
A[3,6]=edges_3[1]
A[4,5]=edges_3[7]#Special
#A[4,6]=0.2
A[4,7]=edges_3[3]
#A[4,8]=0.2
A[5,8]=edges_3[4]#Special
A[6,7]=edges_3[0]
A[7,8]=edges_3[2]
A=np.exp(np.round(np.log(A),2))
################################


A=-A-A.T
L_d=np.diag(-np.sum(A,axis=1))+A

seeds=[(0,0),(2,2)]
n=3

final_P=probs2(L_d,seeds,n)

for i in range(len(seeds)):
	print('Probability being connected to seed ', i+1)
	plt.figure()
	plt.title('Probability being connected to seed '+str(i+1))
	plt.imshow(np.dot(final_P[:,:,i].T,np.array([[0,0,1],[0,1,0],[1,0,0]])).T,cmap='bwr')
	plt.colorbar()
	plt.show()
plt.figure()
plt.title('Segmentation')
plt.imshow(np.dot(np.argmax(final_P,axis=2).T,np.array([[0,0,1],[0,1,0],[1,0,0]])).T,cmap='jet')
#plt.imshow(np.argmax(final_P,axis=2),cmap='jet')
plt.show()
print(final_P)
#%%


forests=np.load('Forest_paralel_3_grid_(0,2)_(2,0).npy')

weight_forests=utils_hystogram.compute_weight_forests(forests,edges_3)
prob_forest=weight_forests/np.sum(weight_forests)
cost_forests=-np.log(weight_forests)
order=np.argsort(cost_forests)

skip_ord=[]
i=0



w,h=matplotlib.figure.figaspect(0.4)
fig, ax = plt.subplots(figsize=(w,h))


ax.plot(cost_forests[order], prob_forest[order],color='red')
ax.bar(cost_forests[order], prob_forest[order],color='lime',width=0.01)


plt.rcParams.update({'font.size': 12})
plt.ylabel("Probability")
plt.xlabel("Cost")


#ax.set_xticks(list(ax.get_xticks())[1:] + [1.5])
#plt.savefig('Probability_cost_new_weights.png', dpi=400)
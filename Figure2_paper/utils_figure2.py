#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:22 2019

@author: enfita
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg 

def delete_row_csr(mat, i):
	if not isinstance(mat, sps.csr_matrix):
		raise ValueError("works only for CSR format -- use .tocsr() first")
	i=int(i)
	n = mat.indptr[i+1] - mat.indptr[i]
	if n > 0:
		mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
		mat.data = mat.data[:-n]
		mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
		mat.indices = mat.indices[:-n]
	mat.indptr[i:-1] = mat.indptr[i+1:]
	mat.indptr[i:] -= n
	mat.indptr = mat.indptr[:-1]
	mat._shape = (mat._shape[0]-1, mat._shape[1])
	

def identity_minus_rows(N, rows):
	#removes the rows indicated in the list rows of the identity matrix of size N
	if np.isscalar(rows):
		rows = [rows]
	J = sps.diags(np.ones(N), 0).tocsr()  # make a diag matrix
	for r in sorted(rows):
		delete_row_csr(J, r)
	return J

def del_seed_laplacian(L,s1,m):
	J=identity_minus_rows(L.shape[0],s1[0]*m+s1[1])
	L_=J*L*J.T
	return L_

def probs2(L_d,seeds,n):
	#Computes probability of being connected to seed1 or seed2. Only 2 seeds
	V=L_d.shape[0]
	m=int(V/n)
	v_s1=int(seeds[0][0]*m+seeds[0][1])
	v_s2=int(seeds[1][0]*m+seeds[1][1])
	rm=0
	if v_s1>v_s2:
		print("Modify order")
		v_aux=v_s1
		v_s1=v_s2
		v_s2=v_aux
		rm=1
		del v_aux
	
	
	L_=del_seed_laplacian(L_d,seeds[rm],m)
	b=np.zeros(V-1)
	b[v_s2-1]=1
	p=scipy.sparse.linalg.spsolve(L_,b) #DIRECT SOLVER
	#p,succsess=scipy.sparse.linalg.cg(L_,b)#ITERATIVE SOLVER
	#print("residual of itereation ",succsess)
	
	p_=np.insert(p, v_s1, 0)
	p_=(p_[v_s2]-p_)/p_[v_s2]
	#p_=(np.max(p_)-p_)/np.max(p_)
	print((p_>=0).all(),(p_<=1).all())
	p_s1=np.reshape(p_,(n,m))
	
	final_P=np.zeros((n,m,2))
	final_P[:,:,0]=p_s1
	final_P[:,:,1]=1-p_s1
	return final_P


def compute_weight_forests(Forests,edge_weights):
	#Returns a list of the weights of the forests
	aux_forests=Forests*edge_weights
	aux_forests[np.where(aux_forests==0)]=1
	aux_forests=aux_forests*(np.array(edge_weights)!=0)
	try:
		values_hyst=np.prod(aux_forests,axis=1)
	except:
		values_hyst=np.prod(aux_forests)
	
	return (values_hyst)
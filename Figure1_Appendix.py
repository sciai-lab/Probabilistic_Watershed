#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:00:30 2019

@author: enfita



Computation of the segmentation of the figure 1 in the Appendix of the NeurIPS 2019 paper
"Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning"
for different mu.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy.sparse as sps
import scipy.sparse.linalg 

s1=0
s2=1
s3=2
s4=3
v1s2=4
v2s2=5
v1s3=6
v2s3=7
v3s3=8
v1s4=9
v2s4=10
v3s4=11
v4s4=12
q=13


V=14
edges_nodes={0:{q,s1},1:{q,v1s2},2:{s2,v1s2},3:{q,v2s2},4:{s2,v2s2},5:{q,v1s3},6:{s3,v1s3},7:{q,v2s3},8:{s3,v2s3},9:{q,v3s3},10:{s3,v3s3},11:{q,v1s4},12:{s4,v1s4},13:{q,v2s4},14:{s4,v2s4},15:{q,v3s4},16:{s4,v3s4},17:{q,v4s4},18:{s4,v4s4}}
num_edges=len(edges_nodes)


mu=1000
cost_s1=8e-4
cost_s2=1e-3
cost_s3=1e-2
cost_s4=1e-1

cost_edges=np.array([cost_s1,cost_s2,0,cost_s2,0,cost_s3,0,cost_s3,0,cost_s3,0,cost_s4,0,cost_s4,0,cost_s4,0,cost_s4,0])

def Adjacency_matrix(edge_weights):
	A = np.zeros((V,V))
	num_edges=len(edge_weights)
	for e in range(num_edges):
		nodes_e= list(edges_nodes[e])
		A[nodes_e[0],nodes_e[1]]=A[nodes_e[1],nodes_e[0]]=edge_weights[e]
	return A
			
def Laplacian(A):
	return np.diag(np.sum(A,axis=0))-A
			

def Laplacian_sps(A):
	diags = A.sum(axis=1)
	D = sps.spdiags(diags.flatten(), [0],A.shape[0],A.shape[1], format='csr')
	
	return D-A


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
	if np.isscalar(rows):
		rows = [rows]
	J = sps.diags(np.ones(N), 0).tocsr()  # make a diag matrix
	for r in sorted(rows,reverse=True):
		delete_row_csr(J, r)
	return J

def del_seed_laplacian(L,seeds):
	J=identity_minus_rows(L.shape[0],seeds)
	L_=J*L*J.T
	return L_

def del_seed_B(B,seeds):
	J=identity_minus_rows(B.shape[0],seeds)
	return J*B
	

def solve_RW_Laplacian(L,seeds):
	seeds.sort()
	L_U=np.delete(np.delete(L,seeds,axis=0),seeds,axis=1)
	probs=np.zeros((len(seeds),L.shape[0]))
	for i in range(len(seeds)):
		B_v=np.delete(L[:,seeds[i]],seeds)
		prob_v=np.linalg.solve(L_U,-B_v)
		for seed in seeds:
			if seed==seeds[i]:
				prob_v=np.insert(prob_v, seed, 1)
			else:
				prob_v=np.insert(prob_v, seed, 0)
		probs[i,:]=prob_v

	return probs

def solve_RW_Laplacian_sps(L,seeds):
	seeds.sort()
	L_U=del_seed_laplacian(L,seeds)
	probs=np.zeros((len(seeds),L.shape[0]))
	for i in range(len(seeds)-1):
		B_v=del_seed_B(L[:,seeds[i]],seeds)
		prob_v=scipy.sparse.linalg.spsolve(L_U,-B_v) #DIRECT SOLVER
		#prob_v,succsess=scipy.sparse.linalg.linalg.cg(L_,b)#ITERATIVE SOLVER
		#print("residual of itereation ",succsess)
		for seed in seeds:
			if seed==seeds[i]:
				prob_v=np.insert(prob_v, seed, 1)
			else:
				prob_v=np.insert(prob_v, seed, 0)
		probs[i,:]=prob_v
		
	probs[-1,:]=1-np.sum(probs,axis=0)
	return probs

def RW_solver(edge_weights,seeds,A=None,return_probs=False):
	if A is None:
		L=Laplacian(Adjacency_matrix(edge_weights))
	elif isinstance(A,sps.dok_matrix):
		L=Laplacian_sps(A)
	else:
		L=Laplacian(A)
	if isinstance(L,sps.csr_matrix):
		probs=solve_RW_Laplacian_sps(L,seeds)
	else:
		probs=solve_RW_Laplacian(L,seeds)
	if return_probs:
		return np.argmax(probs,axis=0).astype(np.int),probs
	
	return np.argmax(probs,axis=0).astype(np.int)


seeds=[s1,s2,s3,s4]
edge_weights=np.exp(-mu*cost_edges)
seg,probs=RW_solver(edge_weights,seeds,return_probs=True)
print(seg)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:50:39 2019

@author: enfita


It computes all the forests indexed by its edges (0/1) separating two seeds in a grid graph.
"""
import numpy as np
import random
import scipy.sparse as sps
from itertools import combinations
import time

class Grid_Graph:
	def __init__(self,n=10,m=None):
		self.dim=(n,n if m == None else m)

		self.V=set()
		for a in range(self.dim[0]):
			for b in range(self.dim[1]):
				self.V.add((a, b))
		
		
	def card_vert(self,):
		return self.dim[0]*self.dim[1]

	def card_edges(self,):
		return self.dim[1]*(self.dim[0]-1)+self.dim[0]*(self.dim[1]-1)


	def v_num_to_coord(self,v):
		j=v%self.dim[0]
		i=int((v-j)/self.dim[0])
		return (i,j)
	def v_coord_to_num(self,v):
		num=int(v[0]*self.dim[0]+v[1])
		return num
	
	def lesseq_than(self,v_1,v_2):
		return self.v_coord_to_num(v_1)<=self.v_coord_to_num(v_2)
	
	def e_coord_to_num(self,e):
		if not self.lesseq_than(e[0],e[1]):
			e=[e[1],e[0]]
		v_1=e[0]
		v_2=e[1]
		
		num_e=2*(v_1[0])*(self.dim[1]-1)+(v_1[0])+2*v_1[1]*(v_1[0]!=self.dim[0]-1)+v_1[1]*(v_1[0]==self.dim[0]-1)
		num_e=num_e if (v_2[0]-v_1[0]==0 or v_1[1]==self.dim[1]-1 or v_1[0]==self.dim[0]-1)  else num_e+1
		return num_e
	
	def get_neigh(self,v):
		if not isinstance(v,tuple):
			v=self.v_num_to_coord(v)
		a,b=v
		neigh=set()
		if a!=0:
			neigh.add((a-1,b))
		if b!=0:
			neigh.add((a,b-1))
		if a!=self.dim[0]-1:
			neigh.add((a+1,b))
		if b!=self.dim[1]-1:
			neigh.add((a,b+1))
		return neigh
	
	def get_neigh_pos(self,v):
		if not isinstance(v,tuple):
			v=self.v_num_to_coord(v)
		a,b=v
		neigh=set()
		if a!=self.dim[0]-1:
			neigh.add((a+1,b))
		if b!=self.dim[1]-1:
			neigh.add((a,b+1))
		return neigh
	


def Polyhedra_forest(G,v_1,v_2):
	#Computes the linear constraints Ax<=b that a forest x  separating seeds v_1 and v_2 must satisfy in a graph G
	
	
	eq_num=pow(2,G.card_vert())-2-G.card_vert()
	unk_num=G.card_edges()
	Ineq=sps.dok_matrix((eq_num,unk_num),dtype=np.float32)
	b=np.zeros((1,eq_num))
	count=0
	count_zero=0
	for size in range(2,G.card_vert()):
		for S in combinations(range(G.card_vert()),size):
			#print('S=',S)
			for s in S:
				#print('s=',s)
				#print('coord_s=',G.v_num_to_coord(s))
				neighs=G.get_neigh_pos(s)
				#print('neighs=',neighs)
				for neigh in neighs:
#					print('neigh=',neigh)
#					print('num_neigh=',G.v_coord_to_num(neigh))
#					print( 'To the matrix',G.v_coord_to_num(neigh) in S)
					if G.v_coord_to_num(neigh) in S:
#						print('coord_e=',G.e_coord_to_num([G.v_num_to_coord(s),neigh]))
						Ineq[count,G.e_coord_to_num([G.v_num_to_coord(s),neigh])]=1
						
			
			if (Ineq[count,:].todense()!=0).any():
				if G.v_coord_to_num(v_1) in S and G.v_coord_to_num(v_2) in S: 
					b[0,count]=size-2
				else:
					b[0,count]=size-1
				count+=1
			else:
				count_zero+=1
		
	
	return Ineq[:-count_zero,:],b[:,:-count_zero]


def Laplacian(n,m=None):
	#Laplacian non-weighted Grid graph of size nxm
	if m==None:
		m=n
	V=n*m
	L = sps.dok_matrix((V,V), dtype=np.float32)
	for v in range(V):
		j=v%m
		i=int(v/(m))
		
		if i!=0:
			L[(i-1)*m+j,v]=L[v,(i-1)*m+j]=-1
			
			L[v,v]=L[v,v]+1
		if i!=(n-1):
			L[(i+1)*m+j,v]=L[v,(i+1)*m+j]=-1
			L[v,v]=L[v,v]+1
		if j!=0:
			L[i*m+j-1,v]=L[v,i*m+j-1]=-1
			L[v,v]=L[v,v]+1
		if j!=(m-1):
			L[i*m+j+1,v]=L[v,i*m+j+1]=-1
			L[v,v]=L[v,v]+1
	return L


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
	for r in sorted(rows):
		delete_row_csr(J, r)
	return J

def del_seed_laplacian(L,s1,m):
	J=identity_minus_rows(L.shape[0],s1[0]*m+s1[1])
	L_=J*L*J.T
	return L_




from multiprocessing import Pool


def check_Constraint(S):
		unk_num=A.shape[1]
		t=np.zeros((1,unk_num), dtype=np.int8)
		t[0,S]=1
		if (A@t[0]<=b).all():
			return t
		else:
			pass
		


def find_forests_comb_edges_par(card_V):
	#Tries all possible combinations of subgraphs of G with |V|-2 edges and checks if its a forest by checking the linear constraints
	unk_num=A.shape[1]

	p=Pool()
	combinations_shuffled=list(combinations(range(unk_num),card_V-2))
	random.shuffle(combinations_shuffled)
	Trees=p.map(check_Constraint,combinations_shuffled)
	
	for tree in Trees:
		if tree is not None:
			if not 'Tree_mat' in locals():
				Tree_mat=tree
			else:
				Tree_mat=np.concatenate((Tree_mat,tree),axis=0)
					
	
	return Tree_mat

	


#%%
n=3
v_1=(0,n-1)
v_2=(n-1,0)
Forest_file='Forest_paralel_'+str(n)+'_grid_('+str(v_1[0])+','+str(v_1[1])+')_('+str(v_2[0])+','+str(v_2[1])+')'

compute_Forests=False
if compute_Forests:
	
	for n in range(4,5):
		v_1=(0,n-1)
		v_2=(n-1,0)
		print(n)
		print(v_1,v_2)
		G=Grid_Graph(n)
		L=Laplacian(n)
		n_forest=int(np.linalg.det(del_seed_laplacian(del_seed_laplacian(L,v_2,3),v_1,3).todense()))#number of possible forests. lemma2.2 paper
		
		A,b=Polyhedra_forest(G,v_1,v_2)

		start_time_par=time.time()
		Forests=find_forests_comb_edges_par(G.card_vert())
		print('Par time = ', time.time()-start_time_par)
		
		np.save(Forest_file,Forests)

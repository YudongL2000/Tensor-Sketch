#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:48:05 2019

@author: zandieh
"""
import numpy as np
from numpy import linalg as LA
import math

# q = 2(#modes), d = 5, 
def TensorInit(d, m, q):
    
    Tree_D = [0 for i in range((q-1).bit_length())]
    Tree_P = [0 for i in range((q-1).bit_length())]
    
    m_=int(m/4)
    q_ = int(q/2)
    for i in range((q-1).bit_length()):
        if i == 0:
            Tree_P[i] = np.random.choice(d, (q_,2,m_))
            Tree_D[i] = np.random.choice((-1,1), (q_,2,d))
        else:
            Tree_P[i] = np.random.choice(2*m_, (q_,2,m_))
            Tree_D[i] = np.random.choice((-1,1), (q_,2,2*m_))
        q_ = int(q_/2)
     
    D = np.random.choice((-1,1), 2*q*m_+1)
    P = np.random.choice(2*q*m_+1, 2*m_)
    
    return Tree_D, Tree_P, D, P
        
def TSRHT(X1, X2, P, D):
    
    Xhat1 = np.fft.fft(X1*D[0,:],axis=1)[:,P[0,:]]
    Xhat2 = np.fft.fft(X2*D[1,:],axis=1)[:,P[1,:]]
    
    Y = np.sqrt(2/P.shape[1])*np.concatenate((Xhat1.real * Xhat2.real, Xhat1.imag * Xhat2.imag), axis=1) 
    
    return Y

def TensorSketch(Tree_D, Tree_P, X):
    n=X.shape[0]
    lgq = len(Tree_D)
    V = [0 for i in range(lgq)]
    E1 = np.concatenate((np.ones((n,1)), np.zeros((n,X.shape[1]-1))),axis=1)
    
    
    for i in range(lgq):
        q = Tree_D[i].shape[0]
        V[i] = np.zeros((q,n,2*Tree_P[i].shape[2]))
        for j in range(q):
            if i == 0:
                V[i][j,:,:] = TSRHT(X, X, Tree_P[i][j,:,:], Tree_D[i][j,:,:])
            else:
                V[i][j,:,:] = TSRHT(V[i-1][2*j,:,:], V[i-1][2*j+1,:,:], Tree_P[i][j,:,:], Tree_D[i][j,:,:])
    
    U = [0 for i in range(2**lgq)]
    U[0] = np.copy(V[lgq-1][0,:,:])
    
    for j in range(1,len(U)):
        p = int((j-1)/2)
        for i in range(lgq):
            if j%(2**(i+1)) == 0 :
                V[i][p,:,:] = np.concatenate((np.ones((n,1)), np.zeros((n,V[i].shape[2]-1))),axis=1)
            else:
                if i == 0:
                    V[i][p,:,:] = TSRHT(X, E1, Tree_P[i][p,:,:], Tree_D[i][p,:,:])
                else:
                    V[i][p,:,:] = TSRHT(V[i-1][2*p,:,:], V[i-1][2*p+1,:,:], Tree_P[i][p,:,:], Tree_D[i][p,:,:])
            p = int(p/2)
        U[j] = np.copy(V[lgq-1][0,:,:])
    
    return U

def OblvFeat(s, q, m, Tree_D, Tree_P, D, P, X):
    
    n = X.shape[0]
    U = TensorSketch(Tree_D, Tree_P, X)
    m = U[0].shape[1]
    
    Z = np.ones((len(D),n))
    
    for i in range(q):
        Z[m*i+1:m*(i+1)+1] = np.sqrt(((2/s)**(i+1))/ math.factorial(i+1)) * U[q-i-1].T
        U[q-i-1]=0
    
    Z = Z*np.exp(-(LA.norm(X, 2, axis=1)**2)/s)
    
    R = np.fft.fft(Z.T*D, axis=1)[:,P]
    
    return np.sqrt(1/len(P))*np.concatenate((R.real, R.imag), axis=1).T
    


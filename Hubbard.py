#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:11:17 2019

@author: agnish
"""
#%%
import numpy as np
aup_d = np.array([[0,0,0,0],
                  [0,0,0,0],
                  [1,0,0,0],
                  [0,-1,0,0]], dtype = float)
aup = aup_d.transpose()
adn_d = np.array([[0,0,0,0],
                  [1,0,0,0],
                  [0,0,0,0],
                  [0,0,1,0]], dtype = float)
adn = adn_d.transpose()

O = np.zeros((4,4), dtype = float)
I = np.eye(4, dtype = float)
Z = np.array([[1,   0,  0,  0],
              [0,  -1,  0,  0],
              [0,   0, -1,  0],
              [0,   0,  0,  1]])

aupZd = np.dot(aup_d, Z)
aupZ  = np.dot(Z, aup)
adnZd = np.dot(adn_d, Z)
adnZ  = np.dot(Z, adn)

nup = np.dot(aup_d, aup)
ndn = np.dot(adn_d, adn)
nd = np.dot(nup,ndn)
nt = nup + ndn


#%%
def form_op(L, t, U, V):
    Op1 = np.array([[U*nd, t*aup_d, t*adn_d, t*aup, t*adn, V*nt, I]])
    OpL = np.array([[I],
                    [aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [nt],
                    [U*nd]])
    Op_mid = np.array([[I,      O,      O,      O,      O,      O,       O],
                       [aupZ,   O,      O,      O,      O,      O,       O],
                       [adnZ,   O,      O,      O,      O,      O,       O],
                       [aupZd,  O,      O,      O,      O,      O,       O],
                       [adnZd,  O,      O,      O,      O,      O,       O],
                       [nt,     O,      O,      O,      O,      O,       O],
                       [U*nd, t*aup_d, t*adn_d, t*aup, t*adn,   V*nt,    I]])
    
    MPO = [Op_mid]*L
    MPO[0] = Op1
    MPO[L-1] = OpL
    return MPO

#%%
def operator_expansion(Op1, Op2):
    import numpy as np
    import scipy.linalg as spl
    sh1 = np.shape(Op1)
    sh2 = np.shape(Op2)
    assert sh1[1] == sh2[0]
    expnd_Op = np.zeros((sh1[0], sh2[1], sh1[2]*sh2[2], sh1[3]*sh2[3]), 
                        dtype = complex)
    for i in range(0, sh1[0], 1):
        for k in range(0, sh2[1], 1):
            for j in range(0, sh1[1], 1):
                expnd_Op[i,k] = expnd_Op[i,k] + spl.kron(Op1[i,j],Op2[j,k])
    
    return expnd_Op


#%%
#       i            k                    i  k
#       |            |                    |  |
#  a---Op1---b  b---OpL---c   ---->   a---TEOp----c
#       |            |                    |  |
#       j            l                    j  l
#  
#  TEOp = TEOp[abijkl]
#
#
import numpy as np
import scipy.linalg as sl

def form_TEOp(L, d, tstep, t, U, V):
    Op1 = np.array([[U*nd, t*aup_d, t*adn_d, t*aup, t*adn, V*nt, I]])
    OpL = np.array([[I],
                    [aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [nt],
                    [U*nd]])
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (d*d,d*d))
    TEOp = sl.expm(-tstep*1j*M)
    
    W = np.transpose(np.reshape(TEOp, (d,d,d,d)), (0,2,1,3))
    W = np.reshape(W, (d*d,d*d))
    u,s,v = np.linalg.svd(W)
    s = np.diag(s)
    u = np.dot(u, np.sqrt(s))
    v = np.dot(np.sqrt(s), v)
    U1 = np.transpose(np.reshape(u, (d,d,1,d*d), order = 'C'), (2,3,0,1))
    U2 = np.reshape(v, (d*d,1,d,d), order = 'C')
    
    odd_op = [np.array([[np.eye(d)]])]*L
    for i in range(0,L/2,1):
        odd_op[2*i] = U1
        odd_op[2*i+1] = U2
    even_op = [np.array([[np.eye(d)]])]*L
    for i in range(1,(L+1)/2, 1):
        even_op[2*i-1] = U1
        even_op[2*i] = U2
    
    return odd_op, even_op

#%%
#       i            k                    i  k
#       |            |                    |  |
#  a---Op1---b  b---OpL---c   ---->   a---TEOp----c
#       |            |                    |  |
#       j            l                    j  l
#  
#  TEOp = TEOp[acijkl]
#
def form_TEOp_new(d, tstep, t, U, V):
    Op1 = np.array([[U*nd, t*aup_d, t*adn_d, t*aup, t*adn, V*nt, I]])
    OpL = np.array([[I],
                    [aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [nt],
                    [U*nd]])
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (d*d,d*d))
    TEOp = sl.expm(-tstep*1j*M)
    
    W = np.transpose(np.reshape(TEOp, (d,d,d,d)), (0,2,1,3))
    return W


#%%
#------------------------------------------------------------------------------
# This function forms the transfer operator for 2-sites
# The basis size, d has to be provided along with time step and the
# magnitude of the transfer integral, T
# Unlike TEOp, this function returns the time evolution for only 2 sites,
# not for the entire chain.
#------------------------------------------------------------------------------

def form_transfer(d, tstep, T):
    Op1 = np.array([[O, T*aup_d, T*adn_d, T*aup, T*adn, I]])
    OpL = np.array([[I],
                    [aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [O]])
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (d*d,d*d))
    TEOp = sl.expm(-tstep*1j*M)
    
    W = np.transpose(np.reshape(TEOp, (d,d,d,d)), (0,2,1,3))
    W = np.reshape(W, (d*d,d*d))
    u,s,v = np.linalg.svd(W)
    s = np.diag(s)
    u = np.dot(u, np.sqrt(s))
    v = np.dot(np.sqrt(s), v)
    U1 = np.transpose(np.reshape(u, (d,d,1,d*d), order = 'C'), (2,3,0,1))
    U2 = np.reshape(v, (d*d,1,d,d), order = 'C')
    transf_op = [U1, U2]
    return transf_op

#%%
def form_transfer_new(d, tstep, T):
    Op1 = np.array([[O, T*aup_d, T*adn_d, T*aup, T*adn, I]])
    OpL = np.array([[I],
                    [aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [O]])
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (d*d,d*d))
    TEOp = sl.expm(-tstep*1j*M)
    W = np.transpose(np.reshape(TEOp, (d,d,d,d)), (0,2,1,3))   
    return W

#%%
#       i            k                    i  k
#       |            |                    |  |
#  a---Op1---b  b---OpL---c   ---->   a---TEOp----c
#       |            |                    |  |
#       j            l                    j  l
#  
#  TEOp = TEOp[acijkl]
#
def form_TEOp_two_site(d, tstep, t, V):
    Op1 = np.array([[t*aup_d, t*adn_d, t*aup, t*adn, V*nt]])
    OpL = np.array([[aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [nt]])
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (d*d,d*d))
    TEOp = sl.expm(-tstep*1j*M)
    
    W = np.transpose(np.reshape(TEOp, (d,d,d,d)), (0,2,1,3))
    return W


#%%
def form_TEOp_three_site(d, tstep, t, V):
    Op1 = np.array([[t*aup_d, t*adn_d, t*aup, t*adn, V*nt]])
    Op2 = np.array([[I, O, O, O, O],
                    [O, I, O, O, O],
                    [O, O, I, O, O],
                    [O, O, O, I, O],
                    [O, O, O, O, I]])
    Op3 = np.array([[aupZ],
                    [adnZ],
                    [aupZd],
                    [adnZd],
                    [nt]])
    M = np.einsum('abij,bckl->acikjl', Op1, Op2)
    M = np.einsum('acikjl,cdmn->ikmjln', M, Op3)
    M = np.reshape(M, (d*d*d,d*d*d))
    TEOp = sl.expm(-tstep*1j*M)
    W = np.transpose(np.reshape(TEOp, (d,d,d,d,d,d)), (0,3,1,4,2,5))
    return W



#%%
def form_TEOp_one_site(d, tstep, U):
    Op = np.array([[0,0,0,0],
                   [0,0,0,0],
                   [0,0,0,0],
                   [0,0,0,U]])
    W = sl.expm(-tstep*1j*Op)
    return W




#%%
#------------------------------------------------------------------------------
# The following piece of code if for time evolving any MPS using an MPO for a
# small time step. The state as MPS, the MPO for 2 sites Tp and the maximum
# allowed bond dimensions, D are to be provided.
# First the odd bonds are evolved. Then the even bonds are evolved.
# Tp1 is the single site evolution operator
# Tp2 is the double(two) site evolution operator
#------------------------------------------------------------------------------

import Contractions as ct
def time_evolution(MPS, Tp2, D, Tp1=None):

    M = MPS[:]
    L = len(M)
    
    #--------------------------------------------------------------------------
    # Evolution of odd-bonds
    #--------------------------------------------------------------------------
    i = 0
    while(i<(L/2)):
        phi = np.einsum('iab,jbc->ijac', M[2*i],M[2*i+1])
        psi = np.einsum('ijac, risj->rsac', phi, Tp2)
        M1, M2, s1, err1 = ct.svd_and_truncate('one_MPS', D, psi)
        M[2*i] = M1
        if(i<(L/2-1)):
            psi = np.einsum('ab,ibc,jcd->ijad', s1, M2, M[2*i+2])
            M2, M3, s2, err2 = ct.svd_and_truncate('one_MPS', D, psi)
            M[2*i+2] = np.einsum('ab,ibc->iac', s2, M3)
            M[2*i+1] = M2
        else:
            M[2*i+1] = np.einsum('ab,ibc->iac', s1, M2)
        i=i+1
    
    #--------------------------------------------------------------------------
    # Evolution of even bonds
    #--------------------------------------------------------------------------
    i = 0
    while(i<(L+1)/2-1):
        phi = np.einsum('iab,jbc->ijac', M[2*i+1],M[2*i+2])
        psi = np.einsum('ijac, risj->rsac', phi, Tp2)
        M1, M2, s1, err1 = ct.svd_and_truncate('one_MPS', D, psi)
        M[2*i+1] = M1
        if(i<(L+1)/2-2):
            psi = np.einsum('ab,ibc,jcd->ijad', s1, M2, M[2*i+3])
            M2, M3, s2, err2 = ct.svd_and_truncate('one_MPS', D, psi)
            M[2*i+3] = np.einsum('ab,ibc->iac', s2, M3)
            M[2*i+2] = M2
        else:
            M[2*i+2] = np.einsum('ab,ibc->iac', s1, M2)
        i=i+1
    
    if(isinstance(Tp1, np.ndarray)==True):
        #--------------------------------------------------------------------------
        # Evolution of single sites
        #--------------------------------------------------------------------------
        TE_ss = [Tp1]*L
        M = ct.MPO_on_MPS(M, TE_ss)
    
    return M

    

#%%
#------------------------------------------------------------------------------
# Time Evolving Block Decimation
# MPS HAS TO BE IN VIDAL FORM i.e.
# MPS = [S0, M0, S1, M1, S2, M2...., Sn, Mn, Sn+1]
# Tp denotes the Operator, it can either be a 2-site operator or a single site
# operator. The code first checks what kind of operatot it is and then proceeds.
# i denotes the site (the first site in case of two-site operator) where we need
# the time evolution.
# D denotes the maximum bond dimension we want
#------------------------------------------------------------------------------

def TEBDi(MPS, D, Tp, i):
    M = MPS[:]
    if(Tp.ndim == 2):       #--> Single Site operator
        A = np.einsum('ab,ibc,cd->iad', M[2*i], M[2*i+1], M[2*i+2], optimize = 'optimal')
        At = np.einsum('ji,ipq->jpq', Tp, A)
        At = np.einsum('ab,ibc,cd->iad', np.diag(1/np.diag(M[2*i])), At, np.diag(1/np.diag(M[2*i+2])), optimize = 'optimal')
        M[2*i+1] = At
    
    elif(Tp.ndim == 4):     #--> Two site operator
        assert (2*i+3)<len(MPS)
        A1 = np.einsum('ab,ibc,cd->iad', M[2*i], M[2*i+1], M[2*i+2])
        A = np.einsum('iad,jde,ef->ijaf', A1, M[2*i+3], M[2*i+4])
        At = np.einsum('kilj,ijac->klac', Tp, A)
        M1, M2, s, err = ct.svd_and_truncate('one_MPS', D, At)
        M[2*i+1] = np.einsum('ab,ibc->iac', np.diag(1/np.diag(M[2*i])), M1, optimize = 'optimal')
        M[2*i+2] = s
        M[2*i+3] = np.einsum('iab,bc->iac', M2, np.diag(1/np.diag(M[2*i+4])), optimize = 'optimal')
    
    elif(Tp.ndim == 6):       #--> 3-Site operator
        assert (2*i+5)<len(MPS)
        A = np.einsum('ab,ibc,cd->iad', M[2*i], M[2*i+1], M[2*i+2], optimize = 'optimal')
        A = np.einsum('iad,jde,ef-> ijaf', A, M[2*i+3], M[2*i+4], optimize = 'optimal')
        A = np.einsum('ijaf,kfg,gh->ijkah', A, M[2*i+5], M[2*i+6], optimize = 'optimal')
        At = np.einsum('ijkah, limjnk -> lmnah', A, Tp)
        V = ct.svd_truncate_all(At, D, 4)
        M[2*i+1] = np.einsum('ab,ibc->iac', np.diag(1/np.diag(M[2*i])), V[0], optimize = 'optimal')
        M[2*i+2] = V[1]
        M[2*i+3] = V[2]
        M[2*i+4] = V[3]
        M[2*i+5] = np.einsum('iab,bc->iac', V[4], np.diag(1/np.diag(M[2*i+6])), optimize = 'optimal')
    
    else:
        raise Exception('Enter a Valid Operator')
    
    return M
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:56:31 2019

@author: agnish
"""


#%%
#------------------------------------------------------------------------------
# Computes <phi|A|psi>
#  |b    b--phi--c    c|
#  |        a|         |
#  |        a|         |
#  1---p p---A---q q---1
#  |         i|        |
#  |         i|        |
#  |j    j--psi--j    j|
#------------------------------------------------------------------------------

import numpy as np
def expectation(psi, Op, phi):
    L = len(psi)
    assert len(psi) == len(Op) == len(phi)
    Lt = np.array([[[1]]])
    i=0
    while(i<L):
        Lt = np.einsum('pqr,api,qjab,brk -> ijk', Lt,phi[i],Op[i],psi[i], optimize = 'optimal')
        i=i+1
    Expec = np.einsum('ijk, ijk', Lt, np.array([[[1]]]))
    return Expec

#%%
#------------------------------------------------------------------------------
# Contracting MPS-MPO-MPS expressions
# p      p----M1----i     i      M1 is of the form M1(a,p,i)
# |           a|          |      Lt is of the form Lt(p,q,r)
# |           a|          |
# Lt---q q----O-----j j---Rt     O is of the form O(q,j,a,b)
# |           b|          |      
# |           b|          |      Rt is of the form Rt(i,j,k)
# r      r----M2----k     k      M2 is of the form M2(b,r,k)
# 
# This function takes input 5 arguments:
# MU, MD, O, direction and j
# j determines the site around which contraction is carried out, e.g
# M = [M[0], M[1], M[2], M[3], M[5]], site 2 would mean around M[2]
# Direction determines the direction of contraction
# Contraction from 'left' direction is carried out as follows:
# Lt_new = np.einsum('pqr,api,qjab,brk -> ijk', Lt,M1,O,M2)
# Contraction from 'right' direction is carried out as follows:
# Rt_new = np.einsum('ijk,api,qjab,brk -> pqr', Rt,M1,O,M2)
#------------------------------------------------------------------------------
def LR_contract_MPS_MPO_MPS(M1, M2, Op, j, direction):    
    L = len(M1)
    assert len(M1)==len(M2)==len(Op)
    if(direction == 'left'):
        Lt = np.array([[[1]]])
        i=0
        while(i<j):
            Lt = np.einsum('pqr,api,qjab,brk -> ijk', Lt,M1[i],Op[i],M2[i], optimize = 'optimal')
            i=i+1
        return Lt
    
    elif(direction == 'right'):
        Rt = np.array([[[1]]])
        i=L-1
        while(i>j):
            Rt = np.einsum('ijk,api,qjab,brk -> pqr', Rt,M1[i],Op[i],M2[i], optimize = 'optimal')
            i=i-1
        return Rt
    
    else:
        print('Enter a valid Direction')
        return None


#%%
#-----------------------------------------------------------------------------
# Contracting matrices on the right and on the left of a given site(index)
# We basically need to form L1,L2, R1, R2 from the given MPS and a given
# element j around which contractions are done
# j determines the site around which contraction is carried out, e.g
# M = [M[0], M[1], M[2], M[3], M[5]], site 2 would mean around M[2]
# Site indexing starts with 0
# Example:
# MPS1 = [M1[0], M1[1], M1[2], M1[3], M1[4], M1[5]]
# MPS2 = [M2[0], M2[1], M2[2], M2[3], M2[4], M2[5]]
# Contrating around site 2 would mean Contracting M1[0], M1[1] and M2[0], M2[1]
# to generate Lt 
# and then contracting M1[3], M1[4], M1[5] and M2[3], M2[4], M2[5] to form Rt
# p  p----M1----i  i      M1 is of the form M1(a,p,i)
# |       |        |      Lt is of the form Lt(p,r)
# |       |a       |
# Lt               Rt     
# |       |a       |      
# |       |        |      Rt is of the form Rt(i,k)
# r  r----M2----k  k      M2 is of the form M2(a,r,k)
# Contraction from 'left' leads to:
# Lt_new = np.einsum('pr,api,ark -> ik', Lt,M1,M2)
# Contraction from 'right' leads to:
# Rt_new = np.einsum('ik,api,ark -> pr', Rt,M1,M2)
#----------------------------------------------------------------------------- 
def LR_contract_MPS_MPS(M1, M2, j, direction):
    L = len(M1)
    assert len(M1)==len(M2)
    
    if(direction == 'right'):
        i=L-1
        Rt=np.array([[1]])
        while(i>j):
            Rt = np.einsum('ik,api,ark -> pr',Rt,M1[i],M2[i], optimize = 'optimal')
            i=i-1
        return Rt
    
    elif(direction == 'left'):
        i=0
        Lt = np.array([[1]])
        while(i<j):
            Lt = np.einsum('pr,api,ark -> ik',Lt,M1[i],M2[i], optimize = 'optimal')
            i=i+1
        return Lt
    
    else:
        print('Enter a valid Direction')
        return None

#%%
#------------------------------------------------------------------------------
# Finding overlap between 2 MPS: psi and phi
# To find <phi|psi>, we first convert phi to its complex conjugate phi_dagger
# Then starting from left we do the 'optimal' contraction
#  i  i--phi_d--j           j
#  |       |                |
#  |       |                |
#  |       k                |
#  Lt                  =  Lt_new
#  |       k                |
#  |       |                |
#  |       |                |
#  m  m---psi---n           n 
#------------------------------------------------------------------------------

def overlap(psi,phi):
    L = len(psi)
    assert len(psi)==len(phi)
    z = np.shape(psi[0])[1]
    phi_d = [0]*L
    # Forming phi_dagger
    for i in range(0,L,1):
        phi_d[i] = np.matrix.conjugate(phi[i])
    
    i = 0
    Lt = np.eye(z)
    while(i<L):
        Lt = np.einsum('im, kij, kmn -> jn', Lt, phi_d[i], psi[i], optimize = 'optimal')
        i=i+1
    
    ovrlp = np.trace(Lt)
    return ovrlp

#%%
#------------------------------------------------------------------------------
# Converts a given matrix in a MPS into a canonical form
#------------------------------------------------------------------------------
def make_canonical(M, direction):
    if(direction == 'left'):
        sh = np.shape(M)
        psi = np.reshape(M,(sh[0]*sh[1],sh[2]))
        u,s,v = np.linalg.svd(psi, full_matrices=False)
        shu = np.shape(u)
        A = np.reshape(u,(sh[0], int(shu[0]/sh[0]), shu[1]))
        s = np.diag(s)
        X = np.dot(s,v)
    elif(direction == 'right'):
        M = np.transpose(M, (0,2,1))
        A, X = make_canonical(M, 'left')
        A = np.transpose(A, (0,2,1))
        X = np.transpose(X, (1,0))
    else:
        raise Exception('Enter a valid direction')
    
    return A, X


#%%
#------------------------------------------------------------------------------
# This code takes a MPS and converst it into the mixed canonical form around a
# given site l or a given bond l
# In the former case, it returns another MPS
# In the latter case it returns an MPS along with the singular values at the 
# specified bond
# Input arguments are the MPS and the site l and which kind of form is required
# Example of site:
# Say MPS = [M1, M2, M3, M4, M5]
#               B1--M1--B2--M2--B3--M3--B4--M4--B5--M5--B6
# Site index:    ---0--- ---1--- ---2--- ---3--- ---4---
# Bond index:   0--- ---1--- ---2--- ---3--- ---4--- ---5
#------------------------------------------------------------------------------
def MC_around_bond_or_site(M, l, which):
    L = len(M)
    psi_mixed = [0]*L
    #Left canonical Matrices
    i=0
    Xl = np.eye(1)
    while(((i<l)and(which=='bond')) or ((i<l)and(which=='site'))):
        N = np.einsum('ab,ibc-> iac', Xl, M[i], optimize = 'optimal')
        psi_mixed[i], Xl = make_canonical(N, 'left')
        i=i+1
    
    # Right Canonical matrices
    i=L-1
    Xr = np.eye(1)
    while(((i>=l)and(which=='bond')) or ((i>l)and(which=='site'))):
        N = np.einsum('iab,bc->iac', M[i], Xr, optimize = 'optimal')
        psi_mixed[i], Xr = make_canonical(N, 'right')
        i=i-1
    
    
    if(which=='bond'):
        S = np.einsum('ij,jk->ik', Xl,Xr)
        u,s,v = np.linalg.svd(S, full_matrices=False)
        s = np.diag(s)
        psi_mixed[l-1] = np.einsum('iab,bc->iac', psi_mixed[l-1], u, optimize = 'optimal')
        psi_mixed[l] = np.einsum('ab,ibc->iac', v, psi_mixed[l], optimize = 'optimal')
        return psi_mixed, s
    elif(which == 'site'):
        psi_mixed[l] = np.einsum('ab,ibc,cd -> iad', Xl, M[l], Xr)
        return psi_mixed
    else:
        raise Exception('Enter either site or bond')
    


#%%
#------------------------------------------------------------------------------
# The following function operates a given MPS with an MPO and returns the new 
# state. The new state ovisously has greater bond dimensions which need to be
# compressed subsequently using MPS_compression
# Op|psi> = |phi>
#      i                i
#      |                |
#   a--W--b     =   e--phi--f
#      |
#      j
#      j
#      |
#  c--psi--d
#
# phi is given by:
# phi = np.einsum('abij,jcd -> iacbd', W, psi)
# sh = np.shape(phi)
# phi = np.reshape(phi, (sh[0], sh[1]*sh[2], sh[3]*sh[4]))
#------------------------------------------------------------------------------
def MPO_on_MPS(psi, Op):
    L = len(psi)
    assert len(psi) == len(Op)
    
    phi = [0]*L
    for i in range(0,L,1):
        phi[i] = np.einsum('abij,jcd -> iacbd', Op[i], psi[i], optimize = 'optimal')
        sh = np.shape(phi[i])
        phi[i] = np.reshape(phi[i], (sh[0], sh[1]*sh[2], sh[3]*sh[4]))

    return phi

#%%
def MPO_on_MPO(Op1, Op2):
    L = len(Op1)
    assert len(Op1) == len(Op2)
    
    W = [0]*L
    for i in range(0,L,1):
        W[i] = np.einsum('abij,cdjk -> ikacbd', Op1[i], Op2[i], optimize = 'optimal')
        sh = np.shape(W[i])
        W[i] = np.reshape(W[i], (sh[0], sh[1], sh[2]*sh[3], sh[4]*sh[5]))
        W[i] = np.einsum('ikab->abik', W[i])

    return W

#%%
#------------------------------------------------------------------------------
# This function takes either 1 element with 2 physical indices or 2 elements 
# with 1 physical index of an MPS and does a svd and truncate 
# operation on them. It takes the elements and the maximum bond dimension size
# as input. It returns the new elements as well as the truncated singular 
# singular values and error in truncation.
# 1 element with 2 physical indices looks like thefollowing:
#     i  j
#     |  |      =  M1[ijac]  ->  phi[iacj]
#  a---M1---c
# 2 elements with 1 physical index look like the following:
#     i         j
#     |         |
#  a--M1--b  b--M2--c
#------------------------------------------------------------------------------

def svd_and_truncate(form, D, M, M2=None):
    if(form == 'one_MPS' or form == 'None'): 
        phi = np.transpose(M, (0,2,3,1))
        sh = np.shape(phi)
        psi = np.reshape(phi, (sh[0]*sh[1], sh[2]*sh[3]))
        u,s,v = np.linalg.svd(psi, full_matrices = False)
        #----------------------------------------------------------------------
        # Finding truncation length. 
        # D1 denotes the length beyong which the singular values are less than 
        # 1e-3 and D denotes the maximum bond dimension. Truncation length tl is
        # taken to be the minimum among D1 and D
        #----------------------------------------------------------------------
        D1 = len(s[s>1e-8])
        tl = min(D, D1)
        #----------------------------------------------------------------------
        if(tl>len(s)):
            tl = len(s)
        err = np.sum(s[tl:]*s[tl:])/np.sum(s*s)
        s = np.sqrt(np.sum(s*s)/np.sum(s[:tl]*s[:tl]))*np.diag(s[:tl])
        u = u[:, :tl]
        shu = np.shape(u)
        v = v[:tl, :]
        shv = np.shape(v)
        sh1 = (sh[0], int(shu[0]/sh[0]), shu[1])
        sh2 = (shv[0], int(shv[1]/sh[3]), sh[3])
        M1 = np.reshape(u, sh1)
        M2 = np.transpose(np.reshape(v, sh2), (2,0,1))
        return M1, M2, s, err
    elif(form == 'two_MPS'):
        sh1 = np.shape(M)
        sh2 = np.shape(M2)
        assert sh1[2] == sh2[1]
        psi = np.einsum('iab,jbc->ijac', M, M2, optimize = 'optimal')
        M1, M2, s, err = svd_and_truncate('one_MPS', D, psi)
        return M1, M2, s, err




#%%
#------------------------------------------------------------------------------
# The following function converts a state, St to a MPS given the basis size d 
# and the maximum bond dimensions D and the length L. The MPS which is returned 
# is in Left canonical form.
#------------------------------------------------------------------------------
def form_MPS(St, d, D, L): 
    A = [0]*L
    dim = np.flip(np.power(d, np.arange(0, L, 1, dtype = int)), axis=0)
    s = np.array([[1.]])
    shs = np.shape(s)
    for i in range(0,L,1):
        psi = np.reshape(St, (shs[0]*d, dim[i]))
        u,s,v = np.linalg.svd(psi, full_matrices=False)
        #----------------------------------------------------------------------
        # Finding truncation length. 
        # D1 denotes the length beyong which the singular values are less than 
        # 1e-3 and D denotes the maximum bond dimension. Truncation length tl is
        # taken to be the minimum among D1 and D
        #----------------------------------------------------------------------
        D1 = len(s[s>1e-7])
        tl = min(D, D1)
        #----------------------------------------------------------------------
        s = np.sqrt(np.sum(s*s)/np.sum(s[0:tl]*s[0:tl]))*s[0:tl]
        s = np.diag(s)
        u = u[:, 0:tl]
        shu = np.shape(u)
        A[i] = np.transpose(np.reshape(u, (int(shu[0]/d), d,  shu[1])), (1,0,2))
        v = v[0:tl, :]
        St = np.dot(s, v)
        shs = np.shape(s)
    
    A[L-1] = St*A[L-1]  # Multipying the last Matrix in the MPS with the
                        # normalization factor
    return A

#%%
#------------------------------------------------------------------------------
# The following function converts a state, St to a MPS given the basis size d 
# and the maximum bond dimensions D and the length L. The MPS which is returned 
# is in Left canonical form.
#------------------------------------------------------------------------------
def form_vidal_MPS(St, d, D, L): 
    A = [0]*L
    svals = []
    dim = np.flip(np.power(d, np.arange(0, L, 1, dtype = int)), axis=0)
    s = np.array([[1.]])
    svals.append(s)
    shs = np.shape(s)
    for i in range(0,L,1):
        psi = np.reshape(St, (shs[0]*d, dim[i]))
        u,s,v = np.linalg.svd(psi, full_matrices=False)
        #----------------------------------------------------------------------
        # Finding truncation length. 
        # D1 denotes the length beyong which the singular values are less than 
        # 1e-3 and D denotes the maximum bond dimension. Truncation length tl is 
        # taken to be the minimum among D1 and D
        #----------------------------------------------------------------------
        D1 = len(s[s>1e-8])
        tl = min(D, D1)
        #----------------------------------------------------------------------
        s = np.sqrt(np.sum(s*s)/np.sum(s[0:tl]*s[0:tl]))*s[0:tl]
        s = np.diag(s)
        svals.append(s)
        u = u[:, 0:tl]
        shu = np.shape(u)
        A[i] = np.transpose(np.reshape(u, (int(shu[0]/d), d,  shu[1])), (1,0,2))
        v = v[0:tl, :]
        St = np.dot(s, v)
        shs = np.shape(s)
    
    vform = [0]*(2*L+1)
    vform[2*L] = svals[len(svals)-1]
    for i in range(0,L,1):
        vform[2*i] = svals[i]
        vform[2*i+1] = np.einsum('ab,ibc->iac', np.diag(1/np.diag(svals[i])), A[i], optimize = 'optimal')
    return vform
    
    
#%%
#------------------------------------------------------------------------------
# Convert Vidal to Left Canonical MPS
#------------------------------------------------------------------------------
    
def vidal_to_LCM(vform):
    L = int(len(vform)/2)
    M = [0]*L
    for i in range(0,L,1):
        M[i] = np.einsum('ab,ibc->iac', vform[2*i], vform[2*i+1], optimize = 'optimal')
    
    M[L-1] = np.einsum('iab,bc->iac', M[L-1], vform[2*L], optimize = 'optimal')
    return M

#%%
#------------------------------------------------------------------------------
# Convert Left Canonical form to Vidal form
#------------------------------------------------------------------------------

def LCM_to_vidal(MPS):
    M1 = MPS[:]
    L = len(M1)
    l = 2*L+1
    vfm = [0]*l
    u = np.array([[1.]])
    s = np.array([[1.]])
    vfm[2*L] = s
    for i in range(L-1, -1, -1):
        M1[i] = np.einsum('iab,bc,cd->iad', M1[i], u, s, optimize = 'optimal')
        shp = np.shape(M1[i])
        psi = np.reshape(np.transpose(M1[i], (1,0,2)), (shp[1], shp[0]*shp[2]))
        u,s,v = np.linalg.svd(psi, full_matrices=False)
        s = np.diag(s)
        shv = np.shape(v)
        v = np.reshape(v, (shv[0], shp[0], int(shv[1]/shp[0])))
        v = np.transpose(v, (1,0,2))
        if(i==0):
            v = np.einsum('ab,bc,icd->iad', u,s,v)
        vfm[2*i+1] = np.einsum('iab,bc->iac', v, np.diag(1/np.diag(vfm[2*i+2])), optimize = 'optimal')
        vfm[2*i] = s
    return vfm

#%%
#------------------------------------------------------------------------------
# Swapping indices of an MPS in Vidal form
#------------------------------------------------------------------------------

def swap_indices_vidal(vform, i, D):
    M = vform[:]
    assert (2*i+3)<len(M)
#    D = np.shape(M[2*i+2])[0]
    A1 = np.einsum('ab,ibc,cd->iad', M[2*i], M[2*i+1], M[2*i+2], optimize = 'optimal')
    A = (-1)*np.einsum('iad,jde,ef->jiaf', A1, M[2*i+3], M[2*i+4], optimize = 'optimal')
    M1, M2, s, err = svd_and_truncate('one_MPS', D, A)
    M[2*i+1] = np.einsum('ab,ibc->iac', np.diag(1/np.diag(M[2*i])), M1, optimize = 'optimal')
    M[2*i+2] = s
    M[2*i+3] = np.einsum('iab,bc->iac', M2, np.diag(1/np.diag(M[2*i+4])), optimize = 'optimal')
    
    return M

#%%
#------------------------------------------------------------------------------
# Permute indices from left of an MPS in Vidal form.
# e.g. M = 012345 Permute 2,4 gives-> 013425
#------------------------------------------------------------------------------
def permute_indices_vidal_left(vform, i, j, D):
    M = vform[:]
    for a in range(i,j,1):
        M = swap_indices_vidal(M, a, D)
    return M
#------------------------------------------------------------------------------
# Permute indices from left of an MPS in Vidal form.
# e.g. M = 012345 Permute 2,4 gives-> 014235
#------------------------------------------------------------------------------
def permute_indices_vidal_right(vform, i, j, D):
    M = vform[:]
    for a in range(j-1,i-1,-1):
        M = swap_indices_vidal(M, a, D)
    return M

#%%
#------------------------------------------------------------------------------
# the following function takes a MPS in non-Vidal form and reverses it.
# Say the MPS, M is provided to us
#        i        j        k        l
#        |        |        |        |
# M = a--M1--b b--M2--c c--M3--d d--M4--e 
# The reversed MPS will look like:
#        l        k        j        i
#        |        |        |        |
# M = e--M4--d d--M3--c c--M2--b b--M1--a 
#------------------------------------------------------------------------------
def reverse_MPS_or_MPO(M, form = None):
    L = len(M)
    M_rev = [0]*L
    for i in range(0,L,1):
        if(form == None or form == 'MPS'):
            M_rev[i] = np.transpose(M[i], (0,2,1))
        elif(form == 'vidal'):
            if(i%2 == 0):
                M_rev[i] = M[i]
            else:
                M_rev[i] = np.transpose(M[i], (0,2,1))
        elif(form == 'MPO'):
            M_rev[i] = np.transpose(M[i], (1,0,2,3))
        else:
            raise Exception('Enter either MPS or MPO')
    M_rev = M_rev[::-1]
    return M_rev


#%%
#------------------------------------------------------------------------------
# The following function swaps two adjacent indices of a MPS
# Input include the MPS, M and the indices to be swapped i (and i+1)
# The process can be illustrated as :
#     i        j                i  j               j  i
#     |        |       ->       |  |      ->       |  |
#  a--M1--b b--M2--c        a--M_comb--c       a--M_comb--c
#
#          j         i
# ->       |         |
#      a--M1'--b b--M2'--c
#   
# The original canonical form of the matrix also needs to be supplied if the
# MPS is in a given canonical form
#------------------------------------------------------------------------------

def swap_indices(MPS, i, form = None):
    M = MPS[:]
    M1 = M[i]
    D = np.shape(M1)[2]
    M2 = M[i+1]
    M_comb = np.einsum('iab,jbc -> jaic', M1, M2, optimize = 'optimal')
    sh = np.shape(M_comb)
    psi = np.reshape(M_comb, (sh[0]*sh[1], sh[2]*sh[3]))
    if(form == 'left' or form == None):
        u, s, v = np.linalg.svd(psi, full_matrices = False)
        if(len(s)>D):
            s = np.sqrt(np.sum(s*s)/np.sum(s[:D]*s[:D]))*s[:D]
            u = u[:, :D]
            v = v[:D, :]
        s = np.diag(s)
        shs = np.shape(s)
        A1 = u
        A2 = np.dot(s,v)
        M[i] = np.reshape(A1, (sh[0], sh[1], shs[0]))
        M[i+1] = np.transpose(np.reshape(A2, (shs[0], sh[2], sh[3])), (1,0,2))
    elif(form == 'right'):
        M_comb = np.einsum('jaic -> icja', M_comb)
        sh = np.shape(M_comb)
        psi = np.reshape(M_comb, (sh[0]*sh[1], sh[2]*sh[3]))
        u, s, v = np.linalg.svd(psi, full_matrices = False)
        s = np.diag(s)
        shs = np.shape(s)
        A1 = u
        A2 = np.dot(s,v)
        M[i+1] = np.transpose(np.reshape(A1, (sh[0], sh[1], shs[0])), (0,2,1))
        #M[i] = np.transpose(np.transpose(np.reshape(A2, (shs[0], sh[2], sh[3])), (1,0,2)), (0,2,1))
        M[i] = np.transpose(np.reshape(A2, (shs[0], sh[2], sh[3])), (1,2,0))
    else:
        raise Exception('Enter a valid form')
    
    return M

#%%
#------------------------------------------------------------------------------
# The following function is used to permute multiple indices    
# Say we have a MPS given by M = [ M0, M1, M2, M3, M4] and we want to permute
# M1 to M4. The final MPS would look something like the following:
# M = [ M0, M2, M3, M4, M1]
# It can be formed from the following steps:
# [01234] -> [02134] -> [02314] -> [02341]
# MPS is the input MPS, i and j refer to the lower and higher index
# The original canonical form of the matrix also needs to be supplied if the
# MPS is in a given canonical form
#------------------------------------------------------------------------------

def permute_indices_left(MPS, i, j, form = None):
    M = MPS[:]
    for a in range(i,j,1):
        M = swap_indices(M, a, form)
    
    return M

#------------------------------------------------------------------------------
# Just like the previous function it also permutes indices but starting from
# right. e.g. M = 012345 permute 2,4 would result in 014235.

def permute_indices_right(MPS, i, j, form = None):
    M = MPS[:]
    for a in range(j-1,i-1,-1):
        M = swap_indices(M, a, form)
    
    return M


#%%
#------------------------------------------------------------------------------
# The following function is used to compute paprtial overlaps. e.g.
# Say we have a MPS given by, M = M0--M1--M2--M4--M5--M6
# We can write in the product state basis of two systems A and B such that,
# M = A0--A1--A2--X--B0--B1--B2, where A's are in left canonical form, B's are
# in right canonical form and X contains the singular values
# Say we want to find the probability of occurence of a particular state, 
# C = C0--C1--C2 in the left half of the system, i.e. in A
# In order to find this probability, a partial overlap is carried out between
# A0--A1--A2--X-- and C
#            C0--C1--C2
# Overlap =  |   |   |
#            A0--A1--A2--X--
# The probability that C occurs in A is given by np.sum(overlap*overlap)
# Input parameters include the original MPS, C and the direction
# Direction determines if we want the overlap with system A(left) or B(right).
#------------------------------------------------------------------------------

def partial_overlap(MPS, C, direction):
    if(direction == 'left'):
        l = len(C)
        MB = MC_around_bond_or_site(MPS, l, 'bond')
        M = MB[0]
        X = MB[1]
        M[l-1] = np.einsum('iab,bc -> iac', M[l-1], X, optimize = 'optimal')
        A = M[:l]
        C_d = [0]*l
        for i in range(0,l,1):
            C_d[i] = np.matrix.conjugate(C[i])
        i = 0
        Lt = np.array([[1]])
        while(i<l):
            Lt = np.einsum('im, kij, kmn -> jn', Lt, C_d[i], A[i], optimize = 'optimal')
            i=i+1
        P = np.sum(Lt*Lt)
        return P
    elif(direction == 'right'):
        M = reverse_MPS_or_MPO(MPS, form='MPS')
        C = reverse_MPS_or_MPO(C, form='MPS')
        P = partial_overlap(M, C, 'left')
        return P
    else:
        raise Exception('Enter a valid direction')
        


#%%
#------------------------------------------------------------------------------
# Given the occupancies of the sites, form a direct product state MPS 
# e.g. say for 4 sites we have the occupancies as 00011011, we simply form the 
# direct product MPS with all bond dimensions 1, the MPS is formed in the 
# Vidal form
#------------------------------------------------------------------------------

def form_dpMPS(x):
    l = len(bin(x))-2
    M =  [0]*(2*int((l+1)/2)+1)
    M[0] = np.array([[1.]])
    for i in range(0, int((l+1)/2), 1):
        a = x-((x>>2)<<2)
        St = np.zeros((4,1), dtype=float)
        St[a,0] = 1
        M[2*i+1] = np.reshape(St, (4,1,1))
        M[2*i+2] = np.array([[1.]])
        x = x>>2
    return M


#%%
#------------------------------------------------------------------------------
# Form a random MPS with the given physical indices d, the maximum bond
# dimension D and the length of the Chain
#------------------------------------------------------------------------------

def random_MPS(L, d, D):
    if(L%2!=0):
       L1 = (L+1)/2
       BD1 = np.arange(0,L1,1)
       BD2 = np.arange(L1-1,-1,-1)
    else:
       L1 = L/2
       BD1 = np.arange(0,L1+1,1)
       BD2 = np.arange(L1-1,-1,-1)
       
    BD = np.concatenate((BD1,BD2))
    BD = np.power(d,BD)
    for i in range(0,len(BD), 1):
        BD[i] = BD[i]*(BD[i]<=D) + D*(BD[i]>D)   #---> Bond Dimensions
    
    M = [0]*L
    for i in range(0,L,1):
        M[i] = np.random.random((d, BD[i], BD[i+1]))
    
    norm = overlap(M,M)
    M[0] = np.sqrt(1/norm)*M[0]
    return M


#%%
#------------------------------------------------------------------------------
# Return a state from a given MPS (non-Vidal form)
#------------------------------------------------------------------------------

def St_from_MPS(MPS):
    M = MPS[:]
    L = len(M)
    for i in range(1, L, 1):
        St = np.einsum('iab,jbc->ijac', M[i-1], M[i], optimize = 'optimal')
        sh = np.shape(St)
        M[i] = np.reshape(St, (sh[0]*sh[1], sh[2], sh[3]))
        
    St = M[L-1]
    return St


#%%
#------------------------------------------------------------------------------
# This function takes a State of the form 
#     i    j    k    l    m       s    t
#     |    |    |    |    |...... |    |
#   a------------------------------------b
# as M[ijklm.....stab] and first converts it to M[aijklm.....stb]    
# Then it performs a svd and truncate on the entire state to return a 
# vidal MPS of the form:
#     i                     j                              t
#     |                     |                              |
#  a--M1--a1 a1--S1--a2 a2--M2--a3 a3--S2--a4 ........ an--Mn--b
# Note that the even indices now contain the matrices and the odd indices contain
# the singular values unlike the vidal MPS's from earlier. The length of the 
# resulting vidal MPS is 2*L-1.
#------------------------------------------------------------------------------

def svd_truncate_all(MPS, D, d):
    M = MPS[:]
    L = M.ndim - 2
    #--------------------------------------------------------------------------
    #Transposing the axes of M to change it's form
    #--------------------------------------------------------------------------
    idx = [L]
    for i in range(0,L,1):
        idx.append(i)
    idx.append(L+1)
    sh = tuple(idx)
    M = np.transpose(M, sh)
    #--------------------------------------------------------------------------
    psi = M
    vidal = [0]*(2*L-1)
    for i in range(0, L, 1):
        sh = np.shape(psi)
        psi = np.reshape(psi, (sh[0]*d, int(psi.size/(sh[0]*d))))
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        #----------------------------------------------------------------------
        # Finding truncation length. 
        # D1 denotes the length beyong which the singular values are less than 
        # 1e-3 and D denotes the maximum bond dimension. Truncation length tl is
        # taken to be the minimum among D1 and D
        #----------------------------------------------------------------------
        D1 = len(s[s>1e-8])
        tl = min(D, D1)
        #----------------------------------------------------------------------
        if(tl>len(s)):
            tl = len(s)
        s = np.sqrt(np.sum(s*s)/np.sum(s[:tl]*s[:tl]))*np.diag(s[:tl])
        u = u[:, :tl]
        shu = np.shape(u)
        v = v[:tl, :]
        psi = np.dot(s, v)
        u = np.reshape(u, (sh[0], int(shu[0]/sh[0]), shu[1]))
        u = np.transpose(u, (1,0,2))
        
        if(i==(L-1)):
            u = np.einsum('iab,bc->iac', u,psi)
        
        if(i>0):
            sinv = np.diag(1/np.diag(vidal[2*i-1]))
            u = np.einsum('ab,ibc->iac', sinv, u)
            
        if(i<L-1):
            vidal[2*i+1] = s    #--> Appending the singular values to vidal
        
        vidal[2*i] = u          #--> Appending the Gamma-tensors to vidal
    
    return vidal


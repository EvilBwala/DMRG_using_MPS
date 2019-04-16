#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:10:02 2019

@author: agnish
"""

##%%
##------------------------------------------------------------------------------
## Doing a local compression of a MPS
##------------------------------------------------------------------------------
#import numpy as np
#import scipy.linalg as sp
#import Contractions as Ct
#
#def local_MPS_compression(M, D):
#    L = len(M)
#    BD = [0]*(L+1)
#    for i in range(0,L,1):
#        BD[i] = np.shape(M[i])[1]
#    BD[L] = np.shape(M[L-1])[2]
#    for i in range(len(BD)):
#        BD[i] = BD[i]*(BD[i]<=D) + D*(BD[i]>D)   #---> Bond Dimensions
#    
#    X = np.array([[1]])
#    for i in range(0,L,1):
#        M[i] = np.einsum('ab,ibc->iac', X, M[i])
#        sh = np.shape(M[i])
#        psi = np.reshape(M[i], (sh[0]*sh[1], sh[2]))
#        u,s,v = np.linalg.qr(psi)
#        M[i] = u
#        


#%%
#------------------------------------------------------------------------------
# Compressing an MPS to one having maximum bond dimensions=D
# First we need to find the transpose and complex conjugate of a MPS
#      a        Transpose and        b----M_d----c
#      |       Complex Conjugate           |
#      |       ----------------->          |
# b----M----c                              a
#  M is of the form M(a,b,c)  ; M_d is of the form M+(a,b,c)
#  M_d = np.matrix.conjugate(M) 
# Now create a random MPS_t with appropriate bond dimensions
# We need to minimize the Frobenius norm of the difference between M and M_temp
# In order to minimize w.r.t. site i, we compute the following quantities:
#  a               e           i               m
#  |               |           |               |
#  |               |           |               |
# LtA      d      RtA    =    LtB      l      RtB
#  |       |       |           |       |       |
#  |       |       |           |       |       |
#  b  b---M_t---c  c           j   j---M---k   k
# LtA and RtA are the contractions from left and right around the site i,
# where L1,R1 contain M_t_d and L2,R2 contain M_t
# LtB and RtB are the contractions from left and right around the site i,
# where L1,R1 contain M_t_d and L2,R2 contain M
# The eqn can be written as: O[aebc]M_t[dbc] = P[lim]
# This can be written as a matrix equation:
# O[ae,bc]M_t[d][bc] = P[l][im]
# Therefore we can find an optimal value for M_t[d][bc] by solving this
# After every iteration, check for error and compare with previous error.
# Terminate the program when there is no improvement in error.
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as sp
import Contractions as Ct
def MPS_compression(M, d, D, max_iter=None, tol=None):
    #M              ---> Given input
    #D              ---> Maximum Bond dimensions
    #d              ---> Physical index dimensions
    #tol            ---> Error Tolerance
    #max_iter       ---> Maximum number of iterations
    L = len(M)
    it1 = np.arange(0,L,1)
    it2 = np.arange(L-2,-1,-1)
    it = np.concatenate((it1,it2))      #---> For assigning sites
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
    
    
    M_t = [0]*L
    M_t_d = [0]*L
    for i in range(0,L,1):
        M_t[i] = np.random.random((d, BD[i], BD[i+1]))
        M_t_d[i] = np.matrix.conjugate(M_t[i])
    
    N1 = np.sqrt(Ct.overlap(M,M))
    N2 = np.sqrt(Ct.overlap(M_t,M_t))
    M[0] = M[0]/N1
    M_t[0] = M_t[0]/N2
    M_t_d[0] = M_t_d[0]/N2
    
    err = abs(1 - Ct.overlap(M,M_t) - Ct.overlap(M_t,M) + Ct.overlap(M_t,M_t))
    i = 0
    
    while(err>tol):
        j = it[i%(2*L-1)]     #---> j denotes the site which has to be optimized
        
        
        # Compute LtA, RtA, LtA, RtB here
        LtA = Ct.LR_contract_MPS_MPS(M_t_d, M_t, j, 'left')
        RtA = Ct.LR_contract_MPS_MPS(M_t_d, M_t, j, 'right')
        LtB = Ct.LR_contract_MPS_MPS(M_t_d, M, j, 'left')
        RtB = Ct.LR_contract_MPS_MPS(M_t_d, M, j, 'right')
        
        Op = np.einsum('ab,ec -> aebc', LtA, RtA)    #--> Operator Op
        shO = np.shape(Op)                           #--> Shape of operator Op
        Op = np.reshape(Op, (shO[0]*shO[1],shO[2]*shO[3]))
        P = np.einsum('ij,ljk,mk -> lim', LtB, M[j], RtB)
        shP = np.shape(P)
        P = np.reshape(P, (shP[0],shP[1]*shP[2]))
        for k in range(0,shP[0],1):
            V = sp.solve(Op,P[k,:])
            M_t[j][k,:,:] = V.reshape(shP[1], shP[2])
        
        M_t_d[j] = np.matrix.conjugate(M_t[j])
        err = abs(1 - Ct.overlap(M,M_t) - Ct.overlap(M_t,M) + Ct.overlap(M_t,M_t))
        
        i=i+1
    return M_t

#%%
#------------------------------------------------------------------------------
# Compressing an MPS to one having maximum bond dimensions=D
# First we need to find the transpose and complex conjugate of a MPS
#      a        Transpose and        b----M_d----c
#      |       Complex Conjugate           |
#      |       ----------------->          |
# b----M----c                              a
#  M is of the form M(a,b,c)  ; M_d is of the form M+(a,b,c)
#  M_d = np.matrix.conjugate(M) 
# Now create a random MPS_t with appropriate bond dimensions
# We need to minimize the Frobenius norm of the difference between M and M_temp
# In order to minimize w.r.t. site i, we compute the following quantities:
#  a               e           i               m
#  |               |           |               |
#  |               |           |               |
# LtA      d      RtA    =    LtB      l      RtB
#  |       |       |           |       |       |
#  |       |       |           |       |       |
#  b  b---M_t---c  c           j   j---M---k   k
# LtA and RtA are the contractions from left and right around the site i,
# where L1,R1 contain M_t_d and L2,R2 contain M_t
# LtB and RtB are the contractions from left and right around the site i,
# where L1,R1 contain M_t_d and L2,R2 contain M
# The eqn can be written as: O[aebc]M_t[dbc] = P[lim]
# This can be written as a matrix equation:
# O[ae,bc]M_t[d][bc] = P[l][im]
# Therefore we can find an optimal value for M_t[d][bc] by solving this
# After every iteration, check for error and compare with previous error.
# Terminate the program when there is no improvement in error.
# p  p--M_t_d--i  i          
# |       |       |       
# |       a       |            
# F               G         
# |       a       |         
# |       |       |         
# r  r----M----k  k             
#------------------------------------------------------------------------------
def Opt_MPS_compression(M, d, D, max_iter = None, tol = None):
    #M              ---> Given input
    #D              ---> Maximum Bond dimensions
    #d              ---> Physical index dimensions
    #tol            ---> Error Tolerance
    #max_iter       ---> Maximum number of iterations
     
#    L = len(M)
#    BD = [0]*(L+1)
#    for i in range(0,L,1):
#        BD[i] = np.shape(M[i])[1]
#    BD[L] = np.shape(M[L-1])[2]
#    for i in range(len(BD)):
#        BD[i] = BD[i]*(BD[i]<=D) + D*(BD[i]>D)   #---> Bond Dimensions
    L = len(M)
    it1 = np.arange(0,L,1)
    it2 = np.arange(L-2,-1,-1)
    it = np.concatenate((it1,it2))      #---> For assigning sites
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
    
    M_t = [0]*L
    M_t_d = [0]*L
    for i in range(0,L,1):
        M_t[i] = np.random.random((d, BD[i], BD[i+1]))
    
    N1 = np.sqrt(Ct.overlap(M,M))       #
    N2 = np.sqrt(Ct.overlap(M_t,M_t))   #
    M[0] = M[0]/N1                      #---> Normalization
    M_t[0] = M_t[0]/N2                  #
    
    M_t = Ct.MC_around_bond_or_site(M_t, 0, 'site')
    M_t_d = [np.matrix.conjugate(i) for i in M_t]
    err = abs(1 - Ct.overlap(M,M_t) - Ct.overlap(M_t,M) + Ct.overlap(M_t,M_t))
    
    F = [0]*L
    G = [0]*L
    F[0] = np.array([[1]])
    for i in range(1,L,1):
        F[i] = np.einsum('pr, ark, api -> ik', F[i-1], M[i-1], M_t_d[i-1])
    G[L-1] = np.array([[1]])
    for i in range(L-2, -1, -1):
        G[i] = np.einsum('ik, ark, api -> pr', G[i+1], M[i+1], M_t_d[i+1])
    
    
    while(err>tol):
        #Swiping from left to right
        i = 0
        while(i<L-1):
            M_t[i] = np.einsum('ij,ljk,mk -> lim', F[i], M[i], G[i])
            M_t[i], X = Ct.make_canonical(M_t[i], 'left')
            M_t[i+1] = np.einsum('ab,ibc -> iac', X, M_t[i+1])
            M_t_d[i] = np.matrix.conjugate(M_t[i])
            M_t_d[i+1] = np.matrix.conjugate(M_t[i+1])
            F[i+1] = np.einsum('pr, ark, api -> ik', F[i], M[i], M_t_d[i])
            i = i+1
        
        #Sweeping from right to left
        while(i<0):
            M_t[i] = np.einsum('ij,ljk,mk -> lim', F[i], M[i], G[i])
            M_t[i], X = Ct.make_canonical(M_t[i], 'right')
            M_t[i-1] = np.einsum('iab,bc -> iac', M[i-1], X)
            M_t_d[i] = np.matrix.conjugate(M_t[i])
            M_t_d[i-1] = np.matrix.conjugate(M_t[i-1])
            G[i-1] = np.einsum('ik, ark, api -> pr', G[i], M[i], M_t_d[i])
            i = i-1
        
        err = abs(1 - Ct.overlap(M,M_t) - Ct.overlap(M_t,M) + Ct.overlap(M_t,M_t))
    return M_t
#%%
#import numpy as np
import time
L = 10

BD = [1,4,16,30,30,30,30,30,16,4,1]
BD = [i*16 for i in BD]
BD[0] = 1
BD[10] = 1
#%%
BD[0] = 1
BD[L] = 1

D = 30
M = [0]*L
for i in range(0,L,1):
    M[i] = np.random.random((4, BD[i], BD[i+1]))
t1 = time.time()
Mo = Opt_MPS_compression(M, 4, D, tol = 1e-4)
t2 = time.time()
t = t2-t1
#%%
#------------------------------------------------------------------------------
# Itertatively finding the Ground State for a Hamiltonian
# This function takes as arguments: \psi> and H
# The Ground State af a Hamiltonian is found by minimizing the quantity,
# <psi\H\psi>/<psi\psi>, this can be done by using a lagrangian multiplier to
# extremize <psi\H\psi> - lambda*<psi\psi>, where lambda is the multiplier.
# This basically boils down to solving the generalized eigenvalue equation:
# Contracting MPS-MPO-MPS expressions
# p                       i               m             v
# |                       |               |             |
# |           a|          |               |             |
# LtA--q q----W-----j j---RtA  - lambda* LtB           RtB  =  0
# |           b|          |               |             |
# |           b|          |               |     x|      |
# r      r----M----k      k               n  n---M---z  z
# This can be written as a generalized eigenvalue equation:
# H[apibrk]v[brk] = lambda*N[ymvxnz]*v[xnz]
# H[api,brk]v[brk] = lambda*N[xmv,xnz]*v[xnz]
# H = np.einsum('pqr, qjab, ijk -> apibrk', LtA, W, RtA)
# shH = np.shape(H)
# H = np.reshape(H, (shH[0]*shH[1]*shH[2], shH[3]*shH[4]*shH[5]))
# The indices x and y are explained below:
# N is a bit weird to build because if we remove M, we are left with 4 indices 
# whereas we need 6 indices, thus we use the identity matrix having dimensions
# d by d
# The expression on the right can now be written as:
# m               v
# |               |
# |      y|       |
# LtB   I(d)     RtB  =  N*v
# |      x|       |
# |      x|       |
# n   n---M---z   z
# N = np.einsum('mn,yx,vz -> ymvxnz', LtB, I(d), RtB)
# shN = np.shape(N)
# N = np.reshape(N, (shN[0]*shN[1]*shN[2], shN[3]*shN[4]*shN[5]))
# shv = np.shape(M)
# v = np.reshape(M, (shv[0]*shv[1]*shv[2]))
# Solving for the lowest lambda gives us the ground state
# Check for convergence in energy, i.e. run this algorithm till
# abs(lambda_new-lambda_old)/lambda_old < tolerance.
#------------------------------------------------------------------------------
import scipy.sparse.linalg as ssl
def GS_search(psi, Op, max_iter, tolerance):
    # Writing a code only for 4 sites
    
    L = len(psi)
    it1 = np.arange(0,L,1)
    it2 = np.arange(L-2,-1,-1)
    it = np.concatenate((it1,it2))      #---> For assigning sites
    d = np.shape(psi[0])[0]         #---> Dimensions of the physical index
    #Op---> Operator(e.g. Hamiltonian) 
    i=0
    psi_d = [np.matrix.conjugate(v) for v in psi]   #---> Forming psi dagger
    I = np.eye(d, dtype = float)
    energy = 0
    err=1
    i = 0
    while(err>tolerance):
        j = it[i%7]
        
        LtA = Ct.LR_contract_MPS_MPO_MPS(psi_d, psi, Op, j, 'left')
        RtA = Ct.LR_contract_MPS_MPO_MPS(psi_d, psi, Op, j, 'right')
        LtB = Ct.LR_contract_MPS_MPS(psi_d, psi, j, 'left')
        RtB = Ct.LR_contract_MPS_MPS(psi_d, psi, j, 'right')
        
        M = psi[j]
        W = Op[j]
        
        H = np.einsum('pqr, qjab, ijk -> apibrk', LtA, W, RtA)
        shH = np.shape(H)
        H = np.reshape(H, (shH[0]*shH[1]*shH[2], shH[3]*shH[4]*shH[5]))
        
        N = np.einsum('mn,yx,vz -> ymvxnz', LtB, I, RtB)
        shN = np.shape(N)
        N = np.reshape(N, (shN[0]*shN[1]*shN[2], shN[3]*shN[4]*shN[5]))
        
        shv = np.shape(M)
        v = np.reshape(M, (shv[0]*shv[1]*shv[2]))
        
        energy_new, eigvec = ssl.eigsh(H, k=1, M=N, which = 'SA', tol=1e-3)
        
        psi[j] = np.reshape(eigvec, (shv[0], shv[1], shv[2]))
        psi_d[j] = np.matrix.conjugate(psi[j])
        
        err = (energy_new-energy)/energy_new
        energy = energy_new
        
        i=i+1
    return energy, psi


#%%
#------------------------------------------------------------------------------
# This function just solves the following OPerator equation to find an 
# optimal MPS
# Contracting MPS-MPO-MPS expressions
# p                       i               m             v
# |                       |               |             |
# |           a|          |               |     y|      |
# LtA--q q----W-----j j---RtA  - lambda* LtB   Id(d)   RtB  =  0
# |           b|          |               |     x|      |
# |           b|          |               |     x|      |
# r      r----M----k      k               n  n---M---z  z
# This can be written as a generalized eigenvalue equation:
# H[apibrk]v[brk] = lambda*N[ymv,xnz]*v[xnz]
# In this case, the MPS is mixed canonical around M and thus N is the Identity
# operator in 6-dimensions.
# H[api,brk]v[brk] = lambda*v[xnz]
# H = np.einsum('pqr, qjab, ijk -> apibrk', LtA, W, RtA)
# shH = np.shape(H)
# H = np.reshape(H, (shH[0]*shH[1]*shH[2], shH[3]*shH[4]*shH[5]))
# shv = np.shape(M)
# v = np.reshape(M, (shv[0]*shv[1]*shv[2]))
# Solving for the lowest lambda gives us the ground state
#------------------------------------------------------------------------------
def mat_opt(M, LtA, RtA, W, tol):  
    H = np.einsum('pqr, qjab, ijk -> apibrk', LtA, W, RtA)
    shH = np.shape(H)
    H = np.reshape(H, (shH[0]*shH[1]*shH[2], shH[3]*shH[4]*shH[5]))
    shv = np.shape(M)
    energy_new, eigvec = ssl.eigsh(H, k=1, which = 'SA', tol = tol)
    psi = np.reshape(eigvec, (shv[0], shv[1], shv[2]))
    return psi, energy_new
        

#%%
#------------------------------------------------------------------------------
# This is the optimized ground search algorithm in which the left and right
# canonical forms of the MPS is made use of.
# It is assumped the the initial input matrix is in Right Canonical form
# In this code, the left and right contractions of the MPS-MPS and MPS-MPO-MPS
# are also stored in a list so that they are not computed everytime. This 
# speeds up the algorithm.
# p                       i               m             v
# |                       |               |             |
# |                       |               |             |
# LtA--q q----W-----j j---RtA  - lambda* LtB           RtB  =  0
# |           b|          |               |             |
# |           b|          |               |     x|      |
# r      r----M----k      k               n  n---M---z  z
# LtA and RtA are computed iteratively from the F and G elements.
# Siince the MPS is in moxed canonical form around the site i, LtB and RtB
# are identity matrices. Using the previous function, the matrix at site i 
# is optimized. Then we update F (while sweeping left to right) and 
# G (while sweeping right to left) and also update the matrices
# of the next site.
#------------------------------------------------------------------------------
import Contractions as ct
def Opt_GS_search(M, Op, tol1, tol2):
#    tol1         #---> Tolerance for mat_opt
#    tol2         #---> Tolerance for energy difference
    L = len(M)
    assert len(M) == len(Op)
    
    psi = ct.MC_around_bond_or_site(M, 0, 'site')
    psi_t = [np.matrix.conjugate(i) for i in psi]
    
    F = [0]*L
    G = [0]*L
    
    F[0] = np.array([[[1]]])
    for i in range(1,L,1):
        F[i] = np.einsum('pqr, brk, qjab, api -> ijk', F[i-1], psi[i-1], Op[i-1], psi_t[i-1])
    
    G[L-1] = np.array([[[1]]])
    for i in range(L-2, -1, -1):
        G[i] = np.einsum('ijk, brk, qjab, api -> pqr', G[i+1], psi[i+1], Op[i+1], psi_t[i+1])
    
    
    energy = 0
    err = 1
    sweep = 0       #---> For counting sweeps
    while(err>tol2):
        i = 0
        # Sweeping left to right
        while(i<L-1):
            psi[i], energy_new = mat_opt(psi[i], F[i], G[i], Op[i], tol1)
            psi[i], X = ct.make_canonical(psi[i], 'left')
            psi[i+1] = np.einsum('ab,ibc -> iac', X, psi[i+1])
            psi_t[i] = np.matrix.conjugate(psi[i])
            psi_t[i+1] = np.matrix.conjugate(psi[i+1])
            F[i+1] = np.einsum('pqr, brk, qjab, api -> ijk', F[i], psi[i], Op[i], psi_t[i])
            i = i+1
        
        # Sweeping right to left
        while(i>0):
            psi[i], energy_new = mat_opt(psi[i], F[i], G[i], Op[i], tol1)
            psi[i], X = ct.make_canonical(psi[i], 'right')
            psi[i-1] = np.einsum('iab, bc -> iac', psi[i-1], X)
            psi_t[i] = np.matrix.conjugate(psi[i])
            psi_t[i-1] = np.matrix.conjugate(psi[i-1])
            G[i-1] = np.einsum('ijk, brk, qjab, api -> pqr', G[i], psi[i], Op[i], psi_t[i])
            i = i-1
        
        Op_sqr = ct.MPO_on_MPO(Op, Op)
        energy_sqr = ct.expectation(psi, Op_sqr, psi)
        err = abs((np.square(energy_new)-energy_sqr)/np.square(energy_new))
#        energy = energy_new
        sweep = sweep + 1
    
    return psi, energy_new

##%%
#M = [0]*6
#D = 10
#BD = [1,4,20,9,20,4,1]
#for i in range(0,6,1):
#    M[i] = np.random.random((4, BD[i], BD[i+1]))
#M_opt = Opt_MPS_compression(M, 4, D, 200, 1e-4)
##%%
#St1 = np.einsum('iab,jbc,kcd,lde,mef,nfg->ijklmn', M[0], M[1], M[2], M[3], M[4], M[5])
#St2 = np.einsum('iab,jbc,kcd,lde,mef,nfg->ijklmn', M_opt[0], M_opt[1], M_opt[2], M_opt[3], M_opt[4], M_opt[5])
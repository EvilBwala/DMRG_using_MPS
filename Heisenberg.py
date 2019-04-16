#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:34:12 2019

@author: agnish
"""

#%%
#------------------------------------------------------------------------------
# This code makes MPO's for various systems
#------------------------------------------------------------------------------
import numpy as np

L = 4                       #---> Length of the MPO i.e. number of sites
d = 2                       #---> Dimensions of the Physical Index
Jz = 1.
J = 1.
h = 0.
I = np.eye(d, dtype=float)
Sz = np.array([[0.5,0],
               [0,-0.5]])
Sp = np.array([[0,1],       #-- > Denotes S+
               [0,0]])
Sm = np.array([[0,0],       #---> Denotes S-
               [1,0]])
O = np.zeros((d,d), dtype = float)

Op1 = np.array([[-h*Sz, (J/2)*Sm, (J/2)*Sp, Jz*Sz, I]])

OpL = np.array([[I],
                [Sp],
                [Sm],
                [Sz],
                [-h*Sz]])

Op_mid = np.array([[I,      O,          O,          O,      O],
                   [Sp,     O,          O,          O,      O],
                   [Sm,     O,          O,          O,      O],
                   [Sz,     O,          O,          O,      O],
                   [-h*Sz,  (J/2)*Sm,   (J/2)*Sp,   Jz*Sz,  I]])

#%%
#Forming the MPO with first site havinf Op1 and last one having OpL
def form_op(L, Jz, J, h):
    d = 2
    I = np.eye(d, dtype=float)
    Sz = np.array([[0.5,0],
                   [0,-0.5]])
    Sp = np.array([[0,1],       #-- > Denotes S+
                   [0,0]])
    Sm = np.array([[0,0],       #---> Denotes S-
                   [1,0]])
    O = np.zeros((d,d), dtype = float)
    
    Op1 = np.array([[-h*Sz, (J/2)*Sm, (J/2)*Sp, Jz*Sz, I]])
    
    OpL = np.array([[I],
                    [Sp],
                    [Sm],
                    [Sz],
                    [-h*Sz]])
    
    Op_mid = np.array([[I,      O,          O,          O,      O],
                       [Sp,     O,          O,          O,      O],
                       [Sm,     O,          O,          O,      O],
                       [Sz,     O,          O,          O,      O],
                       [-h*Sz,  (J/2)*Sm,   (J/2)*Sp,   Jz*Sz,  I]])
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
    M = np.zeros((sh1[0], sh2[1], sh1[2]*sh2[2], sh1[3]*sh2[3]), dtype = complex)
    for i in range(0, sh1[0], 1):
        for k in range(0, sh2[1], 1):
            for j in range(0, sh1[1], 1):
                M[i,k] = M[i,k] + spl.kron(Op1[i,j],Op2[j,k])
    
    return M

#%%
# MPO is the existing MPO
# Op is the new Op to be inserted at site j    
def insert_operator(MPO, Op, j):
    MPO.insert(j, Op)
    return MPO

#%%
#       i            k                    i  k
#       |            |                    |  |
#  a---Op1---b  b---OpL---c   ---->   a---TEOp----b
#       |            |                    |  |
#       j            l                    j  l
#  
#  TEOp = TEOp[abijkl]
#
#
import numpy as np
import scipy.linalg as sl

def form_TEOp(L, d, t):
    sh1 = np.shape(Op1)
    sh2 = np.shape(OpL)
    assert sh1[1] == sh2[0]
    M = np.einsum('abij, bckl -> ikjl', Op1, OpL)
    M = np.reshape(M, (4,4))
    TEOp = sl.expm(t*1j*M)
    
    W = np.reshape(TEOp, (d*d,d*d))
    u,s,v = np.linalg.svd(W)
    s = np.diag(s)
    u = np.dot(u, np.sqrt(s))
    v = np.dot(np.sqrt(s), v)
    U1 = np.einsum('abcd->cdab', np.reshape(u, (d,d,1,d*d), order = 'C'))
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
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:52:27 2019

@author: agnish
"""

#%%
import numpy as np
import Hubbard as hb
import Sparse_Hamiltonian as sh
import scipy.sparse as ss
import Spin_MS_symmetry_adap as smsa

#%%
#------------------------------------------------------------------------------
# Total Parity operator for a system have L sites
#------------------------------------------------------------------------------
def form_Ptot(L):
    P = np.array([[1,0,0,0],
              [0,0,1,0],
              [0,1,0,0],
              [0,0,0,-1]])
    P = ss.csr_matrix(P)
    Ptot = P
    for i in range(1, L, 1):
        Ptot = ss.kron(Ptot,P)
    return Ptot

#%%
#------------------------------------------------------------------------------
# Electron-hole symmetry operator for a system having L sites
#------------------------------------------------------------------------------
def form_Jtot(L):
    J1 = np.array([[0,0,0,-1],
                   [0,1,0,0],   
                   [0,0,1,0],
                   [1,0,0,0]])
    J2 = np.array([[0,0,0,-1],
                   [0,-1,0,0],
                   [0,0,-1,0],
                   [1,0,0,0]])
    J1 = ss.csr_matrix(J1)
    J2 = ss.csr_matrix(J2)
    Jtot = J1
    for i in range(1,L,1):
        if(i%2==0):
            Jtot = ss.kron(Jtot, J1)
        else:
            Jtot = ss.kron(Jtot, J2)
    return Jtot

#%%
#------------------------------------------------------------------------------
# Constructing the C2 operator for a syatem having L sites
#------------------------------------------------------------------------------

def swapBits(x,i,j,n): 
    p1 = 2*i
    p2 = 2*j
    # Move all bits of first 
    # set to rightmost side  
    set1 =  (x >> p1) & ((1<< n) - 1)    
    # Moce all bits of second 
    # set to rightmost side  
    set2 =  (x >> p2) & ((1 << n) - 1)    
    # XOR the two sets  
    xor = (set1 ^ set2)    
    # Put the xor bits back 
    # to their original positions  
    xor = (xor << p1) | (xor << p2)    
      # XOR the 'xor' with the 
      # original number so that the  
      # two sets are swapped 
    result = x ^ xor     
    return result


def orb_occ(x, k):
    return sh.extract_lastkbits(x>>(2*k),1) + sh.extract_lastkbits(x>>(2*k+1),1)

def form_C2(L, e):
    sites = []
    for i in range(0,int(L/2),1):
        sites.append((i,L-1-i))
    stmC2 = ss.lil_matrix((4**L, 4**L), dtype=float)
    stmC2[0,0] = 1
    for i in range(1, 4**L, 1):
        y = i
        for j in sites:
            y = swapBits(y, j[0], j[1], 2)
        itot = e
        iph = 0
        for k in range(0, L, 1):
            nkl = orb_occ(i, k)
            itot = itot-nkl
            iph = iph + nkl*itot
        stmC2[y, i] = (-1)**iph
    stmC2 = stmC2.tocsr()
    return stmC2


#%%
#------------------------------------------------------------------------------
# Finding states in the subspace 1Ag, 1Bu, 3Bu.
# 1Ag --> S=0, ms=0, St.T*stmC2*St = 1
# 1Bu --> S=0, ms=0, St.T*stmC2*St = -1
# 3Bu --> S=1, ms=0, St.T*stmC2*St = -1 
# Inputs: L = number of sites, e = number of electrons (= number of sites),
# space = 
# t, U, V = UV model parameters, c = state we are interested in (0 for ground
# state, 1 for 1st excited state, 2 for 2nd excited state and so on)
# First compute 10 lowest eigenvectors and eigenvalues then check for the 
# ground state and excited states in a given space
#------------------------------------------------------------------------------

def sym_adap_state(L, space, t, U, V, c):
    stmC2 = form_C2(L, L)
    if (space == '1Ag'):
        S = 0; ms = 0; phs = 1;
    elif(space == '1Bu'):
        S = 0; ms = 0; phs = -1;
    elif(space == '3Bu'):
        S = 1; ms = 0; phs = -1;
    elif(space == '3Ag'):
        S = 1; ms = 0; phs = 1;
    else:
        raise Exception("Enter a valid space")
    stmat = smsa.fock_basis_states(L,L,S,ms,t,U,V,10)[0]
    A = ss.csr_matrix(stmat)
    B = (A.transpose()).dot(stmC2.dot(A))
    C = B.diagonal(k=0)
    x = np.where(abs(C-phs)<1e-5)
    idx = x[0][c]
    St = stmat[:,[idx]]
    
    return St, idx

#%%
#-------------------------------------------------------------------------------------------------------------------
# This function is basically the Shiba transform of a given state on a bipartite lattice.
# Every singly occupied state stays unchanged. Every doubly occupied state gets transformed to a vacant state.
# Every vacant state gets transformed to a doubly occupied state.
# The entire wavefunction gets multiplied by a phase which is given by the (-1)^count, where 
# count = number of doubly occupied sites + number of singly ocupied B sites
#-------------------------------------------------------------------------------------------------------------------

def shiba(x, n):
    t = abs(x)
    phase = int(x/t)
    i = 1
    y = 0
    count = 0
    while(i<=n):
        a = sh.extract_lastkbits(t,2)
        if(i%2 == 0)and((a == 1)or(a == 2)): #Checking for singly occupied B-sites of the bipartite lattice
            count = count + 1
        if(a == 3): #Checking for doubly occupied sites
            count = count + 1
        if(a == 1)or(a == 2):
            y = y + (2**(2*(i - 1)))*a
        if(a == 0):
            y = y + (2**(2*(i - 1)))*3
        t = t>>2
        i = i + 1
    transf_state = (-1)**count*y*phase
    return(transf_state)
    
#%%
#-----------------------------------------------------------------------------------------------------------------
# This function is the parity operator for the states. It flips the spin without a change in phase.
# It keeps the empty state as it is and it multiplies the doubly occupied state with a phase of -1
#-----------------------------------------------------------------------------------------------------------------

def parity(x, n): # x = integer representation of the state, n = number of sites
    t = abs(x)
    phase = int(x/t)
    i = 1
    y = 0
    count = 0
    while(i<=n):
        a = sh.extract_lastkbits(t,2)
        if(a == 3):
            count = count + 1
            y = y + (2**(2*(i - 1)))*3
        if(a == 1):
            y = y + (2**(2*(i - 1)))*2
        if(a == 2):
            y = y + (2**(2*(i - 1)))*1
        t = t>>2
        i = i + 1
    transf_state = (-1)**count*y*phase
    return(transf_state)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:52:45 2019

@author: agnish
"""

import Sparse_Hamiltonian as sh
import numpy as np

#%%
#----------------------------------------------------------------------------------------------------------------
# This function takes an array (lst) and an array of the coefficients associated 
# with its elements (lst_coeff) and returns the original array (P), array of 
# absolute elements (A) and the final coefficients (count). It excludes the
# elements whose count is 0.
#----------------------------------------------------------------------------------------------------------------

def P_abs_coeff(lst, lst_coeff):
    P = lst[:]
    P_coeff = lst_coeff[:]
    b=0
    A=[]
    count=[]
    while(b<len(P)):
        j=0
        flag=0 
        #-------------------------------------------------------------------------
        # Flag variable to signal if an element has been encountered before
        # Checking if an element has occured before and updating count
        #-------------------------------------------------------------------------
        
        while(j<len(A)):
            if(abs(P[b])==A[j]):
                count[j]=count[j]+int(P[b]/abs(P[b]))*P_coeff[b]
                flag=1
            j=j+1
        
        #-------------------------------------------------------------------------
        # If the element has not been encountered, append it to the array A
        #-------------------------------------------------------------------------
        
        if(flag==0):
            A.append(abs(P[b]))
            count.append(int(P[b]/abs(P[b]))*P_coeff[b])
        b=b+1
    zero_lst = []
    for i in range(0, len(count), 1):
        if(count[i] == 0):
            zero_lst.append(i)
    
    A = [x for i,x in enumerate(A) if i not in zero_lst]
    count = [x for i,x in enumerate(count) if i not in zero_lst]
    return(A, count)


#%%
#------------------------------------------------------------------------------
# Given the location of the file, it generates the VB basis inside the file
# and stores it in A
#------------------------------------------------------------------------------
def gen_VB_basis_from_file(location):
    f = open(location, 'r')
    lines = f.readlines()
    f.close()
    A = []
    i = 0
    while (i<len(lines)):
        if(lines[i] == '\n'):
            i = i+2
            st = []
            st_coeff = []
            while(lines[i] != '\n'):
                state = int(lines[i][0:21])
                coeff = float(lines[i][21:42])
                st.append(state)
                st_coeff.append(coeff)
                i = i + 1
                if(i==len(lines)):
                    break
            A.append((st, st_coeff))
    return A

#%%
def extract_last2bits(n):
    x=n>>2
    y=x<<2
    return(n-y)

def extract_lastkbits(n,k):
    x=n>>k
    y=x<<k
    return(n-y)

#------------------------------------------------------------------------------
#Finds the next highest integer with a fixed number of 0's and 1's
#------------------------------------------------------------------------------
def next_highest_int(n): 
    c0=0
    c1=0
    c = n
    #Calculating c0
    while(((c&1)==0) and (c!=0)):
        c0=c0+1
        c=c>>1
    #Calculating c1
    while((c&1)==1):
        c1=c1+1
        c=c>>1
    #If there is no bigger number with the same no. of 1's
    if (c0 +c1 == 31 or c0 +c1== 0):
        return -1;
    #Calculating the position of righmost non-trailing 0
    p = c0 + c1;
    n |= (1 << p);
    #Clear all bits to the right of p
    n=n&~((1 << p) - 1);
    #Insert (c1-1) ones on the right.
    n=n|(1 << (c1 - 1)) - 1;
    return n;

#------------------------------------------------------------------------------
# This function retuns the integers for the states in ascending order as well 
# as their occupancies in two separate arrays. n is the number of sites
# and e is the number of electrons
#------------------------------------------------------------------------------


def states_with_MS(n, e, ms): 
    lowest=2**e-1
    z=lowest
    highest=(2**(2*n-e))*lowest
    MS_states=[]
    all_orb_occu=[]
    while(z<=highest):
        x=z
        orb_occu=[]     # stores 0,1,2,3 depending on whether orbital has 0, 
                        # down, up or both electrons
        mstot = 0.0
        
        #----------------------------------------------------------------------
        # Calculating MStotal
        #----------------------------------------------------------------------
        
        while(x>0):
            a=extract_last2bits(x)
            orb_occu.append(a)
            if(a==1):
                mstot=mstot-0.5
            if(a==2):
                mstot=mstot+0.5
            x=x>>2
        if(mstot==ms):
            MS_states.append(z)
            all_orb_occu.append(orb_occu)   # stores orbital occupancy 
                                            # corresponding to a state.
        z=next_highest_int(z)
    return(MS_states, all_orb_occu)

#%%
#------------------------------------------------------------------------------
# x is the binary representation of the state
# S is the Total Spin of the state
#------------------------------------------------------------------------------
def check_legality(x, S):
    y = x
    up_count = 0
    dn_count = 0
    dnc = []
    upc = []
    A = []
    i = 0
    while(y>0):
        z = extract_last2bits(y)
        if(z == 1):
            dn_count = dn_count + 1
            dnc.append(i)
        if(z == 2):
            up_count = up_count + 1
            if(len(dnc) > 0):
                upc.append(i)
        if(len(upc)*len(dnc) != 0):
            d = dnc.pop()
            u = upc.pop()
            if(d < u):    
                A.append([d,u])
        if(up_count-dn_count > 2*S):
            return None
        y = y>>2
        i = i + 1
    return x, A

#%%
#------------------------------------------------------------------------------
# Exchange electrons Exchanges the 01 electron of one site with 10 electron of
# another site. Given the binary representation of a state and the start and 
# end points of the singlet bonds in a list, this function returns all the 
# states possible after the exchange of 01 and 10 electrons.
# Input: x = binary representation of state
#        L = List of the start and end points of singlet bonds obtained from 
#            check_legality function
# Output: lst = list of possible states
#         phs = list of the corresponding phases of the states.
# e.g. x = 2469 and L = [[0,3],[1,2],[4,5]] returns
#       lst = [2469,2457,2406,1701,2394,1689,1638,1626]
#       phs = [1, -1, -1, -1, 1, 1, 1, -1]
#------------------------------------------------------------------------------

from itertools import combinations
from itertools import chain

def modifyBit(x,  p,  b): 
    mask = 1 << p 
    return (x & ~mask) | ((b << p) & mask) 

def genSt(x, L):
    lst = []
    phs = []
    pwset = list(chain.from_iterable(combinations(L, r) for r in range(len(L)+1)))
    for j in range(0, len(pwset), 1):
        A = pwset[j]
        y = x
        for i in range(0,len(A),1):
            B = A[i]
            start = B[0]
            end = B[1]
            y = modifyBit(y, 2*start, 0)
            y = modifyBit(y, 2*start + 1, 1)
            y = modifyBit(y, 2*end, 1)
            y = modifyBit(y, 2*end + 1, 0)
        phase = float((-1)**(len(A))) 
        phs.append(phase)
        lst.append(y)
    return (lst, phs)

#%%
#------------------------------------------------------------------------------
# This function generates the VB basis i.e. the states for the maximum value of 
# S in a given S space. For example if S = 2, it generates the VB states for 
# ms = 2 space inside the S = 2 space. n denotes the number of sites and
# e denotes the number of electrons.
#------------------------------------------------------------------------------
def gen_VB_basis(n, e, S):
    St = states_with_MS(n,e,S)[0]
    A = []
    for i in range(0, len(St), 1):
        if(check_legality(St[i], S) != None):
            x = check_legality(St[i], S)[0]
            L = check_legality(St[i], S)[1]
            A.append(genSt(x, L))
    return A


#%%
#------------------------------------------------------------------------------
# Forming S^- operator
#------------------------------------------------------------------------------
def ms_lowering(x, n):
    temp = x
    B = []
    for i in range(0, n, 1):
        a = sh.extract_last2bits(temp)
        if(a == 2):
            y = x - 2**(2*i)
            B.append(y)
        temp = temp>>2
    return B


#%%
def ms_lowered_VB_basis(A, n):
    L = len(A)
    B = []
    for i in range(0, L, 1):
        l = len(A[i][0])
        msl_st = []
        msl_c = []
        for j in range(0, l, 1):
            state = A[i][0][j]
            c = A[i][1][j]
            ms_lowered = ms_lowering(state, n)
            msl_st = msl_st + ms_lowering(state, n)
            msl_c = msl_c + [c]*len(ms_lowered)
        msl_st, msl_c = P_abs_coeff(msl_st, msl_c)
        B.append((msl_st, msl_c))
    return B


#%%
#------------------------------------------------------------------------------
# Given a list of states in the form of
# [ ([St1], [c1]), 
#   ([St2], [c2]),
#   ([St3], [c3]),
#   ..............]
# this function will form the transformation matrix and also return a dictionary
# which contains the vectors for various states
# Input include n = number of sites, e = number of electrons,
# ms = ms value of the space we are interested in, 
# A = list as specified in the form before
# The function will first compute the integer values of the states in the given
# ms and assign them each a vector, a generic (0,0,1,0,....,0) vector where the 
# position of the 1 varies for each state. In the next step, the transformation
# matrix, stm is formed in the sparse matrix form.
#------------------------------------------------------------------------------

def form_spin_stm(n, e, ms, A):
    st = sh.states_with_MS(n,e,ms)[0]
    #--------------------------------------------------------------------------
    # Creating a dictionary to store the vectors corresponding to integers
    #--------------------------------------------------------------------------
    st_dict = {}
    L = len(st)
    for i in range(0, L, 1):
    #    vec = np.zeros((L,1), dtype = float)
    #    vec[i,0] = 1
        st_dict[st[i]] = i
    
    #--------------------------------------------------------------------------
    # Forming stm
    #--------------------------------------------------------------------------
    import scipy.sparse as ssc
    row = []
    col = []
    data = []
#    stm = np.zeros((len(st), len(A)), dtype = float)
    for i in range(0, len(A), 1):
        l = len(A[i][0])
        norm = np.sqrt(sum([k*k for k in A[i][1]]))
        for j in range(0, l, 1):
            state = A[i][0][j]
            c = A[i][1][j]/norm
            row.append(st_dict[state])
            col.append(i)
            data.append(c)
    
    stm = ssc.coo_matrix((data, (row,col)), shape=(len(st), len(A)))
    stm = stm.tocsr()
    return stm, st_dict

#%%
#------------------------------------------------------------------------------
# The following code stores a Hamiltonian in sparse matrix form. The given
# Hamiltonian is Spin and ms adapted. It then computes thelowest few eigenvalues
# of H and stores the eigenvectors. The eigenvectors are then converted to 
# the Slater basis. Then they are ultimately converted to the full Fock basis.
# n, e, S, ms, t, U, V have their usual meanings
# k = the number of lowest eigenstates and eigenvalues we want to compute
# lst = list of the states we are interested in. It should be less than k
# e.g. k = 5 computes the lowest 5 eigenstates but if we are interested in only
# the 1st, 3rd and 4th eigenstate, lst = [0,2,3]
# Note that the lowest eigenstate is indexed at 0.
#------------------------------------------------------------------------------

import scipy.sparse.linalg as ssl
import scipy.linalg as sl
def fock_basis_states(n, e, S, ms, t, U, V, k, lst = None):
    A = gen_VB_basis(n, e, S)
    Hmt = sh.sparse_hamiltonian(n, e, ms, t, U, V)
    #------------------------------------------------------------------------------
    # Forming the VB basis corresponding to the specified ms
    diff = int(S - ms)
    for i in range(0, diff, 1):
        A = ms_lowered_VB_basis(A, n)
    #------------------------------------------------------------------------------
    stm, sdict = form_spin_stm(n, e, ms, A)
    
    H = (stm.getH()).dot(Hmt.dot(stm))
    ovrlp = (stm.transpose()).dot(stm)
    
    D = H.get_shape()[0]
    if(D <= 40):
        if(k == D):
            H = H.todense()
            ovrlp = ovrlp.todense()
            w, v = sl.eig(H, b=ovrlp)
        else:
            w, v = ssl.eigsh(H, M=ovrlp, k=k, which='SA', tol=1e-5)
    else:
        w, v = ssl.eigsh(H, M=ovrlp, k=k, which='SA', tol=1e-5)
        
    Evec = stm.dot(v)   #---> Conversion to Slater basis
    
    dim = np.shape(Evec)
    d = 4
    st_mat = np.zeros((np.power(d,n), k), dtype = float)
    states = sh.states_with_MS(n, e, ms)[0]
    for i in range(0, dim[1], 1):      # Column index
        for j in range(0, dim[0], 1):
            st_mat[states[j], i] = Evec[j,i]
    if(lst!=None):
        if(all(isinstance(x, int) for x in lst) and max(lst)<k):
            st_mat = st_mat[:, lst]
            w = w[lst]
        elif(all(isinstance(x, int) for x in lst) != True or max(lst)>=k):
            raise Exception('Enter a valid list for lst')
    
    return st_mat, w
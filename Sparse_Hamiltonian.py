#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:18:24 2019

@author: agnish
"""

#%%
def extract_last2bits(n):
    x=n>>2
    y=x<<2
    return(n-y)

def extract_lastkbits(n,k):
    x=n>>k
    y=x<<k
    return(n-y)

#-------------------------------------------------------------------------
#Finds the next highest integer with a fixed number of 0's and 1's
#-------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------------------------
#This function retuns the integers for the states in ascending order as well as their occupancies in two separate arrays.
#n is the number of orbitals and e is the number of electrons
#--------------------------------------------------------------------------------------------------------------------------
def states_with_MS(n, e, ms): 
    lowest=2**e-1
    z=lowest
    highest=(2**(2*n-e))*lowest
    MS_states=[]
    all_orb_occu=[]
    while(z<=highest):
        x=z
        orb_occu=[]    #stores 0,1,2,3 depending on whether orbital has 0, down, up or both electrons
        mstot=0.0
        
        #---------------------------------------------------------
        # Calculating MStotal
        #---------------------------------------------------------
        
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
            all_orb_occu.append(orb_occu) #stores orbital occupancy corresponding to a state.
        z=next_highest_int(z)
    return(MS_states, all_orb_occu)



#----------------------------------------------------------------------------------------------------------------------
#Extracts the number of ones before the kth digit in z and return 1 or -1 depending on whether count is even or odd
#----------------------------------------------------------------------------------------------------------------------

def extract_phase(z,k):
    x=abs(z)
    count=0
    i=0
    while(i<k-1):
        count=count+(x&1)
        x=x>>1
        i=i+1
    if(count%2==0):
        return(1)
    else:
        return(-1)



#------------------------------------------------------------------------------------------------------------------
# Creates up or down spin electron in the integer representation z at the kth orbital with the appropriate phase
# '10' for up electrons and '01' for down electrons
#------------------------------------------------------------------------------------------------------------------

def create_electron(z,k,spin): #Creates up or down spin electron in the integer representation z at the kth orbital
                               #'10' for up electrons and '01' for down electrons
    if (z==None):
        return None
    else:
        x=abs(z)
        if(x!=0):
            initial_phase = int(z/x)
        else:
            initial_phase = 1
        a=x>>(2*(k-1))
        b=extract_last2bits(a)
        if(spin=='up'):
            phase=initial_phase*extract_phase(z,2*k)
            if((b==2)|(b==3)):
                return(None)
            else:
                return(phase*(x+(2**(2*k-1))))
        else:
            phase=initial_phase*extract_phase(z,2*k-1)
            if((b==1)|(b==3)):
                return(None)
            else:
                return(phase*(x+(2**(2*(k-1)))))


#------------------------------------------------------------------------------------------------------------------
# Destroys up or down spin electron in the integer representation z at the kth
# orbital with the appropriate phase '10' for up electrons and '01' for 
# down electrons
#------------------------------------------------------------------------------------------------------------------

def destroy_electron(z,k,spin):
    if (z==None):
        return None
    else:
        x=abs(z)
        if(x!=0):
            initial_phase = int(z/x)
        else:
            initial_phase = 1
        a=x>>(2*(k-1))
        b=extract_last2bits(a)
        if(spin=='up'):
            phase=initial_phase*extract_phase(z,2*k)
            if((b==0)|(b==1)):
                return(None)
            else:
                return(phase*(x-(2**(2*k-1))))
        else:
            phase=initial_phase*extract_phase(z,2*k-1)
            if((b==0)|(b==2)):
                return(None)
            else:
                return(phase*(x-(2**(2*(k-1)))))

#------------------------------------------------------------------------------
# This function calculates the occupancy of any site of a given state
#------------------------------------------------------------------------------
def orbital_occupancy(z, k):
    x = z>>(2*(k-1))
    y = extract_last2bits(x)
    if((y==1)or(y==2)):
        return 1
    elif(y==3):
        return 2
    else:
        return 0


#-------------------------------------------------------------------------------------------------------------------
# This function gives out the final states for a given state after
# hamiltonian acts on it as an array P
# Array A gives the absolute integer values of the final states
# Array count gives the sum of coefficients of the elements in array P
# X=integer value of the MS0 state, n=number of orbitals
#-------------------------------------------------------------------------------------------------------------------

import numpy as np

def final_states(x,n,t,U,V):
    i=1
    P=[]
    P_coeff=[]
    #--------------------------------------
    # Creating the array P
    #--------------------------------------
    
    while(i<n):
        a1=destroy_electron(x,i+1,'up')
        a2=create_electron(a1,i,'up')
        if(a2!=None):
            P.append(a2)
            P_coeff.append(t)
        b1=destroy_electron(x,i+1,'down')
        b2=create_electron(b1,i,'down')
        if(b2!=None):
            P.append(b2)
            P_coeff.append(t)
        c1=destroy_electron(x,i,'down')
        c2=create_electron(c1,i+1,'down')
        if(c2!=None):
            P.append(c2)
            P_coeff.append(t)
        d1=destroy_electron(x,i,'up')
        d2=create_electron(d1,i+1,'up')
        if(d2!=None):
            P.append(d2)
            P_coeff.append(t)
        i=i+1
    
    i=1
    while(i<=n):
        db = orbital_occupancy(x,i)
        if(db == 2):
            P.append(x)
            P_coeff.append(U)
        i=i+1
    
    i=1
    while(i<n):
        n1 = orbital_occupancy(x, i)
        n2 = orbital_occupancy(x, i+1)
        if(n1*n2!=0):
            P.append(x)
            P_coeff.append(n1*n2*V)
        i=i+1
    
    
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

        
    return(P,A,count)

#%%
# Returns index of x in arr if present, else -1 
def binarySearch (arr, l, r, x):  #l and r denote the start and end indices of the array arr respectively
    # Check base case 
    if r >= l: 
  
        mid = int(l + (r - l)/2)
  
        # If element is present at the middle itself 
        if arr[mid] == x: 
            return mid 
          
        # If element is smaller than mid, then it  
        # can only be present in left subarray 
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
  
        # Else the element can only be present  
        # in right subarray 
        else: 
            return binarySearch(arr, mid+1, r, x) 
  
    else: 
        # Element is not present in the array 
        return -1

#--------------------------------------------------------------------------------------------------------------
# This function takes input the number of orbitals as well as the number of electrons and creates 
# a compressed row format(csr) matrix storage for the Hubbard Hamiltonian
#--------------------------------------------------------------------------------------------------------------
from scipy.sparse import csr_matrix

def sparse_hamiltonian(n,e,ms,t,U,V): #n=number of orbitals, e=number of electrons
    ms0states=states_with_MS(n,e,ms)[0]
    P_all=[]
    i=0
    while(i<len(ms0states)):
        P=final_states(ms0states[i],n,t,U,V)
        P_all.append(P)
        i=i+1
    
    
    row_count = np.array([0])
    col_idx_arr = [] #Stores the column index of the first unique element for the Hamiltonian
    val_arr = []
    
    L=len(P_all[0][1])
    a=0
    while(a<L):
        col_index=binarySearch(ms0states,0, len(ms0states)-1, P_all[0][1][a])
        col_val=P_all[0][2][a]
        col_idx_arr.append(col_index)
        val_arr.append(col_val)
        a=a+1
        
    
    k = 0
    while(k < len(ms0states)):
        L=len(P_all[k][1])
        row_count = np.append(row_count, L+row_count[k])
        a=0
        if(k>0):
            while(a<L):
                col_index = binarySearch(ms0states,0, len(ms0states)-1, P_all[k][1][a])
                col_val = P_all[k][2][a]
                col_idx_arr.append(col_index)
                val_arr.append(col_val)
                a=a+1
        k=k+1
        
    dim = len(row_count) - 1
    A = csr_matrix((val_arr, col_idx_arr, row_count), shape=(dim, dim)).transpose()
    return(A)

#%%
#------------------------------------------------------------------------------
# The following function returns the specified states (the ones specified by k)
# e.g. k = 1 is ground state, k = 2 is the 1st excited state and so on.
# The states are in the complete Fock basis instead of being in the particular
# Ms basis. 
# The parameters for forming the sparse hamiltonian need to be input. k denotes
# the number of states we are interested in.
# A denotes the states we are interested in. e.g. if we are interested in only 
# the 1st excited state and 5th excited state, then A = [1,5] (Ground state is
# indexed at 0).
#------------------------------------------------------------------------------
import scipy.sparse.linalg as ssl
import scipy.linalg as sl
def estates_of_H(n, e, ms, t, U, V, k, A = None):
    H = sparse_hamiltonian(n, e, ms, t, U, V)
    
    D = H.get_shape()[0]
    if(D <= 40):
        if(k == D):
            H = H.todense()
            Eval, Evec = sl.eigh(H)
        else:
            Eval, Evec = ssl.eigsh(H, k=k, which='SA', tol = 1e-5)
    else:
        Eval, Evec = ssl.eigsh(H, k=k, which='SA', tol = 1e-5)
    dim = np.shape(Evec)
    d = 4
    st_mat = np.zeros((np.power(d,n), k), dtype = float)
    states = states_with_MS(n, e, ms)[0]
    for i in range(0, dim[1], 1):      # Column index
        for j in range(0, dim[0], 1):
            st_mat[states[j], i] = Evec[j,i]
    if(A!=None):
        if(all(isinstance(x, int) for x in A) and max(A)<k):
            st_mat = st_mat[:, A]
        elif(all(isinstance(x, int) for x in A) != True or max(A)>=k):
            raise Exception('Enter a valid list for A')
    
    return st_mat

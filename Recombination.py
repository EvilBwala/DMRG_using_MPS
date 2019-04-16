#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:26:37 2019

@author: agnish
"""


import numpy as np
import Contractions as ct
import Hubbard as hb
import Sparse_Hamiltonian as sh
import time
import os

#------------------------------------------------------------------------------

L = 4
e1 = 5
e2 = 3
ms1 = -0.5
ms2 = 0.5
t = 1.
U = 4.
V = 2.5
T = 0.1
d = 4
D = 64
tstep = 0.01
Total_time = 200
tol = 1e-4
N = int(Total_time/tstep)

#%%
#------------------------------------------------------------------------------
# Creating a new directory with the name L_L_e1_e2 to store data
# Initializing the overlp, Parameters and iter_time files
#------------------------------------------------------------------------------

cdir = os.getcwd()
newpath = os.path.join(cdir, '{}_{}_{}_{}'.format(L,L,e1,e2)) 
if not os.path.exists(newpath):
    os.makedirs(newpath)

if os.path.exists("{}_{}_{}_{}/Parameters".format(L,L,e1,e2)):
    os.remove("{}_{}_{}_{}/Parameters".format(L,L,e1,e2))
    
with open("{}_{}_{}_{}/Parameters".format(L,L,e1,e2), "a") as ftime:
    ftime.write("L\t\t= {} \ne1\t\t= {} \ne2\t\t= {} \nms1\t\t= {} \
                \nms2\t\t= {} \nt\t\t= {} \nU\t\t= {} \
                \nV\t\t= {} \nT\t\t= {} \nd\t\t= {} \nD\t\t= {} \ntstep\t\t= {} \
                \nTotal_time\t= {} \
                \ntol\t\t= {} \nN\t\t= {}".format(L,e1,e2,ms1,ms2,t,U,V,T,d,D,tstep, \
                Total_time,tol,N))

if os.path.exists("{}_{}_{}_{}/overlap".format(L,L,e1,e2)):
    os.remove("{}_{}_{}_{}/overlap".format(L,L,e1,e2))
with open("{}_{}_{}_{}/overlap".format(L,L,e1,e2), "a") as fovlp:
    fovlp.write("Iteration \t\t Overlap \n")

if os.path.exists("{}_{}_{}_{}/iter_time".format(L,L,e1,e2)):
    os.remove("{}_{}_{}_{}/iter_time".format(L,L,e1,e2))
with open("{}_{}_{}_{}/iter_time".format(L,L,e1,e2), "a") as ftime:
    ftime.write("Iteration \t Total \n")

#%%
#------------------------------------------------------------------------------
# Forming the initial State MPS
#------------------------------------------------------------------------------
St1 = sh.estates_of_H(L,e1,ms1,t,U,V,1)
M1 = ct.form_vidal_MPS(St1, d, D, L)
St2 = sh.estates_of_H(L,e2,ms2,t,U,V,1)
M2 = ct.form_vidal_MPS(St2, d, D, L)
M2 = M2[1:]
vform = M1 + M2         #--> vform is now of the form 1-2-3-4-1'-2'-3'-4'
# We need vform to be of the form 1-1'-2-2'-3-3'-4-4'.
for i in range(0,L,1):
    vform = ct.permute_indices_vidal_right(vform, 2*i+1, L+i, D)
#%%
#------------------------------------------------------------------------------
# Forming the time evolution operators
#------------------------------------------------------------------------------
Tp3 = hb.form_TEOp_three_site(d, tstep, t, V)
Tp1 = hb.form_TEOp_one_site(d, tstep, U)
TETp = hb.form_TEOp_two_site(d, tstep, T, 0)

#%%
M = vform[:]
M_L = ct.vidal_to_LCM(vform)
M_L = ct.vidal_to_LCM(M)
ovlp = [ct.overlap(M_L,M_L)]
tme = []

t0 = time.clock()
for j in range(0,N,1):
    print(j)
    t1 = time.clock()
    #--------------------------------------------------------------------------
    # Odd bonds evolution
    #--------------------------------------------------------------------------
    for i in range(0,int(L/2),1):
        M = hb.TEBDi(M, D, Tp3, 4*i)        #--> Chain 1
        M = hb.TEBDi(M, D, Tp3, 4*i+1)      #--> Chain 2
    #--------------------------------------------------------------------------
    # Even bonds evolution
    #--------------------------------------------------------------------------
    for i in range(0, int((L-1)/2), 1):
        M = hb.TEBDi(M, D, Tp3, 4*i+2)      #--> Chain 1
        M = hb.TEBDi(M, D, Tp3, 4*i+3)      #--> Chain 2
    #--------------------------------------------------------------------------
    # Evolution due to On-site term
    #--------------------------------------------------------------------------
    for i in range(0, 2*L, 1):
        M = hb.TEBDi(M, D, Tp1, i)      #--> Both Chains, all sites
    
    #--------------------------------------------------------------------------
    # Evolution of inter-chain hopping
    #--------------------------------------------------------------------------
    for i in range(0,L,1):
        M = hb.TEBDi(M, D, TETp, 2*i)

    
    M_L = ct.vidal_to_LCM(M)
    ovlp.append(ct.overlap(M_L,M_L))
    with open("{}_{}_{}_{}/overlap".format(L,L,e1,e2), "a") as fovlp:
        fovlp.write("{} \t\t {} \n".format(j, ovlp[j+1]))
    
    #--------------------------------------------------------------------------
    # Renormalizing the wavefunction if overlap exceeds a given tolerance
    #--------------------------------------------------------------------------
    if((abs(ovlp[j+1]-ovlp[0]))/ovlp[0] > tol):
        M[2*L] = 1/np.sqrt(ovlp[j+1])*M[2*L]
        A = ct.vidal_to_LCM(M)
        M = ct.LCM_to_vidal(A)
        
    t2 = time.clock()
    tj = t2-t1
    tme.append(tj)
    
    if(j%40==1):
        np.savez_compressed('{}_{}_{}_{}/{}'.format(L,L,e1,e2,j), *M)
    
    with open("{}_{}_{}_{}/iter_time".format(L,L,e1,e2), "a") as ftime:
        ftime.write("{} \t\t {} \n".format(j, tme[j]))
    
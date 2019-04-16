#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:11:00 2019

@author: agnish
"""

import numpy as np
import Contractions as ct
import Hubbard as hb
import time
import os

#%%
#------------------------------------------------------------------------------
loc = "4_4_5_3" #--------------------------------------------------------------
#------------------------------------------------------------------------------

f = open("{}/Parameters".format(loc), "r")
lines = f.readlines()
L = int(lines[0][5:])
e1 = int(lines[1][6:])
e2 = int(lines[2][6:])
ms1 = float(lines[3][6:])
ms2 = float(lines[4][6:])
t = float(lines[5][5:])
U = float(lines[6][5:])
V = float(lines[7][5:])
T = float(lines[8][5:])
d = int(lines[9][5:])
D = int(lines[10][5:])
tstep = float(lines[11][9:])
#------------------------------------------------------------------------------
Total_time = 200 #-------------------------------------------------------------
#------------------------------------------------------------------------------
tol = float(lines[13][7:])
N = int(Total_time/tstep)
f.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
file_idx = 41 #--------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

f_mat = np.load("{}/{}.npz".format(loc,file_idx))
M_r = [0]*(4*L+1)
for j in range(0, 4*L+1, 1):
    M_r[j] = f_mat['arr_{}'.format(j)]

cdir = os.getcwd()
newpath = os.path.join(cdir, '{}_{}_{}_{}_{}'.format(file_idx,L,L,e1,e2)) 
if not os.path.exists(newpath):
    os.makedirs(newpath)
#%%
#------------------------------------------------------------------------------
# Forming the time evolution operators
#------------------------------------------------------------------------------
Tp3 = hb.form_TEOp_three_site(d, tstep, t, V)
Tp1 = hb.form_TEOp_one_site(d, tstep, U)
TETp = hb.form_TEOp_two_site(d, tstep, T, 0)

#%%
M = M_r[:]
M_L = ct.vidal_to_LCM(M_r)
ovlp = [ct.overlap(M_L,M_L)]
tme1 = []

t0 = time.clock()
for j in range(1,N-file_idx,1):
    print(file_idx+j)
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
        M = hb.TEBDi(M, D, Tp3, 4*i+3)    #--> Chain 2
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
        fovlp.write("{} \t\t {} \n".format(file_idx+j, ovlp[j]))
    
    #--------------------------------------------------------------------------
    # Renormalizing the wavefunction if overlap exceeds a given tolerance
    #--------------------------------------------------------------------------
    if((abs(ovlp[j]-ovlp[0]))/ovlp[0] > tol):
        M[2*L] = 1/np.sqrt(ovlp[j])*M[2*L]
        A = ct.vidal_to_LCM(M)
        M = ct.LCM_to_vidal(A)
    
    if(j%40==0):
        np.savez_compressed('{}_{}_{}_{}_{}/{}'.format(file_idx,L,L,e1,e2,file_idx+j), *M)
    
        
    t2 = time.clock()
    tj = t2-t1
    tme1.append(tj)
    
    with open("{}_{}_{}_{}/iter_time".format(L,L,e1,e2), "a") as ftime:
        ftime.write("{} \t\t {} \n".format(file_idx+j, tme1[j-1]))
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:16:53 2019

@author: agnish
"""

import numpy as np
import Contractions as ct
import Symmetries as syms
import os
import shutil
import time
#%%
loc = "4_4_5_3"
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
Total_time = float(lines[12][13:])
tol = float(lines[13][7:])
N = int(lines[14][5:])
f.close()

cdir = os.getcwd()
newpath = os.path.join(cdir, 'overlaps') 
if os.path.exists(newpath):
    shutil.rmtree(newpath)
if not os.path.exists(newpath):
    os.makedirs(newpath)
fmax = open("overlaps/{}maxyield".format(L), "a")
idx = open("overlaps/{}stateidx".format(L), "a")
#%%
#------------------------------------------------------------------------------
# Name of file where the overlap values are stored is of the form:
# (L)(space1)(space2)(c1+1)(c2+1)
#------------------------------------------------------------------------------

overlp = []
ti = []

for space1 in ['1Ag', '3Ag', '1Bu', '3Bu']:
    for space2 in ['1Ag', '3Ag', '1Bu', '3Bu']:
        for c1 in range(0,2,1):
            for c2 in range(0,2,1):
                t1 = time.time()
                #--------------------------------------------------------------
                # Form the MPS M, against which we want to calculate the overlaps
                #--------------------------------------------------------------
                St1 = syms.sym_adap_state(L, space1, t, U, V, c1)[0]
                St2 = syms.sym_adap_state(L, space2, t, U, V, c2)[0]
                M1 = ct.form_vidal_MPS(St1, d, D, L)
                M2 = ct.form_vidal_MPS(St2, d, D, L)
                M2 = M2[1:]
                M = M1 + M2
                for i in range(0,L,1):
                    M = ct.permute_indices_vidal_right(M, 2*i+1, L+i, D)
                M = ct.vidal_to_LCM(M)
                f = open("overlaps/{}{}{}{}{}".format(L,c1+1,space1,c2+1,space2), "a")
                for i in range(0,250,1):
                    x = 40*i + 1
                    tme = x*tstep*0.6
                    f_mat = np.load("{}/{}.npz".format(loc,x))
                    M_r = [0]*(4*L+1)
                    for j in range(0, 4*L+1, 1):
                        M_r[j] = f_mat['arr_{}'.format(j)]
                    MLC = ct.vidal_to_LCM(M_r)
                    ovlp = abs(ct.overlap(M, MLC))**2
                    f.write("{} \n".format(ovlp))
                    overlp.append(ovlp)
                    ti.append(tme)
                f.close()
                max_yield = max(overlp)
                fmax.write("{} \n".format(max_yield))
                idx.write("{}{}{}{} \n".format(c1+1,space1,c2+1,space2))
                t2 = time.time()
                print("{}{}{}{} \t {}".format(c1+1,space1,c2+1,space2, t2-t1))
fmax.close()
idx.close()
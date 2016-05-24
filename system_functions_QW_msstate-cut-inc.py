import unittest
import warnings
import numpy as np
from copy import copy, deepcopy
# from PyEmma
from pyemma.msm.analysis import pcca

def QW_msstates( bins ):
    # Definition of the metastable states, specifically for the quad well system!
    bnd_A = [-0.90, -0.60]
    bnd_B = [-0.40, -0.10]
    bnd_C = [0.10, 0.40]
    bnd_D = [0.60, 0.90]
    A = []
    B = []
    C = []
    D = []
    for i in range(0,bins.size):
        if ( (bins[i]>bnd_A[0]) and (bins[i]<bnd_A[1]) ):
            A.append(i)
        elif ( (bins[i]>bnd_B[0]) and (bins[i]<bnd_B[1]) ):
            B.append(i)
        elif ( (bins[i]>bnd_C[0]) and (bins[i]<bnd_C[1]) ):
            C.append(i)
        elif ( (bins[i]>bnd_D[0]) and (bins[i]<bnd_D[1]) ):
            D.append(i)

    return A, B, C, D

# shortcut function for msstates, should always be linked to the particular calculation (above) for each system
get_msstates = lambda T, lcc, bins: QW_msstates( bins )

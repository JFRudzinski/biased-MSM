import unittest
import warnings
import numpy as np
from copy import copy, deepcopy
# from PyEmma
from pyemma.msm.analysis import pcca
from functions import rmse_mfpt_all, regulate_constraint

def QW_msstates( bins ):
    # Definition of the metastable states, specifically for the quad well system!
    div = [-0.5, 0.0, 0.5]
    A = []
    B = []
    C = []
    D = []
    for i in range(0,bins.size):
        if (bins[i] < div[0]):
            A.append(i)
        elif (bins[i] < div[1]):
            B.append(i)
        elif (bins[i] < div[2]):
            C.append(i)
        else:
            D.append(i)

    return A, B, C, D

# shortcut function for msstates, should always be linked to the particular calculation (above) for each system
get_msstates = lambda T, lcc, bins: QW_msstates( bins )

def calc_FW( T, mfpt_AA, mss_CG, nmfptb, flag_rel_mfpt, FW_params):
    FW = rmse_mfpt_all(T, mfpt_AA, mss_CG, nmfptb, flag_rel_mfpt)
    return FW

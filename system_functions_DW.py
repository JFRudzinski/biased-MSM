import unittest
import warnings
import numpy as np
from copy import copy, deepcopy
# from PyEmma
from pyemma.msm.analysis import pcca

def DW_msstates( bins ):
    # Definition of the metastable states, specifically for the double well system!
    div = 6.0
    buff = 0.0
    A = []
    B = []
    for i in range(0,bins.size):
        if (bins[i] < div-buff):
            A.append(i)
        elif (bins[i] > div+buff):
            B.append(i)

    return A, B

# shortcut function for msstates, should always be linked to the particular calculation (above) for each system
get_msstates = lambda T, lcc, bins: DW_msstates( bins )

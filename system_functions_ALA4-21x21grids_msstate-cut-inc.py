import unittest
import warnings
import numpy as np
from copy import copy, deepcopy
# from PyEmma
from pyemma.msm.analysis import pcca

def ALA4_msstates( T_AA ):

    # Define the metastable states from analysis of T_AA_mle
    membership=pcca(T_AA, 4)
    # the precise definitions were fine-tuned by hand, I should plot the PCCA states to double check!
    E1 = np.where(membership[:,0]>0.4)[0]
    E2 = np.where(membership[:,1]>0.75)[0]
    H = np.where(membership[:,2]>0.6)[0]
    I = np.where(membership[:,3]>0.75)[0]
    # make sure there is no overlap
    #print "checking the overlap of metastable states:"
    check = [val for val in H if val in I]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in H if val in E1]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in H if val in E2]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in I if val in E1]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in I if val in E2]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in E1 if val in E2]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')

    return H, I, E2, E1

# shortcut function for msstates, should always be linked to the particular calculation (above) for each system
get_msstates = lambda T, lcc, bins: ALA4_msstates( T )


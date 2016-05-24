import unittest
import warnings
import numpy as np
from copy import copy, deepcopy
# from PyEmma
from pyemma.msm.analysis import pcca
from functions import rmse_mfpt_all, regulate_constraint

def ALA3_msstates( T_AA ):

    # Define the metastable states from analysis of T_AA_mle
    membership=pcca(T_AA, 3)
    # the precise definitions were fine-tuned by hand, I should plot the PCCA states to double check!
    lH = np.where(membership[:,0]>0.60)[0]
    aH = np.where(membership[:,1]>0.60)[0]
    B = np.where(membership[:,2]>0.60)[0]
    # make sure there is no overlap
    #print "checking the overlap of metastable states:"
    check = [val for val in lH if val in aH]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in lH if val in B]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')
    check = [val for val in aH if val in B]
    if (len(check) != 0 ):  raise ValueError('Some metastable states are overlapping!')

    return B, aH, lH

# shortcut function for msstates, should always be linked to the particular calculation (above) for each system
get_msstates = lambda T, lcc, bins: ALA3_msstates( T )

def calc_FW( T, mfpt_AA, mss_CG, nmfptb, flag_rel_mfpt, FW_params): 
    FW = rmse_mfpt_all(T, mfpt_AA, mss_CG, nmfptb, flag_rel_mfpt)
    return FW
    

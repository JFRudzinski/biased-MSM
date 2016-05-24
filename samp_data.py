__author__ = 'JFR'

import numpy as np
import math
from copy import copy, deepcopy
from functions import *

class SampData(object):
    """
        Store variables for output and whatnot...
    """

    def __init__(self): 
        """
        Initializes the transition matrix sampler with the observed count matrix

        Variables:
        -----------
        C : ndarray(n,n)
            count matrix containing observed counts. Do not add a prior, because this sampler intrinsically
            assumes a -1 prior!

# variables for the sampling
Topt_CG_b = []
Copt_CG_b = []
muopt_CG_b = []
OBSopt_CG_b = []
OBSopt_mfpt_CG_b = []
EQopt_CG_b = []
EWopt_CG_b = []

        """

        self.T = np.array([])
        self.X = np.array([])
        self.mu = np.array([])
        self.ts = np.array([])
        self.mfpt = np.array([])
        self.EQ = np.array([])
        self.EW = np.array([])
        self.lamb = np.array([])
        self.bins = np.array([])

        self.outfnm = None

    def _init(self, sampler):
        self.T = np.array([deepcopy(sampler.T)])
        self.mu = np.array([deepcopy(sampler.mu)])
        self.X = np.array([deepcopy(sampler.X)])
        self.EQ = np.array([deepcopy(sampler.EQ)])
        self.EW = np.array([deepcopy(sampler.EW)])
        self.lamb = np.array([deepcopy(sampler.lamb)])

    def _update(self, sampler):
        self.T = np.append( self.T, np.array([deepcopy(sampler.T)]), axis=0 )
        self.mu = np.append( self.mu, np.array([deepcopy(sampler.mu)]), axis=0 )
        self.X = np.append( self.X, np.array([deepcopy(sampler.X)]), axis=0 )
        self.EQ = np.append( self.EQ, np.array([deepcopy(sampler.EQ)]) )
        self.EW = np.append( self.EW, np.array([deepcopy(sampler.EW)]) )
        self.lamb = np.append( self.lamb, np.array([deepcopy(sampler.lamb)]) )

    def _remove(self, i0, it, sampler):
        ind = np.arange(it)[i0:]
        self.T = np.delete( self.T, ind, axis=0 )
        self.mu = np.delete( self.mu, ind, axis=0  )
        self.X = np.delete( self.X, ind, axis=0 )
        self.mfpt = np.delete( self.mfpt, ind, axis=0  )
        self.ts = np.delete( self.ts, ind, axis=0 )
        self.EQ = np.delete( self.EQ, ind )
        self.EW = np.delete( self.EW, ind )
        self.lamb = np.delete( self.lamb, ind )


        sampler.T = deepcopy(self.T[i0-1])
        sampler.mu = deepcopy(self.mu[i0-1])
        sampler.X = deepcopy(self.X[i0-1])
        sampler.mfpt = deepcopy(self.mfpt[i0-1])
        sampler.ts = deepcopy(self.ts[i0-1])
        sampler.EQ = deepcopy(self.EQ[i0-1])
        sampler.EW = deepcopy(self.EW[i0-1])
        sampler.lamb = deepcopy(self.lamb[i0-1])

    def _save_output(self):
        np.savez(self.outfnm, T=self.T, mu=self.mu, X=self.X, ts=self.ts, mfpt=self.mfpt, EQ=self.EQ, EW=self.EW, bins=self.bins, lamb=self.lamb)



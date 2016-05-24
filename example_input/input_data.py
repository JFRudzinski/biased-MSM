__author__ = 'JFR'

import numpy as np
import math
from copy import copy, deepcopy

class InputData(object):
    """
        Feed in input data here and whatnot...
    """

    def __init__(self): 
        """
        """
        Nproc = 16 
        self.DEBUG = False  # For special debugging options, see below

        # File paths
        data_path = '/data/isilon/rudzinski/msmb_test/2015_08_12/ALA3/'
        AA_dir = data_path+'AA/'
        CG_dir = data_path+'PLUM/'
        dtraj_dir = 'traj_data/dtraj_data/'
        bin_dir = 'traj_data/bin_data/'
        # bin data
        self.bin_ctrs_fnm = AA_dir+bin_dir+'binctrs_16x16bins.npy'
        self.bin_ctrs_CG_fnm = CG_dir+bin_dir+'binctrs_16x16bins.npy'
        # traj data
        self.dtraj_AA_fnm = AA_dir+dtraj_dir+'dtraj_AA_5traj_phipsi.npy'
        self.dtraj_CG_fnm = CG_dir+dtraj_dir+'dtraj_CG_1traj_phipsi.npy'
        self.TRJ_types = ['1D','2D']
        self.TRJ_type = self.TRJ_types[1]

        # MSM
        self.tau_AA = 50
        self.tau_CG = 8
        self.flag_sliding = True
        self.prior = 0.0
        self.trimfrac = 0.01

        # OBS
        self.nts = 6
        self.nmfpt = 6
        self.nmfptb = 6

        # basic samp and opt options
        # for lack of better treatment of global variable, just define these here for now
        self.OPT_types = [ 'fixed_lamb', 'slow_growth', 'rep_exch' ]
        self.MC_types = [ 'fixed_step', 'corr_move' ]
        self.OPT_type = self.OPT_types[2]
        self.MC_type = self.MC_types[1]
        # some other relatively high level options
        self.fixed_pi = False
        self.flag_ub = False # sets special values for the energies for proper unbiased sampling
        # step variables
        self.NSweep_p_Samp = 2 # sample in multiples of this number of sweeps, >= 2, must be divisible by 2!!!
        self.NSweep_eq = 0 # throw out this many minimization sweeps before collecting statistics
        self.NSamp_max_conv = 100 # number of samples to compute avg from
        self.NSamp_p_lam_max = 5 # max number of samples before quitting, or moving to next lambda for slow_growth
        # lambda-related variables
        self.lamb0 = np.linspace(0.0, 1.0, Nproc) # for rep_exch
        #self.lamb0 = 1.0*np.ones(Nproc) # for fixed_lamb or slow_growth
        self.beta_interp = 1.0 # interpolation param for beta (nb - acc rate was 0.3/0.4 with beta-0.5)
        self.init_interp = 1.0 # interpolation param for T_init and X_init,  1 => CG mle, 0 => AA map'd
        self.beta0C = 3.5e2 # Const to mult F_CG by
        # energy-related variables
        self.EQ_ref = None # When 'None', uses the projected AA value. # -251.658 # avg EQ from lamb = 0 MCMC sampling, with beta1
        self.EWtol = 0.01 # constraint tolerence
        self.Econvfrac = 1e-6 # defn of energy conversion in terms of the fractional difference between samples

        # variables specifically for OPT_type == 'fixed_step'

        # variables specifically for OPT_type == 'corr_move'
        self.corr_step_len = 1
        # variables specifically for MC_type == 'fixed_lamb'

        # variables specifically for MC_type == 'slow_growth'
        self.dlamb0 = 0.01 # start with a relatively high change in lamb_b
        self.Nhrnd = 1 # how many half rounds to perform.  Half round is defined from lamb=1 to EW<tol or visa versa
        self.mult_jump = 0.80 # fact to mult dlamb by in case of an energy jump
        self.mult_stuck = 1.20 # fact to mult dlamb by in case the energy is stuck
        self.dlamb_frac_max = 0.25 # max allowed frac change in lamb
        self.dlamb_frac_min = 0.01 # min allowed frac change in lamb
        self.Ejumpfrac = 10.00 # don't let the avg energy per Sample jump by more than this fraction of the previous avg
        self.Estuckfrac = 0.01 # don't let the avg energy per Sample move less than this fraction of the previous avg
        # variables specifically for MC_type == 'rep_exch'
        self.NSweep_p_Swap_rx = 1

        # output
        self.outfnm = 'output_variables'

        if (self.DEBUG):
            self.NSweep_p_Samp = 2
            self.NSweep_eq = 0
            self.dlamb0 = 0.10
            self.Ejumpfrac = 100.0
            self.Estuckfrac = 0.01
            self.Econvfrac = 0.50
            self.EWtol = 0.01

    def _save_output(self):
        np.savez(self.outfnm+'_ID', bin_ctrs_fnm=self.bin_ctrs_fnm, bin_ctrs_CG_fnm=self.bin_ctrs_CG_fnm, dtraj_AA_fnm=self.dtraj_AA_fnm, dtraj_CG_fnm=self.dtraj_CG_fnm, tau=self.tau, prior=self.prior, nts=self.nts, nmfpt=self.nmfpt, nmfptb=self.nmfptb, EQ_ref=self.EQ_ref, NSweep_p_Samp=self.NSweep_p_Samp, NSweep_eq=self.NSweep_eq, lamb0=self.lamb0, dlamb0=self.dlamb0, beta0C=self.beta0C)






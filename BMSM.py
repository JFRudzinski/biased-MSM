import os
import sys
# change the working directory to that given as a command line argument
wdir = os.environ.get("WORKING_DIRECTORY","./")
if len(sys.argv) > 1: wdir = sys.argv[1]
os.chdir( wdir )
sys.path.insert(0, wdir)
import unittest
import warnings
# numpy
import numpy as np
# stuff from pyemma
import pyemma
from pyemma.msm.io import read_discrete_trajectory
import pyemma.msm.estimation.dense.transition_matrix_biased_sampling_rev as bias_samp
# other tools specifically built for this program
import input_data as ID
import samp_data as SD
from functions import *
# other stuff
from copy import copy, deepcopy
import random
# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rep_ind_perm = np.arange(size)

if (rank == 0):
    print 'starting by getting the simulation data...'
    print '\n'

# load the input data
ID = ID.InputData()
sys_mod_nm = 'system_functions_'+str(ID.sysH)
sys_mod = __import__(sys_mod_nm)
MC_types = deepcopy(ID.MC_types)
OPT_types = deepcopy(ID.OPT_types)
TRJ_types = deepcopy(ID.TRJ_types)
# get the bins
bin_ctrs = np.load(ID.bin_ctrs_fnm)
bin_ctrs_CG = np.load(ID.bin_ctrs_CG_fnm)
if ( ID.TRJ_type == TRJ_types[0] ): # TRJ_type == '1D'
    dtraj_AA = np.load(ID.dtraj_AA_fnm)
    dtraj_AA = dtraj_AA.astype(int)
    dtraj_AA = dtraj_AA.tolist()
    dtraj_CG = np.load(ID.dtraj_CG_fnm)
    dtraj_CG = dtraj_CG.astype(int)
    dtraj_CG = dtraj_CG.tolist()
elif ( ID.TRJ_type == TRJ_types[1] ): # TRJ_type == '2D' 
    ngridsx = len(bin_ctrs[0])
    xmin = min(bin_ctrs[0])
    dx = bin_ctrs[0][1] - bin_ctrs[0][0]
    ngridsy = len(bin_ctrs[1])
    ymin = min(bin_ctrs[1])
    dy = bin_ctrs[1][1] - bin_ctrs[1][0]
    dtraj_AA_2D = np.load(ID.dtraj_AA_fnm)
    dtraj_AA, xgrid_AA, ygrid_AA = dtraj_2D_to_1D( bin_ctrs, dtraj_AA_2D )
    dtraj_CG_2D = np.load(ID.dtraj_CG_fnm)
    dtraj_CG, xgrid_CG, ygrid_CG = dtraj_2D_to_1D( bin_ctrs_CG, dtraj_CG_2D )
else:
    raise ValueError('TRJ_type not supported!  Check input_data.py')

# compute the cmatrices
tau_AA = ID.tau_AA
tau_CG = ID.tau_CG
Cmat_AA = pyemma.msm.estimation.count_matrix(dtraj_AA, tau_AA, sliding=ID.flag_sliding, sparse_return=False, nstates=None)
Cmat_CG = pyemma.msm.estimation.count_matrix(dtraj_CG, tau_CG, sliding=ID.flag_sliding, sparse_return=False, nstates=None)

# ensure the connectedness of the matrix
# AA
lcc_AA = pyemma.msm.estimation.largest_connected_set(Cmat_AA, directed=True)
Cmat_AA_cc = pyemma.msm.estimation.largest_connected_submatrix(Cmat_AA, directed=True, lcc=lcc_AA)
# CG
lcc_CG = pyemma.msm.estimation.largest_connected_set(Cmat_CG, directed=True)
Cmat_CG_cc = pyemma.msm.estimation.largest_connected_submatrix(Cmat_CG, directed=True, lcc=lcc_CG)

# trim the count matrix and bins to ignore low-sampled states
# AA
Cmat_AA_cc, lcc_AA = trim_Cmat( Cmat_AA_cc, lcc_AA, ID)
Cmat_CG_cc, lcc_CG = trim_Cmat( Cmat_CG_cc, lcc_CG, ID)

# Set up the grids/bins
bin_ctrs_AA_cc = []
bin_ctrs_CG_cc = []
xbins_trim_AA = [] 
ybins_trim_AA = []
xbins_trim_CG = []
ybins_trim_CG = []
if ( ID.TRJ_type == TRJ_types[0] ): # TRJ_type == '1D'
    # AA
    bin_ctrs_AA_cc = [bin_ctrs[i] for i in lcc_AA]
    bin_ctrs_AA_cc = np.array(bin_ctrs_AA_cc)
    # CG
    bin_ctrs_CG_cc = [bin_ctrs_CG[i] for i in lcc_CG]
    bin_ctrs_CG_cc = np.array(bin_ctrs_CG_cc)
elif ( ID.TRJ_type == TRJ_types[1] ): # TRJ_type == '2D'
    # AA
    xgrids_trim_AA, ygrids_trim_AA = label_to_bins(lcc_AA,ngridsx)
    xbins_trim_AA, ybins_trim_AA = bins_to_values(xgrids_trim_AA, ygrids_trim_AA, bin_ctrs)
    # CG
    xgrids_trim_CG, ygrids_trim_CG = label_to_bins(lcc_CG,ngridsx)
    xbins_trim_CG, ybins_trim_CG = bins_to_values(xgrids_trim_CG, ygrids_trim_CG, bin_ctrs_CG)

# save the Cmats
#np.save('Cmat_AA',Cmat_AA_cc)
#np.save('Cmat_CG',Cmat_CG_cc)

# mle data
SD_AA = SD.SampData()
SD_AA.outfnm = ID.outfnm+'_AA'
SD_CG = SD.SampData()
SD_CG.outfnm = ID.outfnm+'_CG'

# calculate the mle
SD_AA.T, SD_AA.mu = pyemma.msm.estimation.transition_matrix(Cmat_AA_cc, reversible=True, mu=None, return_statdist = True)
SD_CG.T, SD_CG.mu = pyemma.msm.estimation.transition_matrix(Cmat_CG_cc, reversible=True, mu=None, return_statdist = True)
SD_AA.X = T_to_X(Cmat_AA_cc, SD_AA.T, SD_AA.mu)
SD_CG.X = T_to_X(Cmat_CG_cc, SD_CG.T, SD_CG.mu)
if ( ID.TRJ_type == TRJ_types[0] ): # TRJ_type == '1D'
    SD_AA.bins = deepcopy(bin_ctrs_AA_cc)
    SD_CG.bins = deepcopy(bin_ctrs_CG_cc)
elif ( ID.TRJ_type == TRJ_types[1] ): # TRJ_type == '2D'
    SD_AA.bins = deepcopy(lcc_AA)
    SD_CG.bins = deepcopy(lcc_CG)

# get the pyemma MCMC samplers ready
prior = ID.prior
sampler_CG_b = bias_samp.TransitionMatrixBiasedSamplerRev(Cmat_CG_cc+prior,prior,SD_CG.X,ID.MC_type,ID.lT) # nb - fixedstep MC

# Get the metastable states (assumes you define some metastable states, but no longer system specific!)
mss_AA = np.array( sys_mod.get_msstates( SD_AA.T, lcc_AA, bin_ctrs_AA_cc ) )
Nmss = mss_AA.shape[0]
mss_CG = deepcopy(mss_AA)
for i in range(Nmss):
    mss_tmp = proj_msstates_to_CG_bins( lcc_AA, lcc_CG, mss_AA[i] )
    mss_CG[i] = deepcopy(mss_tmp)
xbins_mss_AA = []
ybins_mss_AA = []
xbins_mss_CG = []
ybins_mss_CG = []
if ( ID.TRJ_type == TRJ_types[1] ): # TRJ_type == '2D'
    for i in range(Nmss):
        xtmp, ytmp = get_msstate_grids( lcc_AA, bin_ctrs, mss_AA[i], ngridsx )
        xbins_mss_AA.append( xtmp )
        ybins_mss_AA.append( ytmp )
        xtmp, ytmp = get_msstate_grids( lcc_CG, bin_ctrs_CG, mss_CG[i], ngridsx )
        xbins_mss_CG.append( xtmp )
        ybins_mss_CG.append( ytmp )
# save the grid info
np.savez('output_grid_data', xbins_trim_AA, ybins_trim_AA, xbins_trim_CG, ybins_trim_CG, xbins_mss_AA, ybins_mss_AA, xbins_mss_CG, ybins_mss_CG)

# Calculate the timescales and mfpts
nts = ID.nts
nmfpt = ID.nmfpt
nmfptb = ID.nmfptb
SD_CG.ts = est_ts_k(SD_CG.T,tau_CG,nts)
SD_AA.ts = est_ts_k(SD_AA.T,tau_AA,nts)
SD_CG.mfpt = mfpt_all(SD_CG.T,mss_CG)
SD_AA.mfpt = mfpt_all(SD_AA.T,mss_AA)

# make shortcut functions for sampling (assumes you define metastable states and the constraint is the set of all mfpts, but no longer system specific!)
est_ts_kfT = lambda T: est_ts_k ( T, tau_CG, nts ) 
mfpt_all_fT = lambda T: mfpt_all(T, mss_CG)
# Choose the biasing function:
FW_fT = lambda T: sys_mod.calc_FW(T, SD_AA.mfpt, mss_CG, nmfptb, ID.flag_rel_mfpt, ID.FW_params)

# 'project' the AA mle onto the CG bins
X_AA_mapd = project_X0_to_Xf(SD_CG.X, SD_AA.X, lcc_CG, lcc_AA)
T_AA_mapd = X_to_T(X_AA_mapd)

# Calculate the energies for the maximum likelihood estimates
SD_AA.EQ = -1.0*( logprob_T(Cmat_AA_cc+prior, SD_AA.T) )
SD_CG.EQ = -1.0*( logprob_T(Cmat_CG_cc+prior, SD_CG.T) )
EQ_AA_mapd = -1.0*( logprob_T(Cmat_CG_cc+prior, T_AA_mapd) )
# nb - F_AA = 0 by definition
SD_AA.EW = rmse_mfpt_all(SD_AA.T, SD_AA.mfpt, mss_AA, nmfptb, ID.flag_rel_mfpt)
SD_CG.EW = FW_fT(SD_CG.T)
F_AA_mapd = FW_fT(T_AA_mapd)

# parameters for the sampling
if (ID.EQ_ref is not None):
    EQ_ref = deepcopy(ID.EQ_ref) # avg EQ from lamb = 0 MCMC sampling, with beta1
else:
    EQ_ref = deepcopy(EQ_AA_mapd)

if ( ID.MC_type == MC_types[0] ): # if MC_type == 'fixed_step'
    if ( ID.fixed_pi ):
        NMC_p_Sweep = int( len(np.where(SD_CG.X>1e-12)[0]) - SD_CG.X.shape[0] ) # the number of changeable elements successful MC moves at each step, for fixed pi remove diagonals
    else:
        NMC_p_Sweep = int( len(np.where(SD_CG.X>1e-12)[0]) ) # the number of changeable elements successful MC moves at each step
elif ( ID.MC_type == MC_types[1] ): # if MC_type == 'corr_move' 
    NMC_p_Sweep = ID.corr_step_len # For this method, this variable now corresponds to the max # of corr steps in the "lower" MC procedure.
else:
    raise ValueError('MC_type not supported!  Check input_data.py')

NSweep_p_Samp = deepcopy(ID.NSweep_p_Samp) # sample in multiples of this number of sweeps
NSweep_eq = deepcopy(ID.NSweep_eq) # throw out this many sweeps before collecting statistics

lamb_b = deepcopy(ID.lamb0[rank])
init_interp = deepcopy(ID.init_interp[rank])
beta1 = np.abs(EQ_ref - SD_CG.EQ) # This value is necessary to recover unbiased sampling with lamb = 1
beta0 = ID.beta0C * deepcopy(SD_CG.EW) # Equivalent of the above, for lamb = 0 but does not correspond to any particular variance
beta_b = (ID.beta_interp)*beta1 + (1.0-ID.beta_interp)*beta0 # linear interpolation of beta, as specified in input_data.py

if ( ID.OPT_type == OPT_types[0] ): # OPT_type == 'fixed_lamb'        
    donothing = True
elif ( ID.OPT_type == OPT_types[1] ): # OPT_type == 'slow_growth' 
    dlamb = deepcopy(ID.dlamb0) # starting change in lambda
elif ( ID.OPT_type == OPT_types[2] ): # OPT_type == 'rep_exch'
    donothing = True
else:
    raise ValueError('OPT_type not supported!  Check input_data.py')

if ( rank == 0 ):
    # print some info to the screen
    print 'temperature and constraint info:'
    print 'beta1 = '+str(beta1)
    print 'beta0 = '+str(beta0)
    print 'beta = '+str(beta_b)
    print 'EW_CG = '+str(SD_CG.EW)
    print '\n'

# save the mle variables
if (rank == 0):
    SD_AA._save_output()
    SD_CG._save_output()

# variables for the sampling
SD_CG_b = SD.SampData()
SD_CG_b.outfnm = ID.outfnm+'_CG_b_'+str(rank)
#SD_CG_b.bins = deepcopy(bin_ctrs_CG_cc)

# Choose an initial model, by interpolating between the CG mle and the AA map'd
T_init = init_interp*SD_CG.T + (1.0-init_interp)*T_AA_mapd
X_init = init_interp*SD_CG.X + (1.0-init_interp)*X_AA_mapd

# Restarting options: for now just use the final model as init
if ( ID.flag_cont ):
    data = np.load(ID.cont_pth+'output_variables_CG_b_'+str(rank)+'.npz')
    Ndat = data['lamb'].shape[0]
    T_init = deepcopy( data['T'][Ndat-1] )
    X_init = deepcopy( data['X'][Ndat-1] )


''' JFR - changed init_interp so that you can specify the different values by hand now, for rx or other
# Particular variable settings for different OPT procedures
if ( ID.OPT_type == OPT_types[0] ): # OPT_type == 'fixed_lamb'        
    donothing = True
elif ( ID.OPT_type == OPT_types[1] ): # OPT_type == 'slow_growth' 
    dlamb = deepcopy(ID.dlamb0) # starting change in lambda
elif ( ID.OPT_type == OPT_types[2] ): # OPT_type == 'rep_exch'
    T_init = lamb_b*SD_CG.T + (1.0-lamb_b)*T_AA_mapd # interpolate based on the rep value of lamb
    X_init = lamb_b*SD_CG.X + (1.0-lamb_b)*X_AA_mapd
else:
    raise ValueError('OPT_type not supported!  Check input_data.py')
'''

# Special check for AA unbiased sampling
if ( ID.flag_ub ):
    SD_CG.EQ = 0.0
    SD_CG.EW = 1.0
    beta_b = 1.0
    EQ_ref = 1.0
    lamb_b = 1.0

# calculate the initial energies of the biased model
EQ_CG_b_init = -1.0*( logprob_T(Cmat_CG_cc+prior, T_init) )
EW_CG_b_init = FW_fT(T_init)
if ( rank == 0 ):
    # print out some info to the screen
    print 'AA mle energies:'
    print 'EQ_AA = '+str( (SD_AA.EQ - SD_CG.EQ) / np.abs(EQ_ref - SD_CG.EQ) )+', EW_AA = '+str(SD_AA.EW/SD_CG.EW)
    print 'AA mle mapd energies:'
    print 'EQ_AA_mapd = '+str( (EQ_AA_mapd - SD_CG.EQ) / np.abs(EQ_ref - SD_CG.EQ) )+', EW_AA_mapd = '+str(F_AA_mapd/SD_CG.EW)
    print 'CG mle energies:'
    print 'EQ_CG = '+str( (SD_CG.EQ - SD_CG.EQ) / np.abs(EQ_ref - SD_CG.EQ) )+', EW_CG = '+str(SD_CG.EW/SD_CG.EW)
    print 'initial CG_b energies:'
    print 'EQ_CG_b = '+str( (EQ_CG_b_init - SD_CG.EQ) / np.abs(EQ_ref - SD_CG.EQ) )+', EW_CG_b = '+str(EW_CG_b_init/SD_CG.EW)
    print '\n'

# Start the sampling
if ( rank == 0 ):
    print 'starting the equilibration sweeps...'
    print '\n'

if (NSweep_eq != 0): # initialization sweep
    sampler_CG_b.sample(NMC_p_Sweep, T_init = T_init, X_init = X_init, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
for i in range (1, NSweep_eq):
    sampler_CG_b.sample(NMC_p_Sweep, T_init = sampler_CG_b.T, X_init = sampler_CG_b.X, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
    if (i % 100 == 0):
        if ( rank == 0 ):
            print 'On equil sweep '+str(i)+' of '+str(NSweep_eq)
            sys.stdout.flush() # This is so the output can be seen on the fly with nohup, it may slow things down

if ( rank == 0 ):
    print 'starting the production sweeps...'
    print '\n'

# now, get ready for the production sweeps
Sweep_ctr = 0      # counts sweeps
Samp_ctr = 0       # counts samples
flag_Econv = False # E convergence indicator
flag_term = False  # optimization termination indicator
NSamp_t_lev = 0    # counts samples at current lambda value (for OPT_type == 'slow_growth')
NSamp_l_lev = 0    # N samples at the previous lambda value (for OPT_type == 'slow_growth')
flag_forw = True   # keeps track of the direction of lambda movement (for OPT_type == 'slow_growth')
Nhrnd = 0          # keeps track of the number of half rounds (for OPT_type == 'slow_growth')

# initialization sweep
if (NSweep_eq != 0): 
    sampler_CG_b.sample(NMC_p_Sweep, T_init = sampler_CG_b.T, X_init = sampler_CG_b.X, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
else:
    sampler_CG_b.sample(NMC_p_Sweep, T_init = T_init, X_init = X_init, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
Sweep_ctr += 1
SD_CG_b._init(sampler_CG_b)
SD_CG_b.ts = np.array([deepcopy( est_ts_kfT(sampler_CG_b.T) )])
SD_CG_b.mfpt = np.array([deepcopy( mfpt_all_fT(sampler_CG_b.T) )])

# test acc prop
#print 'nacc_pdE = '+str(sampler_CG_b.nacc_pdE)
#print 'nprop_pdE = '+str(sampler_CG_b.nprop_pdE)
#print 'nprop_ndE = '+str(sampler_CG_b.nprop_ndE)
#if ( sampler_CG_b.nprop_ndE+sampler_CG_b.nprop_pdE != 0 ):
#    den = sampler_CG_b.nprop_ndE+sampler_CG_b.nprop_pdE
#else:
#    den = 1.0
#print 'pacc_tot = '+str( float(sampler_CG_b.nprop_ndE+sampler_CG_b.nacc_pdE) / float(den) )
#if ( sampler_CG_b.nprop_pdE != 0 ):
#    den = sampler_CG_b.nprop_pdE
#else:
#    den = 1.0
#print 'pacc_pdE = '+str( float(sampler_CG_b.nacc_pdE) / float(den) )


while (flag_term is False):
    # perform a sweep
    sampler_CG_b.sample(NMC_p_Sweep, T_init = sampler_CG_b.T, X_init = sampler_CG_b.X, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
    Sweep_ctr += 1
    # store the data
    SD_CG_b._update(sampler_CG_b)
    SD_CG_b.ts = np.append( SD_CG_b.ts, np.array([deepcopy( est_ts_kfT(sampler_CG_b.T) )]), axis=0 )
    SD_CG_b.mfpt = np.append( SD_CG_b.mfpt, np.array([deepcopy( mfpt_all_fT(sampler_CG_b.T) )]), axis=0 )

    if ((Sweep_ctr % NSweep_p_Samp) == 0):
        # update the sample ctrs
        Samp_ctr += 1
        NSamp_t_lev += 1
        if (NSamp_t_lev > ID.NSamp_max_conv): # only consider convergence over the most recent ID.NSamp_conv samples
            NSamp_t_lev = deepcopy(ID.NSamp_max_conv)
        # get the indices for the samples
        i0 = Sweep_ctr - NSamp_t_lev*ID.NSweep_p_Samp
        ih = Sweep_ctr - (NSamp_t_lev*ID.NSweep_p_Samp)/2 # NSweep_p_Samp must be divisible by 2!
        # calculate the avg energies for this sample
        EQ_avg = np.array(Eavg(SD_CG_b.EQ[i0:]))
        EQ_avg_tot = np.zeros(1)
        EW_avg = np.array(Eavg(SD_CG_b.EW[i0:]))
        EW_avg_tot = np.zeros(1)
        # calculate the fractional change in energy within the sample
        EQ_frac_win = np.array( calc_frac_E_change( SD_CG_b.EQ, i0, ih, Sweep_ctr ) )
        EQ_frac_win_tot = np.zeros(1)
        EW_frac_win = np.array( calc_frac_E_change( SD_CG_b.EW, i0, ih, Sweep_ctr ) )
        EW_frac_win_tot = np.zeros(1)
        # acceptance ratios
        nacc_pdE = np.array( float(sampler_CG_b.nacc_pdE) )
        nacc_pdE_tot = np.zeros(1)
        nprop_pdE = np.array( float(sampler_CG_b.nprop_pdE) )
        nprop_pdE_tot = np.zeros(1)
        nprop_ndE = np.array( float(sampler_CG_b.nprop_ndE) )
        nprop_ndE_tot = np.zeros(1)

        if ( (ID.OPT_type==OPT_types[1]) or ( (ID.OPT_type== OPT_types[0]) and (len( np.where( ID.lamb0 == ID.lamb0[0] )[0] ) == size) ) ): 
        # OPT_type == 'slow_growth' or OPT_type == 'fixed_lamb' with all the same lambda 
            # evaluate the energy convergence together, combine the statistics from all MPI processes
            comm.Allreduce(EQ_avg, EQ_avg_tot, op=MPI.SUM)
            comm.Allreduce(EW_avg, EW_avg_tot, op=MPI.SUM)
            comm.Allreduce(EQ_frac_win, EQ_frac_win_tot, op=MPI.SUM)
            comm.Allreduce(EW_frac_win, EW_frac_win_tot, op=MPI.SUM)
            comm.Allreduce(nacc_pdE, nacc_pdE_tot, op=MPI.SUM)
            comm.Allreduce(nprop_pdE, nprop_pdE_tot, op=MPI.SUM)
            comm.Allreduce(nprop_ndE, nprop_ndE_tot, op=MPI.SUM)
            # get avgs
            EQ_avg_tot /= float(size)
            EW_avg_tot /= float(size)
            EQ_frac_win_tot /= float(size)
            EW_frac_win_tot /= float(size) 
        else: # OPT_type == 'rep_exch' or OPT_type == 'slow_growth' with different lambdas
            # evaluate the energy convergence separately
            EQ_avg_tot = deepcopy(EQ_avg)
            EW_avg_tot = deepcopy(EW_avg)
            EQ_frac_win_tot = deepcopy(EQ_frac_win)
            EW_frac_win_tot = deepcopy(EW_frac_win)
            nacc_pdE_tot = deepcopy(nacc_pdE)
            nprop_pdE_tot = deepcopy(nprop_pdE)
            nprop_ndE_tot = deepcopy(nprop_ndE)

        if ( (ID.OPT_type==OPT_types[1]) and (NSamp_t_lev == 1) and (Samp_ctr != 1) ): # for OPT_type == 'slow_growth', on each first sample of the level, but not the first sample overall
            # check that the energies did not jump too much from the last Samp
            # get the indices for the samples
            i0_l = i0 - NSamp_l_lev*ID.NSweep_p_Samp
            # calculate the fractional change in energy between samples
            EQ_frac_btn = np.array( calc_frac_E_change( SD_CG_b.EQ, i0_l, i0, Sweep_ctr ) )
            EQ_frac_btn_tot = np.zeros(1)
            EW_frac_btn = np.array( calc_frac_E_change( SD_CG_b.EW, i0_l, i0, Sweep_ctr ) )
            EW_frac_btn_tot = np.zeros(1)
            # combine the statistics from all MPI processes
            comm.Allreduce(EQ_frac_btn, EQ_frac_btn_tot, op=MPI.SUM)
            comm.Allreduce(EW_frac_btn, EW_frac_btn_tot, op=MPI.SUM)
            # get avgs
            EQ_frac_btn_tot /= float(size)
            EW_frac_btn_tot /= float(size)
            #
            flag_jump, flag_stuck = check_Eavg_bw_Samps( EQ_frac_btn_tot, EW_frac_btn_tot, EW_avg_tot, ID )
            if ( flag_jump or flag_stuck ): # go back to the end of the previous sample and redo with diff change in lamb
                SD_CG_b._remove( i0, Sweep_ctr, sampler_CG_b )
                if ( flag_jump ): mult = ID.mult_jump
                else: mult = ID.mult_stuck
                dlamb *= mult
                lamb_b = deepcopy(sampler_CG_b.lamb)
                if ( np.abs(dlamb/lamb_b) > ID.dlamb_frac_max ):
                    dlamb = (dlamb/np.abs(dlamb))*ID.dlamb_frac_max*lamb_b
                elif ( np.abs(dlamb/lamb_b) < ID.dlamb_frac_min ):
                    dlamb = (dlamb/np.abs(dlamb))*ID.dlamb_frac_min*lamb_b
                lamb_b = update_lamb( lamb_b, dlamb )
                Sweep_ctr -= NSweep_p_Samp
                NSamp_t_lev -= 1
                Samp_ctr -= 1
                continue

        # now, check if E is converged or max number of samples exceeded
        flag_Econv = is_E_conv(EQ_frac_win_tot, EW_frac_win_tot, EW_avg_tot, ID)
        if ( NSamp_t_lev >= ID.NSamp_p_lam_max ):
            flag_Econv = True

        if ( ID.OPT_type==OPT_types[0] ): # OPT_type == 'fixed_lamb'
            if ( (len( np.where( ID.lamb0 == ID.lamb0[0] )[0] ) == size) and (flag_Econv) ): # all proc have the same lamb => we already combined the statistics
                flag_term = True
            else: # need to check that each processor is converged separately
                flag_Econv_cf = comm.allgather(flag_Econv)
                flag_term = all(flag_Econv_cf)
        elif ( (ID.OPT_type==OPT_types[1]) and (flag_Econv) ): # OPT_type == 'slow_growth'
            if ( is_halfround_comp( flag_forw, ID.EWtol, EW_avg_tot, lamb_b, ID.lamb0) ): # check if you have reach the end of a "half round"
                Nhrnd += 1
                dlamb *= -1.0
                if ( flag_forw ): flag_forw = False
                else: flag_forw = True
            if ( Nhrnd == ID.Nhrnd ):
                flag_term = True
            lamb_b = update_lamb( lamb_b, dlamb )
            NSamp_l_lev = deepcopy(NSamp_t_lev)
            NSamp_t_lev = 0
            if ( not flag_term ): # equilibration steps after changing lambda
                # Equilibration
                if (NSweep_eq != 0):
                    if ( rank == 0 ):
                        print 'starting the equilibration sweeps between lambdas...'
                        print '\n'
                    sampler_CG_b.sample(NMC_p_Sweep, T_init = sampler_CG_b.T, X_init = sampler_CG_b.X, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
                for i in range (1, NSweep_eq):
                    sampler_CG_b.sample(NMC_p_Sweep, T_init = sampler_CG_b.T, X_init = sampler_CG_b.X, EQ_CG = SD_CG.EQ, EQ_AA = EQ_ref, F_fun = FW_fT, F_CG = SD_CG.EW, beta = beta_b, lamb = lamb_b, fixed_pi = ID.fixed_pi)
                    if (i % 100 == 0):
                        if ( rank == 0 ):
                            print 'On equil sweep '+str(i)+' of '+str(NSweep_eq)
                            sys.stdout.flush() # This is so the output can be seen on the fly with nohup, it may slow things down
        elif ( ID.OPT_type==OPT_types[2] ): # OPT_type == 'rep_exch'
            # need to check that each processor is converged separately
            flag_Econv_cf = comm.allgather(flag_Econv)
            flag_term = all(flag_Econv_cf)

        # print the status every sample
        if (rank == 0):
            print_status(Sweep_ctr, sampler_CG_b.lamb, EQ_avg_tot, EW_avg_tot, EQ_frac_win_tot, EW_frac_win_tot, nacc_pdE_tot, nprop_pdE_tot, nprop_ndE_tot)
            sys.stdout.flush() # This is so the output can be seen on the fly with nohup, it may slow things down

    # save data every Sweep
    SD_CG_b._save_output()

    if ( (ID.OPT_type==OPT_types[2]) and ((Sweep_ctr % ID.NSweep_p_Swap_rx) == 0) ): # OPT_type == 'rep_exch' and time to swap
        # first get all the data
        EW_cf = comm.gather(SD_CG_b.EW[Sweep_ctr-1], root=0)
        EQ_cf = comm.gather(SD_CG_b.EQ[Sweep_ctr-1], root=0)
        lamb_cf = comm.gather(SD_CG_b.lamb[Sweep_ctr-1], root=0)
        # then, try swapping
        if ( rank == 0 ):
            rx_swaps( EW_cf, EQ_cf, lamb_cf, rep_ind_perm, ID.beta_rx_fact*beta_b )
        # now, give back the lambdas
        lamb_b = comm.scatter(lamb_cf, root=0) # send back the values either way

print 'ending process '+str(rank)+'...'


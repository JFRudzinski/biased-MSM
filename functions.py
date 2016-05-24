import unittest
import warnings

import numpy as np
# stuff from pyemma
import pyemma
from pyemma.msm.analysis import eigenvalues
from pyemma.msm.analysis import mfpt

# other stuff
from copy import copy, deepcopy
import random


##### GRID FUNCTIONS #####

# for discretization along two order parameters, here are some functions for keeping track of the bin_centers
# functions for transforming between bins and labels
def values_to_bins(x,y,xmin,ymin,dx,dy):
    xgrid = int( np.floor( (x - xmin) / dx ) )
    ygrid = int( np.floor( (y - ymin) / dy ) )
    return xgrid, ygrid

def bins_to_label(xgrid,ygrid,nx):
    label = xgrid + ygrid*nx
    return label

def label_to_bins(label,nx):
    xgrid = label % nx
    ygrid = (label - xgrid) / nx
    return xgrid, ygrid

def bins_to_values(xgrid, ygrid, values):
    values_x = values[0][xgrid]
    values_y = values[1][ygrid]
    return values_x, values_y

def dtraj_2D_to_1D( bin_ctrs, dtraj ):
    # define some variables dealing with the bins
    ngridsx = len(bin_ctrs[0])
    xmin = min(bin_ctrs[0])
    dx = bin_ctrs[0][1] - bin_ctrs[0][0]
    ngridsy = len(bin_ctrs[1])
    ymin = min(bin_ctrs[1])
    dy = bin_ctrs[1][1] - bin_ctrs[1][0]
    # convert dtraj to 1D labels
    dtraj_labels = deepcopy(dtraj[:,:,0])
    xgrid_traj = deepcopy(dtraj[:,:,0])
    ygrid_traj = deepcopy(dtraj[:,:,0])
    for i in range(0,dtraj_labels.shape[0]):
        for j in range(0,dtraj_labels.shape[1]):
            xgrid_traj[i,j] = int(np.floor( (dtraj[i,j,0] - xmin) / dx ) )
            ygrid_traj[i,j] = int(np.floor( (dtraj[i,j,1] - ymin) / dy ) )
            index = xgrid_traj[i,j] + ygrid_traj[i,j]*ngridsx
            dtraj_labels[i,j] = index
    dtraj_labels = dtraj_labels.astype(int)
    dtraj_labels = dtraj_labels.tolist()
    xgrid_traj = xgrid_traj.astype(int)
    xgrid_traj = xgrid_traj.tolist()
    ygrid_traj = ygrid_traj.astype(int)
    ygrid_traj = ygrid_traj.tolist()

    return dtraj_labels, xgrid_traj, ygrid_traj

# trim the count matrix and bins to ignore low-sampled states
# nb - this function changed from 1D to 2D grids!
def trim_Cmat( Cmat, lcc, ID ):
    minsamp = ID.trimfrac * np.sum(Cmat, dtype=float) / float(lcc.size)
    nrem = 0
    for i in range(0,lcc.size):
        shift = i - nrem
        if ( np.sum(Cmat[shift], dtype=float) < minsamp ): # trim from matrix and bins
            Cmat = np.delete(Cmat, (shift), axis=0)
            Cmat = np.delete(Cmat, (shift), axis=1)
            lcc = np.delete(lcc, (shift))
            nrem += 1
    # reensure that the trimmed matrix is connected!
    lcc_tmp = pyemma.msm.estimation.largest_connected_set(Cmat, directed=True)
    Cmat_cc = pyemma.msm.estimation.largest_connected_submatrix(Cmat, directed=True, lcc=lcc_tmp)
    lcc = lcc[lcc_tmp]
    return Cmat_cc, lcc

# 'project' the AA mle onto the CG bins
def project_X0_to_Xf(Xf, X0, binsf, bins0):
    X0p = deepcopy(Xf)
    for i in range(0,binsf.size):
        for j in range(0,binsf.size):
            if ( (binsf[i] in bins0) and (binsf[j] in bins0) ):
                iAA = [k for k, l in enumerate(bins0) if l==binsf[i]][0]
                jAA = [k for k, l in enumerate(bins0) if l==binsf[j]][0]
                X0p[i,j] = X0[iAA,jAA]
            else:
                X0p[i,j] = 1e-6 # zero leads to problems for non-overlapping bins
    return X0p


##### T AND X FUNCTIONS #####

# define the transformation from T to Copt via mu
def T_to_X (C, T, mu):
    X = np.dot(np.diag(mu), T)
    X *= ( (X.shape[0]*X.shape[1])/np.sum(X, dtype=float) )
    return X


def X_to_T(X):
    return  ( X / np.sum(X, dtype=float, axis=1, keepdims=True) )


##### OBSERVABLE FUNCTIONS #####

# define the biasing functions
def est_ts_k ( T, tau, k ):
    ts_est = eigenvalues(T,k=k+1)
    ts_est = np.real(ts_est[1:k+1])
    ts_est = -tau / np.ma.log(ts_est)
    return np.array(ts_est)

def get_msstate_grids( lcc_trim, bin_ctrs, S, ngridsx ):

    # get the grids for each set
    xgrids, ygrids = label_to_bins(lcc_trim[S],ngridsx)
    bin_ctrs_x, bin_ctrs_y = bins_to_values(xgrids, ygrids, bin_ctrs)

    return bin_ctrs_x, bin_ctrs_y

def proj_msstates_to_CG_bins( lcc_trim_AA, lcc_trim_CG, S ):
    # nb - plot added at end of file...
    # define the metastable states in terms of the CG grids
    S_CG = [k for k, l in enumerate(lcc_trim_CG) if (l in lcc_trim_AA[S]) ]

    return S_CG

def mfpt_all(T, mss):
    Nmss = mss.shape[0]
    m_t = []
    for i in range(Nmss):
        for j in range(Nmss):
            if ( i != j ):
                m_t.append( mfpt(T,mss[j],origin=mss[i]) )
    return np.array(m_t)

def rmse_mfpt_all(T, mfpt_AA, mss, nconst, flag_rel):
    Nmss = mss.shape[0]
    mfpt_CG = mfpt_all(T,mss)
    max_ind = np.where( mfpt_AA == max(mfpt_AA) )[0]
    rmse = 0.0
    if (flag_rel): # use ratious relative to the max mfpt
        for i in range(nconst):
            if ( i != max_ind ):
                rAA = mfpt_AA[i] / mfpt_AA[max_ind]
                rCG = mfpt_CG[i] / mfpt_CG[max_ind]
                relsqerr = (rAA - rCG)**2
                relsqerr /= (rAA)**2
                rmse += relsqerr
        rmse = np.sqrt(rmse)
    else: # calculate the direct rmse
        for i in range(nconst):
            rAA = mfpt_AA[i]
            rCG = mfpt_CG[i]
            relsqerr = (rAA - rCG)**2
            relsqerr /= (rAA)**2
            rmse += relsqerr
        rmse = np.sqrt(rmse)
    return rmse

def regulate_constraint(T, A, nu, re):
    tmp = T**nu
    tmp = np.sum(tmp,axis=1)
    tmp = np.exp( - re / (tmp) )
    tmp = A * np.max(tmp)
    return tmp


##### STATUS FUNCTIONS #####

# define some functions to clean up the sampling section
def print_status_OLD(step, ID, sampler, SD):
    print 'on sweep # '+str(step)
    print 'lamb = '+str(sampler.lamb)
    print 'average energies from this sample:' 
    i0 = step - ID.NSweep_p_Samp
    EQ_avg = Eavg(SD.EQ[i0:])
    EW_avg = Eavg(SD.EW[i0:])
    print '<EQ> = '+str(EQ_avg)+', <EW> = '+str(EW_avg)
    print 'acc ratio pos dE = '+str( float(sampler.nacc_pdE) / float(sampler.nprop_pdE) )
    print 'acc ratio total = '+str( float(sampler.nacc_pdE+sampler.nprop_ndE) / float(sampler.nprop_pdE+sampler.nprop_ndE) )
    print '\n'

def print_status(step, lamb, EQ_avg, EW_avg, EQ_frac, EW_frac, nacc_pdE, nprop_pdE, nprop_ndE):
    print 'on sweep # '+str(step)
    print 'lamb = '+str(lamb)
    print 'average energies from this sample:'
    print '<EQ> = '+str(EQ_avg)+', <EW> = '+str(EW_avg)
    print 'with cumm. fractional half differences of:'
    print 'EQ_frac = '+str(EQ_frac)+', EW_frac = '+str(EW_frac)
    if ( nprop_ndE+nprop_pdE != 0 ):
        den = nprop_ndE+nprop_pdE
    else:
        den = 1.0
    print 'acc ratio total = '+str( float(nacc_pdE+nprop_ndE) / float(den) )
    if ( nprop_pdE != 0 ):
        den = nprop_pdE
    else:
        den = 1.0
    print 'acc ratio pos dE = '+str( float(nacc_pdE) / float(den) )
    print '\n'


##### SLOW GROWTH FUNCTIONS #####

# for updating lambda
def update_lamb( lamb, dlamb ):
    lamb -= dlamb
    if ( lamb < 0.0 ):
        lamb = 0.0
    elif ( lamb > 1.0 ):
        lamb = 1.0
    return lamb


##### ENERGY FUNCTIONS #####

# Energy functions for the sampling
def logprob_T(C, T):
    assert np.all(T >= 0)
    return np.sum(np.multiply(C, np.ma.log(T)), dtype=float) # JFR-nb: avoid 0 elements in the log arg with np.ma.log

# functions for sampling convergence
def Eavg( E ):
    return (np.sum(E)/float(len(E)))

def calc_frac_E_change( E, i0, ih, it ):
    Eavgh1 = Eavg(E[i0:ih])
    Eavgh2 = Eavg(E[ih:it])
    return ( np.abs(Eavgh1-Eavgh2)/np.abs(Eavgh1) )

def check_Eavg_bw_Samps_OLD( SD, ID, Sweep_ctr, Nsamp_t_lev, Nsamp_l_lev ):
    i0_l = Sweep_ctr - Nsamp_l_lev*ID.NSweep_p_Samp
    i0_t = Sweep_ctr - Nsamp_t_lev*ID.NSweep_p_Samp
    EQ_frac = calc_frac_E_change( SD.EQ, i0_l, i0_t, Sweep_ctr )
    EW_frac = calc_frac_E_change( SD.EW, i0_l, i0_t, Sweep_ctr )
    flag_jump = False
    flag_stuck = False
    if ( (EQ_frac>ID.Ejumpfrac) or (EW_frac>ID.Ejumpfrac) ):
        flag_jump = True
        print 'The energies have jumped!  EQ_frac = '+str(EQ_frac)+', EW_frac = '+str(EW_frac)
        print 'Redoing this Sample with smaller dlamb.'
    elif ( (EQ_frac<ID.Estuckfrac) and (EW_frac<ID.Estuckfrac) ):
        flag_stuck = True
        print 'The energies are stuck!  EQ_frac = '+str(EQ_frac)+', EW_frac = '+str(EW_frac)
        print 'Redoing this Sample with larger dlamb.'
    return flag_jump, flag_stuck

def check_Eavg_bw_Samps( EQ_frac, EW_frac, EW_avg, ID ):
    flag_jump = False
    flag_stuck = False
    #if ( (EQ_frac>ID.Ejumpfrac) or ((EW_frac>ID.Ejumpfrac) and (EW_avg>ID.EWtol)) ):
    if ( EQ_frac>ID.Ejumpfrac ):
        flag_jump = True
        print 'The energies have jumped!  EQ_frac = '+str(EQ_frac)+', EW_frac = '+str(EW_frac)
        print 'Redoing this Sample with smaller dlamb.'
    #elif ( (EQ_frac<ID.Estuckfrac) and ((EW_frac<ID.Estuckfrac) and (EW_avg>ID.EWtol)) ):
    elif ( EQ_frac<ID.Estuckfrac ):
        flag_stuck = True
        print 'The energies are stuck!  EQ_frac = '+str(EQ_frac)+', EW_frac = '+str(EW_frac)
        print 'Redoing this Sample with larger dlamb.'
    return flag_jump, flag_stuck

def is_E_conv_OLD( SD, ID, Ntot, Nsamp_t_lev ):
    i0 = Ntot - Nsamp_t_lev*ID.NSweep_p_Samp
    #print 'i0 = '+str(i0)
    ih = Ntot - (Nsamp_t_lev*ID.NSweep_p_Samp)/2 # NSweep_p_Samp must be divisible by 2!
    #print 'ih = '+str(ih)
    EQ_frac = calc_frac_E_change( SD.EQ, i0, ih, Ntot )
    #print 'EQ_frac = '+str(EQ_frac)
    EW_frac = calc_frac_E_change( SD.EW, i0, ih, Ntot )
    #print 'EW_frac = '+str(EW_frac)
    if ( (EQ_frac<ID.Econvfrac) and (EW_frac<ID.Econvfrac) ):
        return True
    else:
        return False

def is_E_conv( EQ_frac, EW_frac, EW_avg, ID ):
    if ( ID.flag_ub ):
        if ( (EQ_frac<ID.Econvfrac) ):
            return True
        else:
            return False
    else:
        #if ( (EQ_frac<ID.Econvfrac) and ((EW_frac<ID.Econvfrac) or (EW_avg<ID.EWtol)) ):
        if ( EQ_frac<ID.Econvfrac ):
            return True
        else:
            return False

def is_halfround_comp_OLD( flag_forw, i0, tol, EW, lamb):
    if (flag_forw):
        if ( Eavg(EW) < tol ):
            return True
    else:
        if ( np.abs(lamb-1.0) < 1e-6 ):
            return True
    return False

def is_halfround_comp( flag_forw, tol, EW_avg, lamb, lamb0):
    if (flag_forw):
        if ( EW_avg < tol ):
            print 'Finished half round (forward), EW_avg = '+str(EW_avg)+' < '+str(tol)+'= tol'
            print '\n'
            return True
    else:
        if ( (np.abs(lamb-1.0) < 1e-6) or (lamb > np.max(lamb0)) ):
            print 'Finished half round (backward), lamb = '+str(lamb)
            print '\n'
            return True
    return False


##### REPLICA EXCHANGE FUNCTIONS #####

def rx_swaps( EW_cf, EQ_cf, lamb_cf, rep_ind_perm, beta_b ):
    # determine the swapping pairs (neighbors) one by one until none are left
    rep_ind_tmp = deepcopy(rep_ind_perm)
    while ( len(rep_ind_tmp) != 0 ):
        i_tmp = random.randint(0,len(rep_ind_tmp)-1) # choose one of the remaining available procs
        i_proc = rep_ind_tmp[i_tmp] # get the OG proc number (i.e., lambda index)
        i_ind = np.where(rep_ind_perm == i_proc)[0][0] # get the curr perm index of that process
        jpos = [] # store available adjacent processes
        if ( (i_proc-1) in rep_ind_tmp ):
            jpos.append(i_proc-1)
        if ( (i_proc+1) in rep_ind_tmp ):
            jpos.append(i_proc+1)
        if ( len(jpos) != 0 ):  # if there exists a free neighbor, choose at randoom
            j_tmp = random.randint(0,len(jpos)-1) # choose a swap partner from the possibilites at random
            j_proc = jpos[j_tmp] # get the process 
            j_tmp = np.where(rep_ind_tmp == j_proc)[0][0] # get the curr ind of the process
        else:
            j_proc = -1

        if ( j_proc != -1 ):
            j_ind = np.where(rep_ind_perm == j_proc)[0][0] # get the curr perm index of the process
            Delta = (lamb_cf[i_ind] - lamb_cf[j_ind])*(EQ_cf[j_ind] - EQ_cf[i_ind]) + (lamb_cf[j_ind] - lamb_cf[i_ind])*(EW_cf[j_ind] - EW_cf[i_ind])
            if ( Delta < 0 ):
                pacc = 1.0
            else:
                pacc = np.exp( -beta_b*Delta )
                #pacc = np.exp( -Delta )

            #print 'attempting swap lamb = '+str(lamb_cf[i])+' to lamb = '+str(lamb_cf[j])+' pacc = '+str(pacc)
            #print 'Delta = '+str(Delta)
            #print 'lamb_cf[i] - lamb_cf[j] = '+str(lamb_cf[i] - lamb_cf[j])
            #print 'EQ_cf[j] - EQ_cf[i] = '+str(EQ_cf[j] - EQ_cf[i])
            #print 'EW_cf[j] - EW_cf[i] = '+str(EW_cf[j] - EW_cf[i])

            if( np.random.uniform(0,1,1) < pacc ): # accept the move
                tmp_i = deepcopy(lamb_cf[i_ind])
                lamb_cf[i_ind] = deepcopy(lamb_cf[j_ind])
                lamb_cf[j_ind] = deepcopy(tmp_i)
                tmp_i = deepcopy(rep_ind_perm[i_ind])
                rep_ind_perm[i_ind] = deepcopy(rep_ind_perm[j_ind])
                rep_ind_perm[j_ind] = deepcopy(tmp_i)

            rep_ind_tmp = np.delete(rep_ind_tmp,(i_tmp,j_tmp))
        else:
            rep_ind_tmp = np.delete(rep_ind_tmp,(i_tmp))
    #print 'rep_ind_perm = '+str(rep_ind_perm)
    #print 'lamb_cf = '+str(lamb_cf)
    #print 'done with rx_swaps!'




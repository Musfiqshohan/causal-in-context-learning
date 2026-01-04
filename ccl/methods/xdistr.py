#!/usr/bin/env python3.6
""" Modification to the `distr` package for the structure of the
    Causal Semantic Generative model.
"""
import sys
import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from distr import Distr, edic

import wandb

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def elbo_z2xy(p_x1z: Distr, p_z, p_y1z: Distr, q_z1x: Distr, obs_xy: edic, n_mc: int=0, repar: bool=True) -> tc.Tensor:
    """ For supervised VAE with structure x <- z -> y.
    Observations are supervised (x,y) pairs.
    For unsupervised observations of x data, use `elbo(p_zx, q_z1x, obs_x)` as VAE z -> x. """
    if n_mc == 0:
        log_phi_y1x = q_z1x.expect(lambda dc: p_y1z.logp(dc, dc), obs_xy, 1, repar=True)
        expc_val_x = q_z1x.expect(lambda dc: p_x1z.logp(dc, dc), obs_xy, 1, repar=True)
        expc_val_z = q_z1x.expect(lambda dc: p_z.logp(dc, dc) - q_z1x.logp(dc, dc), obs_xy, 1, repar=True)
    return log_phi_y1x, expc_val_x, expc_val_z

def ccl_elbo(qt_z1x: Distr, p_y1c: Distr, p_x1cs: Distr, p_e1s: Distr, p_z: Distr, p_c1t: Distr, p_s: Distr, q_c1x: Distr, q_s1x: Distr,
             obs_xyte: edic, n_mc: int=1, recon: bool=False, mse_loss: nn.MSELoss=None) -> tc.Tensor:
    
    if n_mc == 0:
        # log_phi_y1xte = qt_z1x.expect(lambda dc: p_y1c.logp(dc, dc), obs_xyte, 0, repar=True)
        
        # p_x1cs
        expc_val_x = qt_z1x.expect(lambda dc: p_x1cs.logp(dc, dc), obs_xyte, 0, repar=True)
        # p_e1s
        expc_val_e = qt_z1x.expect(lambda dc: p_e1s.logp(dc, dc), obs_xyte, 0, repar=True)
        # p_c1t - q_c1x
        expc_val_c = qt_z1x.expect(lambda dc: p_c1t.logp(dc, dc) - q_c1x.logp(dc, dc), obs_xyte, 1, repar=True)
        # p_s - q_s1x
        expc_val_s = qt_z1x.expect(lambda dc: p_s.logp(dc, dc) - q_s1x.logp(dc, dc), obs_xyte, 1, repar=True)

        # return log_phi_y1xte, expc_val_x, expc_val_e, expc_val_c, expc_val_s
        return expc_val_x, expc_val_e, expc_val_c, expc_val_s
    
def ccl_adapt(qt_z1x: Distr, q_c1x: Distr, q_s1x: Distr, p_x1cs: Distr, p_e1s: Distr, p_c1t: Distr, p_s: Distr, obs_xyte: edic, n_mc: int=1, recon: bool=False, mse_loss: nn.MSELoss=None) -> tc.Tensor:
    if n_mc == 0:
        # expc_val_x = p_x1cs.logp(obs_xyte, obs_xyte)
        # expc_val_e = p_e1s.logp(obs_xyte, obs_xyte)
        expc_val_x = qt_z1x.expect(lambda dc: p_x1cs.logp(dc, dc), obs_xyte, 0, repar=True)
        expc_val_e = qt_z1x.expect(lambda dc: p_e1s.logp(dc, dc), obs_xyte, 0, repar=True)
        # p_c1t - q_c1x
        expc_val_c = qt_z1x.expect(lambda dc: p_c1t.logp(dc, dc) - q_c1x.logp(dc, dc), obs_xyte, 1, repar=True)
        # p_s - q_s1x
        expc_val_s = qt_z1x.expect(lambda dc: p_s.logp(dc, dc) - q_s1x.logp(dc, dc), obs_xyte, 1, repar=True)
        
        return expc_val_x, expc_val_e, expc_val_c, expc_val_s
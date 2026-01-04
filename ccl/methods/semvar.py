#!/usr/bin/env python3.6
''' The Semantic-Variation Generative Model.

I.e., the proposed Causal Semantic Generative model (CSG).
'''
import os
import sys
import math
import torch as tc
import torch.nn.functional as F
import torch.nn as nn
sys.path.append('..')
import distr as ds
from . import xdistr as xds
from sklearn.metrics.pairwise import cosine_similarity

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

class SemVar:
    def __init__(self, shape_c, shape_s, shape_x, dim_y,
                mean_x1cs, std_x1cs, mean_e1s, std_e1s, mean_y1c, std_y1c,
                tmean_s1xe = None, tstd_s1xe = None, tmean_c1xt = None, tstd_c1xt = None,
                mean_c = 0., std_c = 1., mean_s = 0., std_s = 1.,
                device = None):
        if device is not None: ds.Distr.default_device = device
        self._parameter_dict = {}
        self.shape_c, self.shape_s, self.shape_x, self.shape_y = shape_c, shape_s, shape_x, (dim_y,)
        self.std_c = std_c

        # P(x|c,s) and P(e|s)
        self.p_x1cs = ds.Normal('x', mean=mean_x1cs, std=std_x1cs, shape=shape_x)
        self.p_e1s = ds.Normal('e', mean=mean_e1s, std=std_e1s, shape=shape_x)

        # P(y|c) : causal mechanism
        self.p_y1c = ds.Normal('y', mean=mean_y1c, std=std_y1c, shape=self.shape_y)

        # p(c,s|t) = p(c|t)p(s|t)
        self.p_c1t = ds.Normal('c', mean=mean_c, std=std_c, shape=shape_c)
        self.p_s = ds.Normal('s', mean=mean_s, std=std_s, shape=shape_s)
        self.p_cs = self.p_c1t * self.p_s

        # q(c,s|x,t,e) = q(s|x,e)q(c|x,t)
        self.qt_s1x = ds.Normal('s', mean=tmean_s1xe, std=tstd_s1xe, shape=shape_s)
        self.qt_c1x = ds.Normal('c', mean=tmean_c1xt, std=tstd_c1xt, shape=shape_c)
        self.qt_cs1x = self.qt_s1x * self.qt_c1x

        self.mse_loss = nn.MSELoss(reduction='none').to(device)

    def get_lossfn(self, n_mc_q: int=0, reduction: str="mean", mode: str="defl", recon: bool=False):
        if reduction == "mean": reducefn = tc.mean
        elif reduction == "sum": reducefn = tc.sum
        elif reduction is None or reduction == "none": reducefn = lambda x: x
        else: raise ValueError(f"unknown `reduction` '{reduction}'")

        if mode == 'ccl':
            def lossfn(x, t, e, y) -> tc.Tensor:
                return xds.ccl_elbo(qt_z1x=self.qt_cs1x, p_y1c=self.p_y1c, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_z=self.p_cs, p_c1t=self.p_c1t, p_s=self.p_s,
                                    q_c1x=self.qt_c1x, q_s1x=self.qt_s1x,
                                    obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q,
                                    recon=recon, mse_loss=self.mse_loss)
        elif mode == 'adapt':
            def lossfn(x, t, e, y) -> tc.Tensor:
                return xds.ccl_adapt(qt_z1x=self.qt_cs1x, q_c1x=self.qt_c1x, q_s1x=self.qt_s1x, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_c1t=self.p_c1t, p_s=self.p_s,
                                    obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q,
                                    recon=recon, mse_loss=self.mse_loss)
        return lossfn
    
    def inference(self, x, t, e, shape_mc:tc.Size=tc.Size()):
        obs_xte = ds.edic({'x': x, 't': t, 'e': e})
        # c, s, x, y, e
        return self.qt_cs1x.draw(shape_mc=shape_mc, conds=obs_xte), self.p_x1cs.draw(shape_mc=shape_mc, conds=obs_xte), \
            self.p_y1c.draw(shape_mc=shape_mc, conds=obs_xte), self.p_e1s.draw(shape_mc=shape_mc, conds=obs_xte)
    
    def reg_mmd_cs(self, gamma=1e-1):
        def mmdloss(x, t, e, y):
            cs_samples = self.qt_cs1x.draw((1,), {'x':x, 't':t, 'e':e, 'y':y}, repar=True)
            c_samples = cs_samples['c'].squeeze(0)
            s_samples = cs_samples['s'].squeeze(0)
            return self.mmd_compute(c_samples, s_samples, gamma=gamma)
        return mmdloss
    
    def reg_cossim_cs(self):
        def cossimloss(x, t, e, y):
            cs_samples = self.qt_cs1x.draw((1,), {'x':x, 't':t, 'e':e, 'y':y}, repar=True)
            c_samples = cs_samples['c'].squeeze(0)
            s_samples = cs_samples['s'].squeeze(0)
            return self.compute_cossim(c_samples, s_samples)
        return cossimloss
    
    def compute_cossim(self, X, Y):
        n_samples = X.shape[0]
        X_norm = F.normalize(X, p=2, dim=1)
        Y_norm = F.normalize(Y, p=2, dim=1)
        cos_sim = tc.mm(X_norm, Y_norm.t())
        return tc.sum(cos_sim * tc.eye(n_samples, device=X.device)) / n_samples

    def mmd_compute(self, x, y, kernel_type='gaussian', gamma=1e-1):
        if type(x) != tc.Tensor: x = tc.from_numpy(x)
        if type(y) != tc.Tensor: y = tc.from_numpy(y)

        if kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x, gamma).mean()
            Kyy = self.gaussian_kernel(y, y, gamma).mean()
            Kxy = self.gaussian_kernel(x, y, gamma).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = tc.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma):
        D = self.my_cdist(x, y)
        K = tc.zeros_like(D)

        K.add_(tc.exp(D.mul(-gamma)))
        return K

#!/usr/bin/env python3.6
'''Supervised VAE (no s-v split, the CSGz ablation baseline)
'''
import sys
import math
import torch as tc
sys.path.append('..')
import distr as ds
from . import xdistr as xds

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

class SupVAE:
    def __init__(self, shape_z, shape_x, dim_y,
            mean_x1z, std_x1z, mean_y1z, std_y1z,
            tmean_z1x = None, tstd_z1x = None,
            mean_z = 0., std_z = 1., device = None):
        if device is not None: ds.Distr.default_device = device
        self.shape_x, self.dim_y, self.shape_z = shape_x, dim_y, shape_z

        # Dec: p(x|z) and p(y|z)
        self.p_x1z = ds.Normal('x', mean=mean_x1z, std=std_x1z, shape=shape_x)
        self.p_y1z = ds.Normal('y', mean=mean_y1z, std=std_y1z, shape=shape_x)

        # Prior: p(z)
        self.p_z = ds.Normal('z', mean=mean_z, std=std_z, shape=shape_z)

        # Enc: q(z|x)
        self.qt_z1x = ds.Normal('z', mean=tmean_z1x, std=tstd_z1x, shape=shape_z)

    def get_lossfn(self, n_mc_q: int=0, reduction: str="mean", mode: str="defl", weight_da: float=None, wlogpi: float=None):
        if reduction == "mean": reducefn = tc.mean
        elif reduction == "sum": reducefn = tc.sum
        elif reduction is None or reduction == "none": reducefn = lambda x: x
        else: raise ValueError(f"unknown `reduction` '{reduction}'")

        if self.qt_z1x is not None:
            def lossfn_src(x: tc.Tensor, t: tc.Tensor, e: tc.Tensor, y: tc.Tensor) -> tc.Tensor:
                return xds.elbo_z2xy(self.p_x1z, self.p_z, self.p_y1z, self.qt_z1x, {'x':x, 't':t, 'y':y}, n_mc_q, wlogpi)
       
        return lossfn_src

    def generate(self, shape_mc: tc.Size=tc.Size(), mode: str="src") -> tuple:
        if mode == "src": smp_s = self.p_s.draw(shape_mc)
        elif mode == "tgt": smp_s = self.pt_s.draw(shape_mc)
        else: raise ValueError(f"unknown 'mode' '{mode}'")
        return self.p_x1s.mode(smp_s, False)['x'], self.p_y1s.mode(smp_s, False)['y']


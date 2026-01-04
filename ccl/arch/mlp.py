#!/usr/bin/env python3.6
'''Multi-Layer Perceptron Architecture.

For causal discriminative model and the corresponding generative model.
'''
import sys, os
import json
import torch as tc
import torch.nn.functional as F
import torch.nn as nn
from numbers import Number
sys.path.append('..')
from distr import tensorify, is_same_tensor, wrap4_multi_batchdims

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def init_linear(nnseq, wmean, wstd, bval):
    for mod in nnseq:
        if type(mod) is nn.Linear:
            mod.weight.data.normal_(wmean, wstd)
            mod.bias.data.fill_(bval)

def mlp_constructor(dims, actv = "Sigmoid", lastactv = True): # `Sequential()`, or `Sequential(*[])`, is the identity map for any shape!
    if type(actv) is str: actv = getattr(nn, actv)
    if len(dims) <= 1: return nn.Sequential()
    else: return nn.Sequential(*(
        sum([[nn.Linear(dims[i], dims[i+1]), actv()] for i in range(len(dims)-2)], []) + \
        [nn.Linear(dims[-2], dims[-1])] + ([actv()] if lastactv else [])
    ))

class MLPBase(nn.Module):
    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()
    def load_or_save(self, filename):
        dirname = "init_models_mlp/"
        os.makedirs(dirname, exist_ok=True)
        path = dirname + filename
        if os.path.exists(path): self.load(path)
        else: self.save(path)

class MLP(MLPBase):
    def __init__(self, dims, actv = "Sigmoid"):
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLP, self).__init__()
        self.f_x2y = mlp_constructor(dims, actv, lastactv = False)
    def forward(self, x): return self.f_x2y(x).squeeze(-1)

class MLPcsy1x(MLPBase):
    def __init__(self, dim_x, dims_postx2pres, dim_s, dim_paras, dims_posts2c, dims_postc2prey, dim_y, dim_t, actv = "Sigmoid",
            std_s1xte_val: float=-1., std_c1sxt_val: float=-1., after_actv: bool=True, ind_cs: bool=False,
            dims_vars2prev_xt=None, dims_vars2prev_xe=None, dims_prevs2c=None, dims_prevs2s=None, dims_t2c=None): # if <= 0, then learn the std.
        """
           x, e == g_s()==> prevs --k()->   s   -\
                                                 |=w()=> c ==> y
           x, t == g_c()==> prevc --h()-> parav -/

        """
        super(MLPcsy1x, self).__init__()
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_s, self.dim_y = dim_x, dim_s, dim_y
        self.dim_t, self.dims_t2c = dim_t, dims_t2c
        dim_pres, dim_c = dims_postx2pres[-1], dims_posts2c[-1]
        self.dim_pres, self.dim_c = dim_pres, dim_c
        self.shape_x, self.shape_s, self.shape_c = (dim_x,), (dim_s,), (dim_c,)
        self.shape_t, self.shape_e = (dim_t,), (dim_x,)
        self.dims_postx2pres, self.dim_paras, self.dims_posts2c, self.dims_postc2prey, self.actv \
                = dims_postx2pres, dim_paras, dims_posts2c, dims_postc2prey, actv
        
        self.ind_cs = ind_cs

        # g(x,t,e)
        self.f_xte2prev = mlp_constructor([dim_x*3] + dims_postx2pres, actv)
        self.f_xe2prev = mlp_constructor([dim_x*2] + dims_postx2pres, actv)
        self.f_xt2prev = mlp_constructor([dim_x+dim_t] + dims_postx2pres, actv)
        
        # k(prev)
        self.f_prev2s = nn.Linear(dim_pres, dim_s)
        # h(prev)
        self.f_prev2parav = nn.Linear(dim_pres, dim_paras)
        # w(s, parav)
        if not ind_cs:
            self.f_sparas2c = mlp_constructor([dim_s + dim_paras] + dims_posts2c, actv, lastactv = False)
        else:
            self.f_paras2c = mlp_constructor([dim_paras] + dims_posts2c, actv, lastactv = False)

        # q(y|c)
        self.f_c2y = mlp_constructor([dim_c] + dims_postc2prey + [dim_y], actv, lastactv = False)
        
        self.std_s1xte_val = std_s1xte_val
        self.std_s1xe_val = std_s1xte_val

        self.std_c1sxt_val = std_c1sxt_val
        self.std_c1xt_val = std_c1sxt_val
        self.learn_std_s1xte = std_s1xte_val <= 0 if type(std_s1xte_val) is float else (std_s1xte_val <= 0).any()
        self.learn_std_c1sxt = std_c1sxt_val <= 0 if type(std_c1sxt_val) is float else (std_c1sxt_val <= 0).any()

        self._prev_cache_s = None
        self._prev_cache_c = None
        self._parav_cache = None
        self._x_cache_prev_s = None
        self._x_cache_prev_c = None
        self._t_cache_prev_s = None
        self._t_cache_prev_c = None
        self._e_cache_prev_s = None
        self._e_cache_prev_c = None
        self._s_cache = None
        self._x_cache_s = None
        self._t_cache_s = None
        self._e_cache_s = None
        self._t_cache_c = None
        self._x_cache_parav = None

        ## std models
        if self.learn_std_s1xte:
            self.nn_std_s = nn.Sequential(
                    mlp_constructor(
                        [dim_pres, dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = self.nn_std_s

        if self.learn_std_c1sxt:
            if not ind_cs:
                _input_dim = dim_s + dim_paras
            else:
                _input_dim = dim_paras
                
            self.nn_std_c = nn.Sequential(
                    nn.BatchNorm1d(_input_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp_constructor(
                        [_input_dim] + dims_posts2c,
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_c, 0., 1e-2, 0.)
            self.f_std_c = wrap4_multi_batchdims(self.nn_std_c, ndim_vars=1)
        
        self.nn_std_y1c = nn.Sequential(
                                mlp_constructor(
                                [dim_c] + dims_postc2prey + [dim_y],
                                nn.ReLU, lastactv = False),
                                nn.Softplus())
        init_linear(self.nn_std_y1c, 0., 1e-2, 0.)
        self.f_std_y1c = self.nn_std_y1c

    def _get_prevs(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(t, self._t_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._t_cache_prev_s = t
            self._e_cache_prev_s = e
            _input = tc.cat([x, t, e], dim=-1)
            self._prev_cache_s = self.f_xte2prev(_input) # g(x,t,e)
        return self._prev_cache_s
    
    def _get_prevs_ind(self, x,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._e_cache_prev_s = e
            _input = tc.cat([x, e], dim=-1)
            self._prev_cache_s = self.f_xe2prev(_input) # g(x,t,e)
        return self._prev_cache_s
    
    def _get_prevc(self, x,t):
        if (not is_same_tensor(x, self._x_cache_prev_c))\
        and (not is_same_tensor(t, self._t_cache_prev_c)):
            self._x_cache_prev_c = x
            self._t_cache_prev_c = t
            _input = tc.cat([x, t], dim=-1)
            self._prev_cache_c = self.f_xt2prev(_input) # g(x,t,e)
        return self._prev_cache_c
    
    def s1xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(t, self._t_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._t_cache_s = t
            self._e_cache_s = e

            self._s_cache = self.f_prev2s(self._get_prevs(x,t,e))
        return self._s_cache

    def std_s1xte(self, x,t,e):
        if self.learn_std_s1xte:
            return self.f_std_s(self._get_prevs(x,t,e))
        else:
            return tensorify(x.device, self.std_s1xte_val)[0].expand(x.shape[:-1]+(self.dim_s,))

    def s1xe(self, x,e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._e_cache_s = e
            self._s_cache = self.f_prev2s(self._get_prevs_ind(x,e))
        return self._s_cache
    
    def std_s1xe(self, x,e):
        if self.learn_std_s1xte:
            return self.f_std_s(self._get_prevs_ind(x,e))
        else:
            return tensorify(x.device, self.std_s1xte_val)[0].expand(x.shape[:-1]+(self.dim_s,))

    def _get_parav(self, x,t):
        # h(prev)
        if (not is_same_tensor(x, self._x_cache_parav))\
        and (not is_same_tensor(t, self._t_cache_c)):
            self._x_cache_parav = x
            self._t_cache_c = t
            self._parav_cache = self.f_prev2parav(self._get_prevc(x,t))
        return self._parav_cache

    def c1sxt(self, s, x, t): # q(c|s,x,t)
        parav = self._get_parav(x,t) # parav | g(x,t,e)
        # w(s, parav)
        return self.f_sparas2c(tc.cat([s, parav], dim=-1))

    def std_c1sxt(self, s, x, t):
        if self.learn_std_c1sxt:
            parav = self._get_parav(x,t)
            return self.f_std_c(tc.cat([s, parav], dim=-1))
        else:
            return tensorify(x.device, self.std_c1sxt_val)[0].expand(x.shape[:-1]+t.shape[:-1]+(self.dim_c,))

    def c1xt(self, x,t):
        '''
        q(c|x,t,e) = q(c|s,x,t,e)q(s|x,t,e)
        '''
        parav = self._get_parav(x,t) # parav | g(x,t,e)
        return self.f_paras2c(parav)
    
    def c1x(self, x,t,e):
        '''
        q(c|x,t,e) = q(c|s,x,t,e)q(s|x,t,e)
        '''
        # if self.ind_cs:
        return self.c1xt(x,t)
        # else:
        #     return self.c1sxt(self.s1xte(x,t,e), x,t)

    def std_c1xt(self, x,t):
        if self.learn_std_c1sxt:
            parav = self._get_parav(x,t)
            return self.f_std_c(parav)
        else:
            return tensorify(x.device, self.std_c1xt_val)[0].expand(x.shape[:-1]+t.shape[:-1]+(self.dim_c,))
    
    def y1c(self, c):
        '''
        q(y|c)
        '''
        return self.f_c2y(c)
    
    def std_y1c(self, c):
        return self.f_std_y1c(c)
    
    def ys1x(self, x):
        c = self.c1x(x) # q(c|x,t,e)
        return self.y1c(c), c

    def forward(self, x,t,e):
        '''
        q(y|c) = q(y|c)q(c,s|x,t,e)
        '''
        return self.y1c(self.c1x(x,t,e))

class MLPcsy1xte(MLPBase):
    def __init__(self, dim_x, dim_t, dim_e, dim_y, dim_c, dim_s, actv = "Sigmoid",
            std_s_val: float=-1., std_c_val: float=-1., after_actv: bool=True, ind_cs: bool=False, use_fc=False,
            dims_vars2prev_xt=None, dims_vars2prev_xe=None, dims_c2y=None, dims_prevs2c=None, dims_prevs2s=None, dims_t2c=None,
            dims_precs2x=None, dims_pres2e=None, dim_x2c=None):
        super(MLPcsy1xte, self).__init__()
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_t, self.dim_e, self.dim_y, self.dim_c, self.dim_s = dim_x, dim_t, dim_e, dim_y, dim_c, dim_s
        self.dim_c, self.dim_s = dim_c, dim_s

        self.dim_pres4c = dims_vars2prev_xt[-1]
        self.dim_pres4s = dims_vars2prev_xe[-1]

        self.shape_x, self.shape_s, self.shape_c = (dim_x,), (dim_s,), (dim_c,)
        self.shape_t, self.shape_e = (dim_t,), (dim_e,)
        self.actv = actv
        
        self.ind_cs = ind_cs
        self.dims_t2c = dims_t2c
        self.dims_precs2x = dims_precs2x
        self.dims_pres2e = dims_pres2e

        # func for xt to h1 and xe to h2
        self.f_xt2prev = mlp_constructor([dim_x+dim_t] + dims_vars2prev_xt, actv, lastactv = False)
        self.f_xe2prev = mlp_constructor([dim_x+dim_e] + dims_vars2prev_xe, actv, lastactv = False)
        
        # Mean of inferece models
        # func for h1 to c and h2 to s
        self.f_prev2c = mlp_constructor([self.dim_pres4c] + dims_prevs2c + [dim_c], actv, lastactv=False)
        self.f_prevs2s = mlp_constructor([self.dim_pres4s] + dims_prevs2s + [dim_s], actv, lastactv=False)

        # f_x2c
        self.f_x2c = mlp_constructor([self.dim_x] + dim_x2c + [dim_c], actv, lastactv=False)
  
        # p(y|c)
        self.f_c2y = mlp_constructor([dim_c] + dims_c2y + [dim_y], actv, lastactv = False)

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = nn.Sequential(
                nn.Linear(dim_y, 512),
                nn.ReLU(),
                nn.Linear(512, 1))
        
        # Std. of inference models
        self.std_c_val = std_c_val
        self.std_c1xt_val = std_c_val
        self.std_s_val = std_s_val
        self.std_s1xe_val = std_s_val

        self.learn_std_s = std_s_val <= 0 if type(std_s_val) is float else (std_s_val <= 0).any()
        self.learn_std_c = std_c_val <= 0 if type(std_c_val) is float else (std_c_val <= 0).any()

        ## std models
        if self.learn_std_c:
            self.f_std_c = nn.Sequential(
                    mlp_constructor(
                        [self.dim_pres4c] + dims_prevs2c + [dim_c],
                        actv, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.f_std_c, 0., 1e-2, 0.)

            # f_std_c1x
            self.f_std_c1x = nn.Sequential(
                    mlp_constructor(
                        [self.dim_x] + dim_x2c + [dim_c],
                        actv, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.f_std_c1x, 0., 1e-2, 0.)

        if self.learn_std_s:
            self.f_std_s = nn.Sequential(
                    mlp_constructor(
                        [self.dim_pres4s] + dims_prevs2s + [dim_s],
                        actv, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.f_std_s, 0., 1e-2, 0.)
        
        self.f_std_y1c = nn.Sequential(
                    mlp_constructor(
                        [dim_c] + dims_c2y + [dim_y],
                        actv, lastactv = False),
                    nn.Softplus()
                )
        
        init_linear(self.f_std_y1c, 0., 1e-2, 0.)

        self._x_cache_prev_c = None
        self._t_cache_prev_c = None
        self._prev_cache_xt = None
        self._x_cache_prev_s = None
        self._e_cache_prev_s = None
        self._parav_cache_xe = None
        self._x_cache_c = None
        self._t_cache_c = None
        self._c_cache = None
        self._x_cache_s = None
        self._e_cache_s = None
        self._s_cache = None
    
    def _get_prevs_xt(self, x,t):
        if (not is_same_tensor(x, self._x_cache_prev_c))\
        and (not is_same_tensor(t, self._t_cache_prev_c)):
            self._x_cache_prev_c = x
            self._t_cache_prev_c = t
            _input = tc.cat([x, t], dim=-1)
            self._prev_cache_xt = self.f_xt2prev(_input)
        return self._prev_cache_xt
    
    def _get_prevs_xe(self, x,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._e_cache_prev_s = e
            self._parav_cache_xe = self.f_xe2prev(tc.cat([x, e], dim=-1))
        return self._parav_cache_xe

    def s1xe(self, x, e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._e_cache_s = e
            self._s_cache = self.f_prevs2s(self._get_prevs_xe(x,e))
        return self._s_cache
    
    def std_s1xe(self, x, e):
        if self.learn_std_s:
            return self.f_std_s(self._get_prevs_xe(x,e))

    def c1xt(self, x,t):
        if (not is_same_tensor(x, self._x_cache_c))\
        and (not is_same_tensor(t, self._t_cache_c)):
            self._x_cache_c = x
            self._t_cache_c = t
            self._c_cache = self.f_prev2c(self._get_prevs_xt(x,t))
        return self._c_cache
    
    def std_c1xt(self, x,t):
        if self.learn_std_c:
            return self.f_std_c(self._get_prevs_xt(x,t))
        
    def c1x(self, x):
        return self.f_x2c(x)
    
    def std_c1x(self, x):
        if self.learn_std_c:
            return self.f_std_c1x(x)
    
    def y1c(self, c):
        return self.f_c2y(c)
    
    def std_y1c(self, c):
        return self.f_std_y1c(c)

    def forward(self, x,t,e):
        if self.use_fc: return self.fc(self.y1c(self.c1xt(x,t)))
        else: return self.y1c(self.c1xt(x,t))

class MLPx1cs(MLPBase):
    def __init__(self, dim_c = None, dim_s = None, dim_x = None,
                 dim_t = None, dim_e = None, actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if dim_t is None: dim_t = discr.dim_t
        if dim_e is None: dim_e = discr.dim_e
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLPx1cs, self).__init__()
        self.dim_c, self.dim_s, self.dim_x, self.dim_t, self.dim_e = dim_c, dim_s, dim_x, dim_t, dim_e
        
        dims_precs2x = discr.dims_precs2x
        dims_pres2e = discr.dims_pres2e

        # X | C, S
        self.f_vparas2x = mlp_constructor([dim_s + dim_c] + dims_precs2x + [dim_x], actv, lastactv = False)

        # Env | S
        self.f_vparas2e = mlp_constructor([dim_s] + dims_pres2e + [dim_e], actv, lastactv = False)
 
    def x1cs(self, c, s): return self.f_vparas2x(tc.cat([c, s], dim=-1))
    def e1s(self, s): return self.f_vparas2e(s)
    def forward(self, c, s): return self.x1cs(c, s), self.e1s(s)

# VAE Discriminator
class MLPz1x(MLPBase):
    def __init__(self, dim_z = None, std_z1x_val = -1., dims_z2y = None, dims_x2z = None, dim_x = None, dim_y = None, dim_t = None,
                 dims_t2c = None, after_actv=None, actv = None):
        if actv is None: actv = "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLPz1x, self).__init__()
        self.dim_z, self.dim_x, self.dim_t, self.dims_x2z, self.actv = dim_z, dim_x, dim_t, dims_x2z, actv
        self.shape_z = (dim_z,)
        self.dims_t2c = dims_t2c
        self.f_x2z = mlp_constructor([dim_x] + dims_x2z + [dim_z], actv, lastactv=False)
        self.f_z2y = mlp_constructor([dim_z] + dims_z2y + [dim_y], actv, lastactv = False)
        self.learn_std_z = std_z1x_val <= 0 if type(std_z1x_val) is float else (std_z1x_val <= 0).any()

        if self.learn_std_z:
            self.nn_std_z = nn.Sequential(
                    mlp_constructor(
                        [dim_x] + dims_x2z + [dim_z],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_z, 0., 1e-2, 0.)
            self.f_std_z = self.nn_std_z

        self.nn_std_y1z = nn.Sequential(
                                mlp_constructor(
                                [dim_z] + dims_z2y + [dim_y],
                                nn.ReLU, lastactv = False),
                                nn.Softplus())
        init_linear(self.nn_std_y1z, 0., 1e-2, 0.)
        self.f_std_y1z = self.nn_std_y1z

    def z1x(self, x): return self.f_x2z(x)
    def std_z1x(self, x): return self.f_std_z(x)
    def y1z(self, z): return self.f_z2y(z)
    def std_y1z(self, z): return self.f_std_y1z(z)
    def forward(self, x): return self.y1z(self.z1x(x))

# VAE Generator
class MLPx1z(MLPBase):
    def __init__(self, dim_z = None, dims_x2z = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_z is None: dim_z = discr.dim_z
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_x2z is None:
            dims_x2z = discr.dims_x2z[::-1]
        super(MLPx1z, self).__init__()
        self.dim_z, self.dim_x, self.dims_x2z, self.actv = dim_z, dim_x, dims_x2z, actv
        self.f_z2x = mlp_constructor([dim_z] + dims_x2z + [dim_x], actv, lastactv=False)

    def x1z(self, z): return self.f_z2x(z)
    def forward(self, z): return self.x1z(z)

class MLPv1s(MLPBase):
    def __init__(self, dim_c = None, dims_pres2postv = None, dim_s = None,
            actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_s is None: dim_s = discr.dim_s
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postv is None: dims_pres2postv = discr.dims_posts2c[::-1][1:]
        super(MLPv1s, self).__init__()
        self.dim_c, self.dim_s, self.dims_pres2postv, self.actv = dim_c, dim_s, dims_pres2postv, actv
        self.f_s2v = mlp_constructor([dim_c] + dims_pres2postv + [dim_s], actv)

    def v1s(self, s): return self.f_s2v(s)
    def forward(self, s): return self.v1s(s)
    
class MLPc1t(MLPBase):
    def __init__(self, dim_c = None,
                 dim_t = None, actv = None, *, discr = None):
        if dim_c is None:
            try: dim_c = discr.dim_c
            except: dim_c = discr.dim_z
        if dim_t is None: dim_t = discr.dim_t
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        dims_t2c = discr.dims_t2c
        super(MLPc1t, self).__init__()

        # Replace lambda functions with proper nn.Module activations
        if actv == 'lrelu':
            self.act_f = nn.LeakyReLU(negative_slope=0.1)
        elif actv == 'xtanh':
            self.act_f = XTanh(alpha=0.1)
        elif actv == 'sigmoid':
            self.act_f = nn.Sigmoid()
        elif actv == 'none':
            self.act_f = nn.Identity()
        
        dims = [dim_t] + dims_t2c + [dim_c]
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.act_f)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        # layers.append(nn.Softplus())
        
        self.f_std_c1t = nn.Sequential(*layers)
        init_linear(self.f_std_c1t, 0., 1e-2, 0.)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def std_c1t(self, t):
        return self.f_std_c1t(t).exp()
    
    def forward(self, t):
        return self.std_c1t(t)
    
class XTanh(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return tc.tanh(x) + self.alpha * x

class MLPs1ct(MLPBase):
    def __init__(self, dim_s = None, dims_postx2pres = None,
                 dim_t = None, actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_t is None: dim_t = discr.dim_t
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if dims_postx2pres is None: dims_postx2pres = discr.dims_posts2c[::-1][1:] + [discr.dim_paras, discr.dim_paras]
        super(MLPs1ct, self).__init__()
        self.dim_s, self.dim_t = dim_s, dim_t
        self.dims_postx2pres, self.actv = dims_postx2pres, actv
        
        slope = 0.1
        self.input_dim = dim_s + dim_t
        self.output_dim = dim_s
        self.n_layers = len(self.dims_postx2pres)
        
        if isinstance(dims_postx2pres, Number):
            self.hidden_dim = [dims_postx2pres] * (self.n_layers - 1)
        elif isinstance(dims_postx2pres, list):
            self.hidden_dim = dims_postx2pres
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(dims_postx2pres))

        if isinstance(actv, str):
            self.activation = [actv] * (self.n_layers - 1)
        elif isinstance(actv, list):
            self.hidden_dim = actv
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(actv))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def std_s1ct(self, c, t):
        h = tc.cat([c, t], dim=-1)
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h.exp()
    
    def forward(self, c, t):
        return self.std_s1ct(c, t)
        
def create_discr_from_json(stru_name: str, dim_x: int, dim_y: int, dim_t: int, actv: str=None,
        std_s1xte_val: float=-1., std_c1sxt_val: float=-1., after_actv: bool=True, ind_cs: bool=False, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPcsy1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPcsy1x(dim_x=dim_x, dim_y=dim_y, dim_t=dim_t, std_s1xte_val=std_s1xte_val, std_c1sxt_val=std_c1sxt_val,
            after_actv=after_actv, ind_cs=ind_cs, **stru)

def create_ccl_discr_from_json(stru_name: str, dim_x: int,dim_t: int, dim_e: int, dim_y: int, actv: str=None,
        std_s_val: float=-1., std_c_val: float=-1., y_dtype='emb', after_actv: bool=True, ind_cs: bool=False, use_fc: bool=False, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPcsy1xte'][y_dtype][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPcsy1xte(dim_x=dim_x, dim_t=dim_t, dim_e=dim_e, dim_y=dim_y, std_s_val=std_s_val, std_c_val=std_c_val,
                     after_actv=after_actv, ind_cs=ind_cs, use_fc=use_fc, **stru)

def create_vae_discr_from_json(stru_name: str, dim_x: int, dim_y: int, dim_t: int, actv: str=None,
                               std_z1x_val: float=-1., after_actv: bool=True, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPz1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPz1x(dim_x=dim_x, dim_y=dim_y, dim_t=dim_t, std_z1x_val=std_z1x_val,
                  after_actv=after_actv, **stru)

def create_gen_from_json(model_type: str="MLPx1cs", discr: MLPcsy1x=None, stru_name: str=None, dim_x: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_x=dim_x, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_x=dim_x, discr=discr, **stru)
    
def create_prior_from_json(model_type: str="MLPc1t", discr: MLPcsy1x=None, stru_name: str=None, dim_c: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_c=dim_c, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        return eval(model_type)(dim_c=dim_c, discr=discr, actv=actv, **stru)

def create_s_prior_from_json(model_type: str="MLPs1ct", discr: MLPcsy1x=None, stru_name: str=None, dim_s: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_s=dim_s, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_s=dim_s, discr=discr, **stru)

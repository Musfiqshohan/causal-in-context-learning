import os
import sys
import copy
import argparse
import tqdm
import tempfile
import pandas as pd
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import types
from dataclasses import dataclass
from typing import Optional
from types import SimpleNamespace
import distr as ds

from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data import Subset

from distr import edic
from arch import mlp
from methods import SemVar, Adaptor

from sklearn.model_selection import train_test_split

from copy import deepcopy
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccl.utils.utils import unique_filename, boolstr, print_infrstru_info, EarlyStopping, compute_cossim, EvalRetrival, print_results, set_seed_all, RetrievalEval
from ccl.utils.utils_data import HighDimSCMRealWorldDataset, load_dataset

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

import wandb

RESULTS_DIC = {}

@dataclass
class BatchData:
    X: tc.Tensor
    Y: tc.Tensor
    T: tc.Tensor
    E: tc.Tensor
    label_T: tc.Tensor
    label_E: tc.Tensor
    index: tc.Tensor
    Y_num: Optional[tc.Tensor] = None
    label_subT: Optional[tc.Tensor] = None

def custom_collate_fn(batch):
    # Convert list of objects (assumed to be from Dataset) to BatchData
    if isinstance(batch[0], types.SimpleNamespace):
        return BatchData(
            X=tc.stack([item.X.clone().detach() for item in batch]),
            Y=tc.stack([item.Y.clone().detach() for item in batch]),
            T=tc.stack([item.T.clone().detach() for item in batch]),
            E=tc.stack([item.E.clone().detach() for item in batch]),
            label_T=tc.tensor([item.label_T.item() for item in batch]),  # .item() 사용하여 Python 숫자로 변환
            label_E=tc.tensor([item.label_E.item() for item in batch]),
            index=tc.tensor([item.index.item() for item in batch]),
            Y_num=tc.stack([item.Y_num.clone().detach() for item in batch]) if batch[0].Y_num is not None else None,
            label_subT=tc.stack([item.label_subT.clone().detach() for item in batch]) if batch[0].label_subT is not None else None
        )
    return batch

class ParamGroupsCollector:
    def __init__(self):
        self.param_groups = []

    def reset(self):
        self.param_groups = []

    def collect_params(self, model, lr, weight_decay=None):
        if hasattr(model, 'parameter_groups'):
            groups_inc = list(model.parameter_groups())
            for grp in groups_inc:
                if 'lr_ratio' in grp:
                    grp['lr'] = lr * grp['lr_ratio']
                elif 'lr' not in grp:  # Do not overwrite existing lr assignments
                    grp['lr'] = lr
                if weight_decay is not None:
                    grp['weight_decay'] = weight_decay
                elif 'weight_decay' not in grp:  # Default weight_decay if not set
                    grp['weight_decay'] = 0.0
            self.param_groups += groups_inc
        else:
            self.param_groups += [
                {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay or 0.0}
            ]

class ShrinkRatio:
    def __init__(self, w_iter, decay_rate):
        self.w_iter = w_iter
        self.decay_rate = decay_rate

    def __call__(self, n_iter):
        return (1 + self.w_iter * n_iter) ** (-self.decay_rate)

class ResultsContainer:
    def __init__(self, len_ts, frame, ag, is_binary, device, ckpt = None):
        for k,v in locals().items():
            if k not in {"self", "ckpt"}: setattr(self, k, v)
        self.dc = dict( epochs = [], losses = [],
                accs_tr = [], llhs_tr = [], accs_val = [], llhs_val = [] )
        if len_ts:
            ls_empty = [[] for _ in range(len_ts)]
            self.dc.update( ls_accs_ts = ls_empty, ls_llhs_ts = deepcopy(ls_empty) )
        else:
            self.dc.update( accs_ts = [], llhs_ts = [] )

    def update(self, *, epk, loss):
        self.dc['epochs'].append(epk)
        self.dc['losses'].append(loss)

def namespace_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].__dict__.keys():
        batch_dict[key] = [getattr(sample, key) for sample in batch]
    for key in batch_dict:
        if isinstance(batch_dict[key][0], tc.Tensor):
            batch_dict[key] = tc.stack(batch_dict[key])
    return SimpleNamespace(**batch_dict)

# Init models
def auto_load(dc_vars, names, ckpt):
    if ckpt:
        if type(names) is str: names = [names]
        for name in names:
            model = dc_vars[name]
            model.load_state_dict(ckpt[name+'_state_dict'])
            if hasattr(model, 'eval'): model.eval()

def get_frame(discr, gen, dc_vars, device = None, discr_src = None):
    if type(dc_vars) is not edic:
        dc_vars = edic(dc_vars)

    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_c = discr.shape_c if hasattr(discr, "shape_c") else (dc_vars['dim_c'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)
    
    if dc_vars['only_x4c']:
        tmean_c=discr.c1x
        qstd_c = discr.std_c1x
    else:
        tmean_c=discr.c1xt
        qstd_c = discr.std_c1xt

    tmean_s=discr.s1xe
    qstd_s = discr.std_s1xe

    mode = dc_vars['mode']

    if dc_vars['cond_prior']:
        prior_std = mlp.create_prior_from_json("MLPc1t", discr, actv=dc_vars['actv_prior'],jsonfile=dc_vars['mlpstrufile']).to(device)
        std_c = prior_std.std_c1t
        print("Conditional prior std")
        print(prior_std)
    else:
        std_c = dc_vars['sig_c']

    frame = SemVar(shape_c=shape_c, shape_s=shape_s, shape_x=shape_x, dim_y=dc_vars['dim_y'],
                   mean_x1cs=gen.x1cs, std_x1cs=dc_vars['pstd_x'],
                   mean_e1s=gen.e1s, std_e1s=dc_vars['pstd_e'],
                   mean_y1c=discr.y1c, std_y1c=discr.std_y1c,
                   tmean_c1xt=tmean_c, tstd_c1xt=qstd_c,
                   tmean_s1xe=tmean_s, tstd_s1xe=qstd_s, 
                   mean_c=dc_vars['mu_c'], std_c=std_c, mean_s=dc_vars['mu_s'], std_s = dc_vars['sig_s'],
                   device=device
                   )
    return frame

def get_discr(archtype, dc_vars):
    if archtype == "mlp":
        if dc_vars['ind_cs']:
            discr = mlp.create_ccl_discr_from_json(
                    *dc_vars.sublist([
                        'discrstru', 'dim_x', 'dim_t', 'dim_e', 'dim_y', 'actv']),
                    y_dtype=dc_vars['y_dtype'],
                    after_actv=dc_vars['after_actv'],
                    ind_cs=dc_vars['ind_cs'],
                    use_fc=dc_vars['use_fc'],
                    jsonfile=dc_vars['mlpstrufile']
                )
    else: raise ValueError(f"unknown `archtype` '{archtype}'")
    return discr

def get_gen(archtype, dc_vars, discr):
    if archtype == "mlp":
        gen = mlp.create_gen_from_json(
            "MLPx1cs", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
    return gen

def get_models(archtype, dc_vars, ckpt = None, device = None, only_x4c = False):
    if type(dc_vars) is not edic: dc_vars = edic(dc_vars)
    discr = get_discr(archtype, dc_vars)
    if ckpt is not None:
        print("Loading discr from ckpt")
        auto_load(locals(), 'discr', ckpt)
    discr.to(device)

    gen = get_gen(archtype, dc_vars, discr)
    if ckpt is not None:
        print("Loading Gen from ckpt")
        auto_load(locals(), 'gen', ckpt)
    gen.to(device)

    frame = get_frame(discr, gen, dc_vars, device)

    print("Inference model")
    print(discr)
    print("===")
    print("Gen. model")
    print(gen)

    return discr, gen, frame

def dc_state_dict(dc_vars, *name_list):
    return {name+"_state_dict" : dc_vars[name].state_dict()
            for name in name_list if hasattr(dc_vars[name], 'state_dict')}

def save_ckpt(ag, ckpt, discr, gen, frame, opt, dc_vars, is_toy=False, filename=None):
    if ag.exp_name in ['mgsm', 'toy']: dirname = "ckpt_" + ag.mode + "/" + ag.traindom + "/"
    else: dirname = "ckpt_" + ag.mode + "/" + ag.exp_name + "/"
    os.makedirs(dirname, exist_ok=True)
    i = 0
    testdoms = ag.testdoms
    if ag.exp_name == 'ood_nlp': td = ag.traindom.split('_')[-1]
    else: td = ag.testdoms

    dim_x = dc_vars['dim_x']
    dim_y = dc_vars['dim_y']
    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_c = discr.shape_c if hasattr(discr, "shape_c") else (dc_vars['dim_c'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)

    if filename is None:
        Y_INFO = ag.y_dtype if ag.y_emb_option == 'default' else ag.y_emb_option
        filename = unique_filename(
                dirname + (f"ood_toy" if is_toy else f"{ag.exp_name}_{ag.emb_model}_{Y_INFO}"), ".pt", n_digits = 3, count_start = ag.ckp_id,
            ) if ckpt is None else ckpt['filename']
                
    dc_vars = edic(locals()).sub([
            'dirname', 'filename', 'testdoms',
            'shape_x', 'shape_c', 'shape_s', 'dim_x', 'dim_y']) | ( edic(vars(ag)) - {'testdoms'}
        ) | dc_state_dict(locals(), "discr", "gen", "frame", "opt")

    tc.save(dc_vars, filename)
    return filename
    
def load_ckpt(filename: str, loadmodel: bool=False, device: tc.device=None, archtype: str="mlp", map_location: tc.device=None):
    ckpt = tc.load(filename, map_location, weights_only=False)
    if loadmodel:
        return (ckpt,) + get_models(archtype, ckpt, ckpt, device)
    else: return ckpt

# Built methods
def get_ce_or_bce_loss(discr, y_dtype: int, reduction: str="none", mode='ind', use_fc=False):
    if y_dtype == 'clf_binary': lossobj = tc.nn.BCEWithLogitsLoss(reduction=reduction)
    elif y_dtype == 'clf_multi': lossobj = tc.nn.CrossEntropyLoss(reduction=reduction)
    elif y_dtype == 'regression': lossobj = tc.nn.MSELoss(reduction=reduction)
    elif y_dtype == 'emb': lossobj = tc.nn.MSELoss(reduction=reduction)
    
    if y_dtype == 'regression' or use_fc:
        lossfn = lambda x: tc.sqrt(lossobj(discr(*x[:3]), x[-1]))
    else:
        lossfn = lambda x: lossobj(discr(*x[:-1]), x[-1])
    
    return lossobj, lossfn

def add_ce_loss(lossobj, celossfn, ag, loss_reg_cs=None):
    # shrink_sup = ShrinkRatio(w_iter=ag.wsup_wdatum*ag.n_bat, decay_rate=ag.wsup_expo)

    def lossfn(*x_y_maybext_niter):
        # (xs, ts, envs, ys, y_num_s, phase, epk) or (xs, ts, envs, ys, phase, epk)

        state = x_y_maybext_niter[-2]
        if ag.mode == 'ccl':
            # log_phi_loss, x_recon_loss, e_recon_loss, c_recon_loss, s_recon_loss = lossobj(*x_y_maybext_niter[:4]) # [batch_size]
            x_recon_loss, e_recon_loss, c_recon_loss, s_recon_loss = lossobj(*x_y_maybext_niter[:4]) # [batch_size]

            # if ag.current_epoch % 10 != 0: e_recon_loss.detach()
            # elbo = -1*ag.wlogpi * log_phi_loss + -1*ag.wrecon_x * x_recon_loss + -1*ag.wrecon_e * e_recon_loss + -1*ag.wrecon_c * c_recon_loss
            elbo = -1*ag.wrecon_x * x_recon_loss + -1*ag.wrecon_e * e_recon_loss + -1*ag.wrecon_c * c_recon_loss
            if s_recon_loss is not None: elbo -= ag.wrecon_s * s_recon_loss

            if loss_reg_cs is not None: reg_cs_loss = loss_reg_cs(*x_y_maybext_niter[:4])
            if not ag.elbo_only:
                if len(x_y_maybext_niter) == 6: celoss = celossfn(x_y_maybext_niter[:4]).mean(dim=-1)
                if len(x_y_maybext_niter) == 7: celoss = celossfn(x_y_maybext_niter[:5]).mean(dim=-1)

        if ag.log_loss:
            RESULTS_DIC.update({
                            f"Epoch": ag.current_epoch,
                            f"{state}_Elbo": elbo.mean().item(),
                            # f"{state}_log_phi_loss": -1*log_phi_loss.mean().item(),
                            f"{state}_Loss_x": x_recon_loss.mean().item() if ag.recon else -1*x_recon_loss.mean().item(),
                            f"{state}_Loss_e": e_recon_loss.mean().item() if ag.recon else -1*e_recon_loss.mean().item(),
                            f"{state}_Loss_c": -1*c_recon_loss.mean().item()
                        })
            
            if s_recon_loss is not None:
                RESULTS_DIC[f'{state}_Loss_s'] = -1*s_recon_loss.mean().item()
            if loss_reg_cs is not None:
                RESULTS_DIC[f'{state}_Reg_CS_loss'] = reg_cs_loss.item()
            if not ag.elbo_only:
                RESULTS_DIC[f'{state}_Loss_sup'] = celoss.mean().item()

        if s_recon_loss is None: s_recon_loss = 0

        # raw_elbo = -1 * (log_phi_loss +  x_recon_loss +  e_recon_loss + c_recon_loss)
        raw_elbo = -1 * (x_recon_loss +  e_recon_loss + c_recon_loss + s_recon_loss) + celoss
        if ag.elbo_only:
            if loss_reg_cs is not None:
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()+reg_cs_loss})

                if ag.log_loss: wandb.log(RESULTS_DIC)
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()+reg_cs_loss})

                if ag.log_loss: wandb.log(RESULTS_DIC)
                return ag.wgen * elbo.mean() + ag.wreg_cs * reg_cs_loss
            else:
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()})
                if ag.log_loss: wandb.log(RESULTS_DIC)
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()})
                if ag.log_loss: wandb.log(RESULTS_DIC)
                return ag.wgen * elbo.mean()
        else:
            if loss_reg_cs is not None:
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()-1*reg_cs_loss})
                if ag.log_loss: wandb.log(RESULTS_DIC)
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()-1*reg_cs_loss})
                if ag.log_loss: wandb.log(RESULTS_DIC)
                # -1*ag.wlogpi * log_phi_loss
                elbo += ag.wlogpi * celoss
                elbo /= ag.wgen

                return ag.wgen * elbo.mean() - ag.wreg_cs * reg_cs_loss # + ag.wsup * celoss
            else:
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()})
                if ag.log_loss: wandb.log(RESULTS_DIC)
                RESULTS_DIC.update({f"{state}_total_Loss": raw_elbo.mean()})
                if ag.log_loss: wandb.log(RESULTS_DIC)

                elbo += ag.wlogpi * celoss
                # elbo /= ag.wgen
                # return ag.wgen * elbo.mean() + ag.wsup * celoss
                return elbo.mean()
    return lossfn

def ood_methods(discr, frame, ag):
    celossfn = get_ce_or_bce_loss(discr, ag.y_dtype, 'none', use_fc=ag.use_fc)[1]
    
    lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, 'ccl', recon=ag.recon)

    if ag.wreg_cs > -1.0:
        if ag.reg_cs == "cossim": loss_reg_cs = frame.reg_cossim_cs()
        elif ag.reg_cs == "mmd": loss_reg_cs = frame.reg_mmd_cs()
    else: loss_reg_cs = None

    lossfn = add_ce_loss(lossobj, celossfn, ag, loss_reg_cs)
    return lossfn

def adapt_methods(gen, frame, ag):
    lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, 'adapt', recon=ag.recon)

    if ag.wreg_cs > -1.0:
        if ag.reg_cs == "cossim": loss_reg_cs = frame.reg_cossim_cs()
        elif ag.reg_cs == "mmd": loss_reg_cs = frame.reg_mmd_cs()
    else: loss_reg_cs = None

    def lossfn(*x_y_maybext_niter):
        log_x_recon_loss, log_e_recon_loss, expc_val_c, expc_val_s = lossobj(*x_y_maybext_niter[:4])
        if loss_reg_cs is not None: reg_cs_loss = loss_reg_cs(*x_y_maybext_niter[:4])

        if ag.current_epoch % 5 == 0:
            # print(f" Epk {ag.current_epoch} Log X prob: {log_x_recon_loss.mean().item()}")
            # print(f" Epk {ag.current_epoch} Log E prob: {log_e_recon_loss.mean().item()}")
            # print(f" Epk {ag.current_epoch} C L2 norm: {tc.sum(c_hat**2, dim=1).mean().item()}")
            # print(f" Epk {ag.current_epoch} S L2 norm: {tc.sum(s_hat**2, dim=1).mean().item()}")
            # if loss_reg_cs is not None:
            #     print(f" Epk {ag.current_epoch} {ag.reg_cs} loss: {reg_cs_loss.mean().item()}")

            wandb.log({
                'epoch': ag.current_epoch,
                'log_x_recon_loss': log_x_recon_loss.mean().item(),
                'log_e_recon_loss': log_e_recon_loss.mean().item(),
                'kl_c_loss': expc_val_c.mean().item(),
                'kl_s_loss': expc_val_s.mean().item()
            })

        # if ag.current_epoch > 10: log_e_recon_loss.detach_()
        raw_loss = -1*tc.mean(log_x_recon_loss+log_e_recon_loss+expc_val_c+expc_val_s) # + tc.sum(c_hat**2, dim=1).mean() + tc.sum(s_hat**2, dim=1).mean()
        if ag.wreg_cs == -1: reg_loss = tc.mean(-1*ag.wada_x*log_x_recon_loss + -1*ag.wada_e*log_e_recon_loss + -1*ag.lam_c*expc_val_c + -1*ag.lam_s*expc_val_s)
            # + ag.lam_c * tc.sum(c_hat**2, dim=1).mean() + ag.lam_s * tc.sum(s_hat**2, dim=1).mean()
        else:
            reg_loss = tc.mean(-1*ag.wada_x*log_x_recon_loss + -1*ag.wada_e*log_e_recon_loss + -1*ag.lam_c*expc_val_c + -1*ag.lam_s*expc_val_s) - ag.wreg_cs * reg_cs_loss.mean()
            # + ag.lam_c * tc.sum(c_hat**2, dim=1).mean() + ag.lam_s * tc.sum(s_hat**2, dim=1).mean()
            
            raw_loss -= reg_cs_loss.mean()
            if ag.current_epoch % 5 == 0:
                wandb.log({'epoch': ag.current_epoch, 'reg_cs_loss': reg_cs_loss.mean().item()})

        if ag.current_epoch % 5 == 0:
            wandb.log({'epoch': ag.current_epoch, 'Adapt_total_loss': raw_loss.item()})

        return reg_loss
    
    return lossfn

def process_continue_run(ag):
    # Process if continue running
    if ag.init_model not in {"rand", "fix"}: # continue running
        ckpt = load_ckpt(ag.init_model, loadmodel=False)
        # if ag.mode != ckpt['mode']: raise RuntimeError("mode not match")
        for k in vars(ag):
            if k not in {"wandb_project", "emb_model", "exp_name", "traindom","testdoms", "n_epk", "n_bat", "gpu", "deploy", "init_model", "ckp_id", "adapt_ood", "adapt_id", "lam_c", "lam_s",
                         "wada_x", "wada_e", "wreg_cs", "mode", "lr", "wl2", "save_results", "small_model", "debug", "patience",
                         "recon", "lr_discr", "lr_gen", "wl2_discr", "wl2_gen", "wdb_tag", "deploy_id", "deploy_test", "deploy_ood", "tr_val_split",
                         "online_adapt", "latent_update", "lr_l_c", "lr_l_s", "wd_l_c", "wd_l_s"}: # use the new final number of epochs
                setattr(ag, k, ckpt[k])

    else: ckpt = None
    return ag, ckpt

class InferenceCollector:
    def __init__(self):
        # Initialize empty lists to store tensors
        self.xs = []
        self.ys = []
        self.ts = []
        self.cs = []
        self.ss = []
        self.envs = []
        self.label_t = []
        self.label_e = []
        self.sample_idx = []
        self.c_hat = []
        self.s_hat = []
        self.c_std = []
        self.s_std = []
        self.x_hat = []
        self.y_hat = []
        self.e_hat = []
        self.label_subT = []

    def collect_batch(self, data_bat, c_hat, s_hat, c_std, s_std, x_hat, y_hat, e_hat):
        batch_data = {
            "X": data_bat.X,
            "Y": data_bat.Y,
            "T": data_bat.T,
            "E": data_bat.E,
            "label_T": data_bat.label_T,
            "label_E": data_bat.label_E,
            "index": data_bat.index
        }

        if hasattr(data_bat, "label_subT"):
            if data_bat.label_subT is not None:
                batch_data["label_subT"] = data_bat.label_subT
        if hasattr(data_bat, "C"):
            batch_data["C"] = data_bat.C
        if hasattr(data_bat, "S"):
            batch_data["S"] = data_bat.S
  
        batch_data = {k: v.cpu().numpy() for k, v in batch_data.items()}

        self.xs.append(batch_data["X"])
        self.ys.append(batch_data["Y"])
        self.ts.append(batch_data["T"])
        self.envs.append(batch_data["E"])
        self.label_t.append(batch_data["label_T"])
        self.label_e.append(batch_data["label_E"])
        self.sample_idx.append(batch_data["index"])

        if "label_subT" in batch_data:
            self.label_subT.append(batch_data["label_subT"])
        if "C" in batch_data:
            self.cs.append(batch_data["C"])
        if "S" in batch_data:
            self.ss.append(batch_data["S"])

        self.c_hat.append(c_hat.cpu().numpy())
        self.s_hat.append(s_hat.cpu().numpy())
        self.c_std.append(c_std.cpu().numpy())
        self.s_std.append(s_std.cpu().numpy())
        self.x_hat.append(x_hat.cpu().numpy())
        self.y_hat.append(y_hat.cpu().numpy())
        self.e_hat.append(e_hat.cpu().numpy())

    def to_dataframe(self):
        def safe_concat(lst):
            return np.vstack(lst) if len(lst) > 0 else np.array([])

        xs = safe_concat(self.xs)
        ys = safe_concat(self.ys)
        ts = safe_concat(self.ts)
        envs = safe_concat(self.envs)
        label_t = np.concatenate(self.label_t)
        label_e = np.concatenate(self.label_e)
        sample_idx = np.concatenate(self.sample_idx)
        c_hat = safe_concat(self.c_hat)
        s_hat = safe_concat(self.s_hat)
        c_std = safe_concat(self.c_std)
        s_std = safe_concat(self.s_std)
        x_hat = safe_concat(self.x_hat)
        y_hat = safe_concat(self.y_hat)
        e_hat = safe_concat(self.e_hat)

        label_subT = safe_concat(self.label_subT) if self.label_subT else np.full((len(xs),), np.nan)
        cs = safe_concat(self.cs) if self.cs else np.full((len(xs),), np.nan)
        ss = safe_concat(self.ss) if self.ss else np.full((len(xs),), np.nan)

        df = pd.DataFrame({
            'sample_idx': sample_idx.flatten(),
            'X': xs.tolist(),
            'Y': ys.tolist(),
            'T': ts.tolist(),
            'E': envs.tolist(),
            'C': cs.tolist(),
            'S': ss.tolist(),
            'Index_T': label_t.tolist(),
            'SubTask': label_subT.tolist(),
            'Index_E': label_e.tolist(),
            'C_hat': c_hat.tolist(),
            'S_hat': s_hat.tolist(),
            'C_std': c_std.tolist(),
            'S_std': s_std.tolist(),
            'X_hat': x_hat.tolist(),
            'Y_hat': y_hat.tolist(),
            'E_hat': e_hat.tolist(),
        })
        return df
    
def inference_variables(ag, lossfn_eval, frame, discr, gen, data_loader, device, phase='val', ts_val_loader=None, is_toy=False, n_mc=0,
                        sample_cs_draw=False, gen_probs=True, adapt_ood=False, optim=None):
    collector = InferenceCollector()
    total_loss, total_x_recon_loss, total_y_recon_loss, total_e_recon_loss = 0, 0, 0, 0
    total_c_recon_loss, total_s_recon_loss = 0, 0
    total_cos_sim = 0
    env_keys = []
    
    # Initialize environment loss tracking dictionary
    env_losses = {}

    mse_loss = nn.MSELoss(reduction='none').to(device)

    if adapt_ood:
        early_stopping = EarlyStopping(patience=ag.patience if hasattr(ag, 'patience') else 5, verbose=True)
        
        for e in tqdm.tqdm(range(ag.n_epk), leave=False, desc="Adapt OOD"):
            setattr(ag, 'current_epoch', e)
            epoch_adapt_loss = 0.0
            if adapt_ood: discr.train(); gen.train()
            for i_bat, data_bat in enumerate(data_loader, start=1):
                xs = data_bat.X.to(device, dtype=tc.float32)
                ys = data_bat.Y.to(device, dtype=tc.float32)
                ts = data_bat.T.to(device, dtype=tc.float32)
                envs = data_bat.E.to(device, dtype=tc.float32)

                if 'Y_num' in data_bat.keys(): y_num_s = data_bat.Y_num.to(device, dtype=tc.float32); use_fc = True
                else: use_fc = False

                if len(ys.size()) == 1:
                    ys = ys.unsqueeze(1)

                data_args = (xs, ts, envs, ys, phase)
                
                if use_fc and len(y_num_s.size()) == 1:
                    y_num_s = y_num_s.unsqueeze(1)
                    data_args = (xs, ts, envs, ys, y_num_s, phase)

                optim.zero_grad()
                adapt_loss = lossfn_eval(*data_args, 0)
                adapt_loss.backward()
                optim.step()
                epoch_adapt_loss += adapt_loss
            
            if ts_val_loader is not None:
                with tc.no_grad():
                    discr.eval(); gen.eval()
                    results_val, env_keys = inference_variables(ag, lossfn_eval, frame, discr, gen, ts_val_loader, device, phase='val',
                                                    sample_cs_draw=sample_cs_draw, gen_probs=gen_probs, is_toy=is_toy)
                    
                    flag_early_stop, flag_update_best_model = \
                        early_stopping(results_val['avg_loss'])
                    wandb.log({'epoch':e, f'current_best_total_XE_Recon_Loss': early_stopping.best_loss})

                    if flag_update_best_model or e == 0:
                        print(f"Updating best model at epoch {e}")
                        best_model_discr = discr.state_dict()
                        best_model_gen = gen.state_dict()

                        wandb.log({
                            'adapt_val_avg_Loss': results_val['avg_loss'],
                            'adapt_val_X_Recon_Loss': results_val['avg_x_recon_loss'],
                            'adapt_val_Y_Recon_Loss': results_val['avg_y_recon_loss'],
                            'adapt_val_E_Recon_Loss': results_val['avg_e_recon_loss'],
                            'adapt_val_XE_Total_Recon_Loss': results_val['avg_x_recon_loss'] + results_val['avg_e_recon_loss'],
                            'adapt_val_Cos_Sim': results_val['avg_cos_sim'],
                        })
            else:
                flag_early_stop, flag_update_best_model = \
                        early_stopping(epoch_adapt_loss/(i_bat+1))

                if flag_update_best_model or e == 0:
                    print(f"Updating best model at epoch {e}")
                    best_model_discr = discr.state_dict()
                    best_model_gen = gen.state_dict()

            print(f"Epk {e} OOD ADAPT LOSS: {epoch_adapt_loss/(i_bat+1)}")

            if flag_early_stop:
                discr.load_state_dict(best_model_discr)
                gen.load_state_dict(best_model_gen)
                frame = get_frame(discr, gen, vars(ag), device)
                break

        if ts_val_loader is not None:
            dataset1 = data_loader.dataset
            dataset2 = ts_val_loader.dataset
            combined_dataset = ConcatDataset([dataset1, dataset2])
            data_loader = DataLoader(combined_dataset, batch_size=ag.n_bat, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=4, persistent_workers=True)

    for i_bat, data_bat in tqdm.tqdm(enumerate(data_loader, start=1), leave=False, desc="Inference"):
        xs = data_bat.X.to(device, dtype=tc.float32)
        ys = data_bat.Y.to(device, dtype=tc.float32)
        ts = data_bat.T.to(device, dtype=tc.float32)
        envs = data_bat.E.to(device, dtype=tc.float32)
        label_e = data_bat.label_E.to(device)
        use_fc = False

        if len(ys.size()) == 1:
            ys = ys.unsqueeze(1)

        data_args = (xs, ts, envs, ys, phase)
        
        if use_fc and len(y_num_s.size()) == 1:
            y_num_s = y_num_s.unsqueeze(1)
            data_args = (xs, ts, envs, ys, y_num_s, phase)

        if not phase == 'test':
            total_loss += lossfn_eval(*data_args, 0)

        if n_mc == 0:
            discr.eval(); gen.eval()
            with tc.no_grad():
                if sample_cs_draw:
                    cs_samples = frame.qt_cs1x.draw((1,), {'x':xs, 't':ts, 'e':envs})
                    c_hat = cs_samples['c'].squeeze(0)
                    s_hat = cs_samples['s'].squeeze(0)
                else:
                    c_hat = frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                    s_hat = frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']
                    c_std = frame.qt_c1x.std({'x':xs, 't':ts, 'e':envs})['c']
                    s_std = frame.qt_s1x.std({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']

                    if ag.debug: import pdb; pdb.set_trace()

                if gen_probs:
                    x_hat = frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                    y_hat = frame.p_y1c.mean({'c':c_hat})['y']
                    e_hat = frame.p_e1s.mean({'s':s_hat})['e']
                else:
                    x_hat, e_hat = gen(c_hat, s_hat)
                    y_hat = discr(xs, ts, envs)

        if phase == 'test':
            collector.collect_batch(data_bat, c_hat, s_hat, c_std, s_std, x_hat, y_hat, e_hat)

        # Calculate losses per sample
        x_recon_loss_per_sample = mse_loss(x_hat, xs).mean(dim=1)
        if phase == 'test' and ag.exp_name == 'ood_nlp':
            y_recon_loss_per_sample = tc.zeros_like(x_recon_loss_per_sample)
        else:
            y_recon_loss_per_sample = mse_loss(y_hat, ys).mean(dim=1)
        e_recon_loss_per_sample = mse_loss(e_hat, envs).mean(dim=1)
        
        # More efficient environment loss tracking
        # Get environment IDs for each sample in the batch
        env_ids = label_e.view(label_e.size(0), -1).sum(dim=1).cpu().numpy()
        
        # Update environment losses in a vectorized way
        for i, env_id in enumerate(env_ids):
            env_key = f"env_{env_id}"
            if env_key not in env_losses:
                env_losses[env_key] = {
                    'x_loss': 0.0, 'y_loss': 0.0, 'e_loss': 0.0, 'count': 0
                }
            
            env_losses[env_key]['x_loss'] += x_recon_loss_per_sample[i].item()
            env_losses[env_key]['y_loss'] += y_recon_loss_per_sample[i].item()
            env_losses[env_key]['e_loss'] += e_recon_loss_per_sample[i].item()
            env_losses[env_key]['count'] += 1

        # Continue tracking total losses for overall averages
        total_x_recon_loss += x_recon_loss_per_sample.sum()
        total_y_recon_loss += y_recon_loss_per_sample.sum()
        total_e_recon_loss += e_recon_loss_per_sample.sum()
        total_cos_sim += compute_cossim(c_hat.detach().cpu().numpy(), s_hat.detach().cpu().numpy())

    # Calculate overall averages
    avg_loss = total_loss / len(data_loader)
    avg_x_recon_loss = total_x_recon_loss / len(data_loader.dataset)
    avg_y_recon_loss = total_y_recon_loss / len(data_loader.dataset)
    avg_e_recon_loss = total_e_recon_loss / len(data_loader.dataset)
    avg_cos_sim = total_cos_sim / len(data_loader.dataset)

    avg_c_recon_loss = None
    avg_s_recon_loss = None

    # Calculate per-environment averages for all phases
    for env_key, losses in env_losses.items():
        if losses['count'] > 0:
            avg_x_env = losses['x_loss'] / losses['count']
            avg_y_env = losses['y_loss'] / losses['count']
            avg_e_env = losses['e_loss'] / losses['count']

            # Log to wandb if available
            if 'wandb' in globals():
                wandb.log({
                    f"{phase}_{env_key}_X_Recon_Loss": avg_x_env,
                    f"{phase}_{env_key}_Y_Recon_Loss": avg_y_env,
                    f"{phase}_{env_key}_E_Recon_Loss": avg_e_env
                })

    results={
        'avg_loss': avg_loss.item() if type(avg_loss) == tc.Tensor else avg_loss,
        'avg_x_recon_loss': avg_x_recon_loss.item(),
        'avg_y_recon_loss': avg_y_recon_loss.item() if type(avg_y_recon_loss) == tc.Tensor else avg_y_recon_loss,
        'avg_e_recon_loss': avg_e_recon_loss.item(),
        'avg_c_recon_loss': avg_c_recon_loss,
        'avg_s_recon_loss': avg_s_recon_loss,
        'avg_cos_sim': avg_cos_sim,
    }

    # Add per-environment results to the results dictionary
    for env_key, losses in env_losses.items():
        if losses['count'] > 0:
            results[f"{env_key}_x_recon_loss"] = losses['x_loss'] / losses['count']
            results[f"{env_key}_y_recon_loss"] = losses['y_loss'] / losses['count']
            results[f"{env_key}_e_recon_loss"] = losses['e_loss'] / losses['count']
    
    env_keys = list(env_losses.keys())
    
    # Store environment losses in collector for consistency
    collector.env_losses = env_losses

    if phase == 'test':
        results_df = collector.to_dataframe()
        return results, results_df
    return results, env_keys

def main(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, val_src_loader, ts_tgt_loader, is_toy=None):
    print(ag)
    print_infrstru_info()
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")
    
    # Models
    res = get_models(archtype, edic(locals()) | vars(ag), ckpt, device)
    discr, gen, frame = res
    discr.train()
    gen.train()

    lossfn = ood_methods(discr, frame, ag)

    # Optimizer
    pgc = ParamGroupsCollector()
    pgc.collect_params(discr, ag.lr, ag.wl2)
    pgc.collect_params(gen, ag.lr, ag.wl2)

    opt = getattr(tc.optim, ag.optim)(pgc.param_groups, weight_decay=ag.wl2)
    auto_load(locals(), 'opt', ckpt)

    res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)
    print(f"Run in mode '{ag.mode}' for {ag.n_epk} epochs:")

    early_stopping = EarlyStopping(
        patience=ag.patience if hasattr(ag, 'patience') else 10,
        verbose=True
    )

    mse_loss = nn.MSELoss()

    epk0 = 1
    n_per_epk = len(tr_src_loader)
    _filename = None
    for epk in range(epk0, ag.n_epk+1):
        setattr(ag, 'current_epoch', epk)
        n_min_batch, epk_loss = 0, 0
        pbar = tqdm.tqdm(total=n_per_epk, desc=f"Train epoch = {epk}", ncols=80, leave=False)
        for i_bat, data_bat in enumerate(tr_src_loader, start=1):

            n_min_batch = data_bat.index.size(0)
            xs = data_bat.X.to(device, dtype=tc.float32)
            ys = data_bat.Y.to(device, dtype=tc.float32)
            if ag.use_fc: y_num_s = data_bat.Y_num.to(device, dtype=tc.float32)

            if len(ys.size()) == 1:
                ys = ys.unsqueeze(1)
            if ag.use_fc and len(y_num_s.size()) == 1:
                y_num_s = y_num_s.unsqueeze(1)

            ts = data_bat.T.to(device, dtype=tc.float32)
            envs = data_bat.E.to(device, dtype=tc.float32)

            if ag.use_fc: data_args = (xs, ts, envs, ys, y_num_s, 'tr')
            else: data_args = (xs, ts, envs, ys, 'tr')

            opt.zero_grad()

            n_iter_tot = (epk-1)*n_per_epk + i_bat-1
            loss = lossfn(*data_args, n_iter_tot)

            if tc.isnan(loss):
                print("\n Loss is NaN, stopping training")
                return
            
            loss.backward()
            opt.step()
            pbar.update(1)

            epk_loss += loss.item()

        pbar.close()

        if epk % ag.eval_interval == 0:
            epk_loss /= n_min_batch
            res.update(epk=epk, loss=epk_loss)
            print(f"Mode '{ag.mode}': Epoch {epk}, Tr Loss = {epk_loss:.3f},")

            with tc.no_grad():
                discr.eval(); gen.eval()
                lossfn_eval = ood_methods(discr, frame, ag)

                setattr(ag, 'log_loss', True)
                results_tr, tr_env_keys = inference_variables(ag, lossfn_eval, frame, discr, gen, tr_src_loader, device, phase='tr',
                                                  sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
                
                results_val, val_env_keys = inference_variables(ag, lossfn_eval, frame, discr, gen, val_src_loader, device, phase='val',
                                                  sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs, is_toy=is_toy)              
                print_results(ag, RESULTS_DIC)

                if ts_tgt_loader is not None:
                    results_test, _ = inference_variables(ag, lossfn_eval, frame, discr, gen, ts_tgt_loader, device, phase='test',
                                                      sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
                else: results_test = None

                print(f"Mode '{ag.mode}': Epoch {epk}, Val Loss = {results_val['avg_loss']:.3f}")
                if results_test is not None: print(f"Mode '{ag.mode}': Epoch {epk}, Test Loss = {results_test['avg_loss']:.3f}")

                if ag.early_stop_metric == 'sup': es_metric = RESULTS_DIC['val_Loss_sup']
                elif ag.early_stop_metric == 'total_loss': es_metric = RESULTS_DIC['val_total_Loss']
                elif ag.early_stop_metric == 'y_emb': es_metric = results_val['avg_y_recon_loss']
                elif ag.early_stop_metric == 'val_loss': es_metric = results_val['avg_loss']

                flag_early_stop, flag_update_best_model = early_stopping(es_metric)
                wandb.log({'epoch':epk, f'current_best_{ag.early_stop_metric}': early_stopping.best_loss})
                if flag_update_best_model and not ag.no_save:
                    _filename = save_ckpt(ag, ckpt, discr, gen, frame, opt,
                                              edic(locals()) | vars(ag),
                                              is_toy=is_toy, filename=_filename)
                if flag_early_stop:
                    print("Early stopping triggered")
                    return _filename, frame
                
                num_tr_envs = len(tr_env_keys)
                num_val_envs = len(val_env_keys)
                if flag_update_best_model:
                    wdb_results = {
                        'epoch': epk,
                        'Tr_avg_Loss': epk_loss,
                        'Tr_X_Recon_Loss': results_tr['avg_x_recon_loss'],
                        'Tr_Y_Recon_Loss': results_tr['avg_y_recon_loss'],
                        'Tr_E_Recon_Loss': results_tr['avg_e_recon_loss'],
                        'Tr_Cos_Sim': results_tr['avg_cos_sim'],

                        'Val_avg_Loss': results_val['avg_loss'],
                        'Val_X_Recon_Loss': results_val['avg_x_recon_loss'],
                        'Val_Y_Recon_Loss': results_val['avg_y_recon_loss'],
                        'Val_E_Recon_Loss': results_val['avg_e_recon_loss'],
                        'Val_Cos_Sim': results_val['avg_cos_sim'],
                    }

                    for env_key in tr_env_keys:
                        wdb_results[f'Tr_{env_key}_X_Recon_Loss'] = results_tr[f'{env_key}_x_recon_loss']
                        wdb_results[f'Tr_{env_key}_Y_Recon_Loss'] = results_tr[f'{env_key}_y_recon_loss']
                        wdb_results[f'Tr_{env_key}_E_Recon_Loss'] = results_tr[f'{env_key}_e_recon_loss']
                    for env_key in val_env_keys:
                        wdb_results[f'Val_{env_key}_X_Recon_Loss'] = results_val[f'{env_key}_x_recon_loss']
                        wdb_results[f'Val_{env_key}_Y_Recon_Loss'] = results_val[f'{env_key}_y_recon_loss']
                        wdb_results[f'Val_{env_key}_E_Recon_Loss'] = results_val[f'{env_key}_e_recon_loss']

                    wandb.log(wdb_results)

                    if results_test is not None:
                        wandb.log({
                            'Te_avg_Loss': results_test['avg_loss'],
                            'Te_X_Recon_Loss': results_test['avg_x_recon_loss'],
                            'Te_Y_Recon_Loss': results_test['avg_y_recon_loss'],
                            'Te_E_Recon_Loss': results_test['avg_e_recon_loss'],
                            'Te_Cos_Sim': results_test['avg_cos_sim']
                    })

            discr.train(); gen.train(); setattr(ag, 'log_loss', False)

    print("Training finished")
    print("Validation loss did not saturated till the end of training")
    if not ag.no_save:
        _filename = save_ckpt(ag, ckpt, discr, gen, frame, opt,
                              edic(locals()) | vars(ag),
                              is_toy=is_toy, filename=_filename)
    return _filename, frame

def main_depoly(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader,
        ts_tgt_loader = None,
        ts_val_loader = None,
        is_toy = None,
        retrieval_eval = None
    ):
    print(ag)
    print_infrstru_info()
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Models
    dc_vars = edic(locals()) | vars(ag) | \
        {'dim_x':ckpt['dim_x'], 'dim_y':ckpt['dim_y'],}
    
    setattr(ag, 'dim_x', ckpt['dim_x'])
    setattr(ag, 'dim_y', ckpt['dim_y'])

    res = get_models(archtype, dc_vars, ckpt, device)
    discr, gen, frame = res

    mse_loss = nn.MSELoss()

    if ag.adapt_ood:
        if ag.online_adapt or ag.latent_update:
            adaptor = Adaptor.Adaptor(ag, discr, gen)

            if ts_val_loader is not None:
                dataset1 = ts_tgt_loader.dataset
                dataset2 = ts_val_loader.dataset
                combined_dataset = ConcatDataset([dataset1, dataset2])
                ts_tgt_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=False, prefetch_factor=4, persistent_workers=True, collate_fn=namespace_collate_fn)

            if ag.latent_update: adaptor.adapt_latent(ts_tgt_loader, device)
            else: adaptor.adapt_para(ts_tgt_loader, device)
            results_OOD_df = adaptor.get_dataframe()

            discr.load_state_dict(adaptor.original_discr_params)
            gen.load_state_dict(adaptor.original_gen_params)

    if not ag.adapt_ood:
        res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)
        with tc.no_grad():
            discr.eval(); gen.eval()
            lossfn_eval = ood_methods(discr, frame, ag)
            setattr(ag, 'log_loss', True)
            setattr(ag, 'current_epoch', 0)

            # Run inference for ID data if needed
            ckp_id = ag.ckp_id
            results_id_path = f"./results/{ag.exp_name}/results_{ag.mode}_{ag.emb_model}_emb_{ckp_id}_ID.feather"
            if ag.deploy_id or (ag.save_results and not os.path.exists(results_id_path)):
                results_ID, results_ID_df = inference_variables(ag, lossfn_eval, frame, discr, gen, tr_src_loader, device,
                                                             phase='test', sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
                
                print(f"Mode '{ag.mode}': ID Recon Loss - X: {results_ID['avg_x_recon_loss']}, E: {results_ID['avg_e_recon_loss']}, Y: {results_ID['avg_y_recon_loss']}")

                wandb.log({
                    'ID_Loss': results_ID['avg_loss'],
                    'ID_X_Recon_Loss': results_ID['avg_x_recon_loss'],
                    'ID_Y_Recon_Loss': results_ID['avg_y_recon_loss'], 
                    'ID_E_Recon_Loss': results_ID['avg_e_recon_loss'],
                    'ID_Cos_Sim': results_ID['avg_cos_sim'],
                })
            
            # Run inference for OOD data if not adapting
            if (not ag.adapt_ood) and (ag.deploy_test or ag.deploy_ood):
                results_OOD, results_OOD_df = inference_variables(ag, lossfn_eval, frame, discr, gen, ts_tgt_loader, device,
                                                               phase='test', sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs, is_toy=is_toy)

                print(f"Mode '{ag.mode}': OOD Recon Loss - X: {results_OOD['avg_x_recon_loss']}, E: {results_OOD['avg_e_recon_loss']}, Y: {results_OOD['avg_y_recon_loss']}")

                # Log OOD results
                wandb.log({
                    'OOD_Loss': results_OOD['avg_loss'],
                    'OOD_X_Recon_Loss': results_OOD['avg_x_recon_loss'], 
                    'OOD_Y_Recon_Loss': results_OOD['avg_y_recon_loss'],
                    'OOD_E_Recon_Loss': results_OOD['avg_e_recon_loss'],
                    'OOD_Cos_Sim': results_OOD['avg_cos_sim']
                })

                # Log ID results and differences if ID inference was done
                if ag.deploy_id or (ag.save_results and not os.path.exists(results_id_path)):
                    print(f"Mode '{ag.mode}': ID-OOD Abs Diff - X: {abs(results_ID['avg_x_recon_loss'] - results_OOD['avg_x_recon_loss'])}, "
                        f"E: {abs(results_ID['avg_e_recon_loss'] - results_OOD['avg_e_recon_loss'])}, "
                        f"Y: {abs(results_ID['avg_y_recon_loss'] - results_OOD['avg_y_recon_loss'])}")

                    wandb.log({
                        'Abs_Diff_X_Recon_Loss': abs(results_ID['avg_x_recon_loss'] - results_OOD['avg_x_recon_loss']),
                        'Abs_Diff_E_Recon_Loss': abs(results_ID['avg_e_recon_loss'] - results_OOD['avg_e_recon_loss']),
                        'Abs_Diff_Y_Recon_Loss': abs(results_ID['avg_y_recon_loss'] - results_OOD['avg_y_recon_loss'])
                    })

            if is_toy:
                wandb.log({
                    'ID_C_Recon_Loss': results_ID['avg_c_recon_loss'],
                    'OOD_C_Recon_Loss': results_OOD['avg_c_recon_loss'],
                    'ID_S_Recon_Loss': results_ID['avg_s_recon_loss'],
                    'OOD_S_Recon_Loss': results_OOD['avg_s_recon_loss'],
                })
    
    if ag.save_results:
        if not ag.exp_name in ['toy', 'mgsm']: _path = f"{ag.exp_name}/{ag.traindom}/{ag.testdoms.split('/')[-1]}"
        else: _path = ag.exp_name

        ckp_id = ag.ckp_id
        results_id_path = f"./results/{ag.exp_name}/results_{ag.mode}_{ag.emb_model}_emb_{ckp_id}_ID.feather"
        if not os.path.exists(results_id_path):
            results_ID_df.to_feather(results_id_path)
        else: print(f"Results already saved to {results_id_path}")
        
        if ag.deploy_test or ag.deploy_ood:
            results_adapt_path = f"./results/{_path}/{ag.adapt_id}/results_{ag.mode}_{ag.emb_model}_emb_{ag.traindom}_{ag.testdoms.split('/')[-1]}_{ckp_id}_ada_OOD.feather"
            if not os.path.exists(f"./results/{_path}/{ag.adapt_id}"): os.makedirs(f"./results/{_path}/{ag.adapt_id}")

            if ag.adapt_ood == True: results_OOD_df.to_feather(results_adapt_path); print(f"Results saved to {results_adapt_path}")
            else:
                results_OOD_df.to_feather(f"./results/{_path}/results_{ag.mode}_{ag.emb_model}_emb_{ag.traindom}_{ag.testdoms.split('/')[-1]}_{ckp_id}_OOD.feather")
                file_path = f"./results/{_path}/results_{ag.mode}_{ag.emb_model}_emb_{ag.traindom}_{ag.testdoms.split('/')[-1]}_{ckp_id}_OOD.feather"
                print(f"Results saved to {file_path}")
        
    if (ag.deploy_test or ag.deploy_ood) and retrieval_eval is not None:
        retrieval_eval.set_results_ood(results_OOD_df)
        for embs in ['c_hat']: # , 'c_norm'
            for metric in ['cossim']: # , 'knn'
                print(f"\nEvaluating retrieval with {embs} and {metric}...\n")
                ret_results = retrieval_eval(embs, metric)
                for k, v in ret_results.items():
                    wandb.log({f"{embs}_{metric}_{k}": v})

    return frame

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--gpu", type=int, default = 0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type = str, default='ccl')
    parser.add_argument("--exp_name", type = str, choices=["mgsm", "toy", "ood_nlp", "llm_ret"])
    parser.add_argument("--emb_model", type=str, default='gpt')
    parser.add_argument("--debug", type=boolstr, default=False)
    parser.add_argument("--early_stop_metric", type=str, default='total_loss', choices=['val_loss', 'sup', 'y_emb','total_loss'])
    parser.add_argument("--verbose", type=boolstr, default=False)
    parser.add_argument("--deploy", type=boolstr, default=False)
    parser.add_argument("--deploy_id", type=boolstr, default=False)
    parser.add_argument("--deploy_test", type=boolstr, default=False)
    parser.add_argument("--deploy_ood", type=boolstr, default=False)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="CCL_v2")
    parser.add_argument("--wdb_tag", type=str, default="")
    parser.add_argument("--log_loss", type=boolstr, default=False)

    # Data
    parser.add_argument("--traindom", type = str)
    parser.add_argument("--testdoms", type = str)
    parser.add_argument("--sample_ratio", type = float, default = 1.0)
    parser.add_argument("--tr_val_split", type = float, default = .7)
    parser.add_argument("--n_bat", type = int, default = 128)
    parser.add_argument("--y_dtype", type = str, default = 'emb')
    parser.add_argument("--y_emb_option", type = str, default='default', choices=['default', 'long', 'eq'])
    parser.add_argument("--subtask", type = boolstr, default=False)

    # Model
    parser.add_argument("--ckp_id", type = int, default = 0)
    parser.add_argument("--init_model", type = str, default = "rand") # or a model file name to continue running
    parser.add_argument("--discrstru", type = str)
    parser.add_argument("--genstru", type = str)
    parser.add_argument("--mlpstrufile", type = str, default = "./arch/mlpstru.json")
    parser.add_argument("--use_fc", type=boolstr, default=False)
    parser.add_argument("--model_size", type=str, default='small')
    parser.add_argument("--small_model", type=boolstr, default=False)
    parser.add_argument("--large_model", type=boolstr, default=False)

    # Process
    parser.add_argument("--n_epk", type = int, default = 800)
    parser.add_argument("--eval_interval", type = int, default = 5)
    parser.add_argument("--patience", type = int, default = 100)
    parser.add_argument("--no_save", type=boolstr, default=True)
    parser.add_argument("--save_results", type=boolstr, default=False)
    parser.add_argument("--elbo_only", type=boolstr, default=False)

    # Optimization
    parser.add_argument("--optim", type = str, default="Adam")
    parser.add_argument("--lr", type = float, default=1e-3)
    parser.add_argument("--wl2", type = float, default=1e-2)
    parser.add_argument("--reduction", type = str, default = "mean")
    parser.add_argument("--reg_cs", type = str, default = "cossim")
    parser.add_argument("--wreg_cs", type = float, default = -1.0)
    parser.add_argument("--gamma_mmd", type = float, default = 1)

    # Supervised loss weight
    parser.add_argument("--wsup", type = float, default = 1.)
    parser.add_argument("--wsup_expo", type = float, default = 0.75) # only when "wsup" is not 0
    parser.add_argument("--wsup_wdatum", type = float, default = 6.25e-6) # only when "wsup" and "wsup_expo" are not 0
    
    # ELBO weight
    parser.add_argument("--recon", type = boolstr, default = False)
    parser.add_argument("--wgen", type = float, default = 1.)
    parser.add_argument("--wlogpi", type = float, default = 1.)
    parser.add_argument("--wrecon_x", type = float, default = 1.)
    parser.add_argument("--wrecon_e", type = float, default = 1.)
    parser.add_argument("--wrecon_c", type = float, default = 1.)
    parser.add_argument("--wrecon_s", type = float, default = 1.)

    # For Prior
    parser.add_argument("--actv", type = str, default = "ReLU")
    parser.add_argument("--after_actv", type = boolstr, default = False)
    parser.add_argument("--mu_c", type = float, default = 0.01)
    parser.add_argument("--sig_c", type = float, default = -1.)
    parser.add_argument("--mu_s", type = float, default = 0.01)
    parser.add_argument("--sig_s", type = float, default = 1.)
    parser.add_argument("--cond_prior", type = boolstr, default = True)
    parser.add_argument("--actv_prior", type = str, default = "xtanh")
    parser.add_argument("--ind_cs", type = boolstr, default = True)

    # For Inference models
    parser.add_argument("--n_mc_q", type = int, default = 0)
    parser.add_argument("--only_x4c", type = boolstr, default = False)

    # For Generation models
    parser.add_argument("--pstd_x", type = float, default = 3e-1)
    parser.add_argument("--pstd_t", type = float, default = 3e-1)
    parser.add_argument("--pstd_e", type = float, default = 3e-1)
    
    parser.add_argument("--true_sup", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--wda", type = float, default = .25)

    # Deploy settings
    parser.add_argument("--sample_cs_draw", type=boolstr, default=False)
    parser.add_argument("--gen_probs", type=boolstr, default=True)
    
    # Adapt OOD settings
    parser.add_argument("--adapt_ood", type=boolstr, default=False)
    parser.add_argument("--online_adapt", type=boolstr, default=False)
    parser.add_argument("--latent_update", type=boolstr, default=False)
    parser.add_argument("--adapt_id", type=str, default="random")
    parser.add_argument("--lr_l_c", type=float, default=1e-4, help="learning rate for latent variable C")
    parser.add_argument("--lr_l_s", type=float, default=1e-4, help="learning rate for latent variable S")
    parser.add_argument("--wd_l_c", type=float, default=1e-3, help="weight decay for latent variable C")
    parser.add_argument("--wd_l_s", type=float, default=1e-3, help="weight decay for latent variable S")
    parser.add_argument("--lr_discr", type = float)
    parser.add_argument("--lr_gen", type = float)
    parser.add_argument("--wl2_discr", type = float)
    parser.add_argument("--wl2_gen", type = float)
    parser.add_argument("--wada_x", type = float, default = 1)
    parser.add_argument("--wada_e", type = float, default = 1)
    parser.add_argument("--lam_c", type = float, default = 1e-3)
    parser.add_argument("--lam_s", type = float, default = 1e-3)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    ag = parser.parse_args()
    if ag.sample_cs_draw: print("sample_cs_draw is True"); sys.exit(0)

    ag, ckpt = process_continue_run(ag)
    archtype = "mlp"

    if ag.init_model != 'rand': setattr(ag, 'ckp_id', ag.init_model.split('_')[-1].split('.')[0])

    if ag.lr_discr is None: ag.lr_discr = ag.lr
    if ag.lr_gen is None: ag.lr_gen = ag.lr
    if ag.wl2_discr is None: ag.wl2_discr = ag.wl2
    if ag.wl2_gen is None: ag.wl2_gen = ag.wl2

    set_seed_all(ag.seed)

    # Dataset
    dataset_id, dataset_ood, y_dtype= load_dataset(ag, ag.exp_name, ag.traindom, ag.testdoms, ag.emb_model, ag.y_dtype, ag.y_emb_option)

    if not ag.deploy_ood: sample = dataset_id[0]
    else: sample = dataset_ood[0]
    shape_x = sample.X.shape
    dim_t = sample.T.shape[-1]
    dim_e = sample.E.shape[-1]

    if y_dtype == 'emb':
        dim_y = sample.Y.shape[-1]
    elif y_dtype == 'regression':
        dim_y = 1
    elif y_dtype == 'classification':
        dim_y = dataset_id.num_class
    else:
        raise ValueError(f"unknown `y_dtype` '{y_dtype}'")

    print(f"Input X shape: {shape_x}")
    setattr(ag, 'dim_x', shape_x[-1])
    setattr(ag, 'dim_t', dim_t)
    setattr(ag, 'dim_e', dim_e)

    if ag.exp_name == 'toy':
        setattr(ag, 'discrstru', 'toy')
    else:
        if ag.model_size == 'small': setattr(ag, 'discrstru', f'{ag.exp_name}_{ag.emb_model}_small')
        elif ag.model_size == 'middle': setattr(ag, 'discrstru', f'{ag.exp_name}_{ag.emb_model}_middle')
        elif ag.model_size == 'large': setattr(ag, 'discrstru', f'{ag.exp_name}_{ag.emb_model}_large')
    print(ag.discrstru)

    # Convert SimpleNamespace to a proper dataset if needed
    if isinstance(dataset_id, types.SimpleNamespace):
        dataset_id = HighDimSCMRealWorldDataset(
            X=dataset_id.X,
            Y=dataset_id.Y,
            T=dataset_id.T,
            E=dataset_id.E,
            label_T=dataset_id.label_T,
            label_E=dataset_id.label_E,
            index=np.arange(len(dataset_id.X))
        )

    # DataLoader
    if ag.deploy:
        if isinstance(dataset_ood, types.SimpleNamespace):
            dataset_ood = HighDimSCMRealWorldDataset(
                X=dataset_ood.X,
                Y=dataset_ood.Y,
                T=dataset_ood.T,
                E=dataset_ood.E,
                label_T=dataset_ood.label_T,
                label_E=dataset_ood.label_E,
                index=np.arange(len(dataset_ood.X))
            )

        tr_src_loader = DataLoader(dataset_id, batch_size=ag.n_bat, shuffle=False, 
                                  drop_last=False, pin_memory=False, prefetch_factor=4, persistent_workers=True, 
                                  num_workers=2, collate_fn=custom_collate_fn)
        if ag.adapt_ood:
            total_len = len(dataset_ood)
            train_len = int(total_len * ag.tr_val_split)
            val_len = total_len - train_len

            print("Adapt OOD")
            print(f"Train len: {train_len}, Val len: {val_len}")

            if train_len < ag.n_bat:
                setattr(ag, 'n_bat', int(train_len//5))

            if ag.tr_val_split == 1:
                # Use entire dataset for training if split ratio is 1
                ts_tgt_loader = DataLoader(dataset_ood, batch_size=ag.n_bat, shuffle=True, 
                                         pin_memory=False, prefetch_factor=4, persistent_workers=True, num_workers=2,
                                         collate_fn=custom_collate_fn)
                ts_val_loader = None
            else:
                # Split dataset into train and validation if split ratio < 1
                train_dataset, val_dataset = random_split(dataset_ood, [train_len, val_len])
                ts_tgt_loader = DataLoader(train_dataset, batch_size=ag.n_bat, shuffle=True,
                                         pin_memory=False, prefetch_factor=4, persistent_workers=True, num_workers=2,
                                         collate_fn=custom_collate_fn)
                ts_val_loader = DataLoader(val_dataset, batch_size=ag.n_bat, shuffle=True,
                                         pin_memory=False, prefetch_factor=4, persistent_workers=True, num_workers=2,
                                         collate_fn=custom_collate_fn)
        else:
            ts_tgt_loader = DataLoader(dataset_ood, batch_size=ag.n_bat, shuffle=True,
                                     pin_memory=False, prefetch_factor=4, persistent_workers=True, num_workers=2,
                                     collate_fn=custom_collate_fn)
            ts_val_loader = None

        setattr(ag, 'mode', 'ind')
    else:
        # Get all indices and labels
        all_indices = np.arange(len(dataset_id))
        labels_T = np.array([data.label_T for data in dataset_id])
        labels_E = np.array([data.label_E for data in dataset_id])

        if ag.sample_ratio < 1.0:
            # Calculate number of samples needed
            n_samples = int(len(dataset_id) * ag.sample_ratio)
            
            # Stratified sampling based on label_T
            sampled_indices = []
            for label in np.unique(labels_T):
                label_indices = all_indices[labels_T == label]
                n_label_samples = int(np.ceil(n_samples * len(label_indices) / len(dataset_id)))
                sampled_label_indices = np.random.choice(label_indices, size=n_label_samples, replace=False)
                sampled_indices.extend(sampled_label_indices)
            
            # Convert to numpy array and shuffle
            sampled_indices = np.array(sampled_indices)
            np.random.shuffle(sampled_indices)
            
            # Take only n_samples in case we have slightly more due to rounding
            sampled_indices = sampled_indices[:n_samples]
            
            # Update indices and labels for train-test split
            all_indices = sampled_indices
            labels_E = labels_E[sampled_indices]

        # Perform train-test split stratified by label_E
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=1 - ag.tr_val_split,
            random_state=ag.seed,
            stratify=labels_E
        )

        train_dataset = Subset(dataset_id, train_idx)
        val_dataset = Subset(dataset_id, val_idx)

        tr_src_loader = DataLoader(train_dataset, batch_size=ag.n_bat, shuffle=True,
                                 drop_last=False, num_workers=2, pin_memory=False,
                                 prefetch_factor=4, persistent_workers=True, collate_fn=custom_collate_fn)
        val_src_loader = DataLoader(val_dataset, batch_size=ag.n_bat, shuffle=True,
                                      num_workers=2, pin_memory=False, prefetch_factor=4, persistent_workers=True,
                                      collate_fn=custom_collate_fn)
        ts_tgt_loader = None
        ts_val_loader = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        wandb.init(
                project=ag.wandb_project,
                config=ag,
                save_code=True,
                dir=tmp_dir,
                notes=ag.wdb_tag
            )
        
        if ag.sample_cs_draw: wandb.finish()
        if ag.deploy:
            # if ag.adapt_ood and ag.exp_name == 'ood_nlp': retrieval_eval = RetrievalEval(ag)
            # else: retrieval_eval = None
            retrieval_eval = None
            frame = main_depoly(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, ts_tgt_loader, ts_val_loader, is_toy=ag.exp_name=='toy', retrieval_eval=retrieval_eval)
        else:
            filename, frame = main(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, val_src_loader, ts_tgt_loader, is_toy=ag.exp_name=='toy')
            print(f"checkpoint saved to '{filename}'.")
            if ag.verbose:
                device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")
                er = EvalRetrival(device, dataset_id, sample_cs_draw=ag.sample_cs_draw, data_dir=ag.exp_name)
                er(frame)

            wandb.finish()

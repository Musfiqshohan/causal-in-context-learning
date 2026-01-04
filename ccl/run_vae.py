import os
import sys
import argparse
import tqdm
import tempfile
import pandas as pd
import numpy as np
import torch as tc
import torch.nn.functional as F

import distr as ds

from torch.utils.data import DataLoader, random_split

from distr import edic
from arch import mlp
from methods import SupVAE

from copy import deepcopy
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccl.utils import Averager, unique_filename, boolstr, zip_longer, print_infrstru_info, EarlyStopping, compute_cossim, EvalRetrival

from src.utils import set_seed_all

# Synthetic Data
from utils_data import HighDimSCMRealWorldDataset, load_dataset

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

import wandb

class ParamGroupsCollector:
    def __init__(self, lr):
        self.reset(lr)

    def reset(self, lr):
        self.lr = lr
        self.param_groups = []

    def collect_params(self, *models):
        for model in models:
            if hasattr(model, 'parameter_groups'):
                groups_inc = list(model.parameter_groups())
                for grp in groups_inc:
                    if 'lr_ratio' in grp:
                        grp['lr'] = self.lr * grp['lr_ratio']
                    elif 'lr' not in grp: # Do not overwrite existing lr assignments
                        grp['lr'] = self.lr
                self.param_groups += groups_inc
            else:
                self.param_groups += [
                        {'params': model.parameters(), 'lr': self.lr} ]

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
    shape_z = discr.shape_z if hasattr(discr, "shape_z") else (dc_vars['dim_z'],)
    
    mean_z1x=discr.z1x
    std_z1x = discr.std_z1x if hasattr(discr, "std_z1x") else dc_vars['qstd_z']

    if dc_vars['cond_prior']:
        prior_std = mlp.create_prior_from_json("MLPc1t", discr, actv=dc_vars['actv_prior'],jsonfile=dc_vars['mlpstrufile']).to(device)
        std_z = prior_std.std_c1t
        print("Conditional prior std")
        print(prior_std)
    else:
        std_z = dc_vars['sig_z']

    frame = SupVAE(shape_z=shape_z, shape_x=shape_x, dim_y=dc_vars['dim_y'],
                   mean_x1z=gen.x1z, std_x1z=dc_vars['pstd_x'],
                   mean_y1z=discr.y1z, std_y1z=discr.std_y1z,
                   tmean_z1x=mean_z1x, tstd_z1x=std_z1x,
                   mean_z=dc_vars['mu_z'], std_z=std_z,
                   device=device)
    return frame

def get_discr(archtype, dc_vars):
    if archtype == "mlp":
        discr = mlp.create_vae_discr_from_json(
                *dc_vars.sublist([
                    'discrstru', 'dim_x', 'dim_y', 'dim_t', 'actv']),
                    jsonfile=dc_vars['mlpstrufile']
            )
    else: raise ValueError(f"unknown `archtype` '{archtype}'")
    return discr

def get_gen(archtype, dc_vars, discr):
    if archtype == "mlp":
        gen = mlp.create_gen_from_json(
            "MLPx1z", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
    return gen

def get_models(archtype, dc_vars, ckpt = None, device = None):
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
    dirname = "ckpt_" + ag.mode + "/" + ag.traindom + "/"
    os.makedirs(dirname, exist_ok=True)
    i = 0
    testdoms = ag.testdoms
    dim_x = dc_vars['dim_x']
    dim_y = dc_vars['dim_y']
    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_z = discr.shape_z if hasattr(discr, "shape_z") else (dc_vars['dim_z'],)

    if filename is None:
        filename = unique_filename(
                dirname + (f"vae_toy" if is_toy else f"vae_{testdoms}_{ag.emb_model}"), ".pt", n_digits = 3
            ) if ckpt is None else ckpt['filename']
        
    dc_vars = edic(locals()).sub([
            'dirname', 'filename', 'testdoms',
            'shape_x', 'shape_z', 'dim_x', 'dim_y']) | ( edic(vars(ag)) - {'testdoms'}
        ) | dc_state_dict(locals(), "discr", "gen", "frame", "opt")

    tc.save(dc_vars, filename)
    return filename
    
def load_ckpt(filename: str, loadmodel: bool=False, device: tc.device=None, archtype: str="mlp", map_location: tc.device=None):
    ckpt = tc.load(filename, map_location)
    if loadmodel:
        return (ckpt,) + get_models(archtype, ckpt, ckpt, device)
    else: return ckpt

# Built methods
def get_ce_or_bce_loss(discr, y_dtype: int, reduction: str="mean", mode='ind'):
    if y_dtype == 'clf_binary': lossobj = tc.nn.BCEWithLogitsLoss(reduction=reduction)
    elif y_dtype == 'clf_multi': lossobj = tc.nn.CrossEntropyLoss(reduction=reduction)
    elif y_dtype == 'regression': lossobj = tc.nn.MSELoss(reduction=reduction)
    elif y_dtype == 'emb': lossobj = tc.nn.MSELoss(reduction=reduction)
    
    if y_dtype == 'regression':
        lossfn = lambda x: tc.sqrt(lossobj(discr(x[0]), x[-1]))
    else:
        lossfn = lambda x: lossobj(discr(x[0]), x[-1])
    return lossobj, lossfn

def add_ce_loss(lossobj, celossfn, ag):
    # shrink_sup = ShrinkRatio(w_iter=ag.wsup_wdatum*ag.n_bat, decay_rate=ag.wsup_expo)

    def lossfn(*x_y_maybext_niter):
        state = x_y_maybext_niter[-2]
        if ag.mode == 'ind':
            log_phi_loss, x_recon_loss, z_recon_loss = lossobj(*x_y_maybext_niter[:-2]) # [batch_size]

            elbo = ag.wlogpi * log_phi_loss
            elbo += ag.wrecon_x * x_recon_loss
            elbo += ag.wrecon_z * z_recon_loss

            elbo = -1*elbo.mean()

            if not ag.elbo_only: celoss = celossfn(x_y_maybext_niter[:-2])

        if ag.log_loss:
            results = {
                            f"Epoch": ag.current_epoch,
                            f"{state}_Elbo": elbo.item(),
                            f"{state}_log_phi_loss": -1*log_phi_loss.mean().item(),
                            f"{state}_Loss_x": -1*x_recon_loss.mean().item(),
                            f"{state}_Loss_z": -1*z_recon_loss.mean().item(),
                        }

            if not ag.elbo_only:
                results[f'{state}_Loss_sup'] = celoss.item()

            wandb.log(results)

        if ag.elbo_only: return ag.wgen * elbo
        else: return ag.wgen * elbo + ag.wsup * celoss
    return lossfn

def ood_methods(discr, frame, ag):
    if ag.true_sup:
        celossfn = get_ce_or_bce_loss(partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q), ag.y_dtype, ag.reduction)[1]
    else:
        celossfn = get_ce_or_bce_loss(discr, ag.y_dtype, ag.reduction)[1]
    
    if ag.mode == 'ind' and not ag.debug:
        loss_mode = 'ccl'
    elif ag.mode == 'ind' and ag.debug:
        loss_mode = 'debug'
    
    lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, loss_mode)

    lossfn = add_ce_loss(lossobj, celossfn, ag)
    return lossfn

def process_continue_run(ag):
    # Process if continue running
    if ag.init_model not in {"rand", "fix"}: # continue running
        ckpt = load_ckpt(ag.init_model, loadmodel=False)
        if ag.mode != ckpt['mode']: raise RuntimeError("mode not match")
        for k in vars(ag):
            if k not in {"testdoms", "n_epk", "gpu", "deploy", "init_model"}: # use the new final number of epochs
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
        self.z_hat = []
        self.x_hat = []
        self.y_hat = []
        self.label_subT = []

    def collect_batch(self, data_bat, z_hat, x_hat, y_hat):
        """Collect tensors from a batch"""
        # Move tensors to CPU and convert to numpy
        self.xs.append(data_bat['X'].cpu().numpy())
        self.ys.append(data_bat['Y'].cpu().numpy())
        self.ts.append(data_bat['T'].cpu().numpy())
        self.envs.append(data_bat['E'].cpu().numpy())
        self.label_t.append(data_bat['label_T'].cpu().numpy())
        self.label_e.append(data_bat['label_E'].cpu().numpy())
        self.sample_idx.append(data_bat['index'].cpu().numpy())

        if 'label_subT' in data_bat.keys():
            self.label_subT.append(data_bat['label_subT'].cpu().numpy())

        if 'C' in data_bat.keys():
            self.cs.append(data_bat['C'].cpu().numpy())
        if 'S' in data_bat.keys():
            self.ss.append(data_bat['S'].cpu().numpy())

        self.z_hat.append(z_hat.cpu().numpy())
        self.x_hat.append(x_hat.cpu().numpy())
        self.y_hat.append(y_hat.cpu().numpy())

    def to_dataframe(self):
        """Convert collected data to pandas DataFrame"""
        # Concatenate all batches
        xs = np.concatenate(self.xs, axis=0)
        ys = np.concatenate(self.ys, axis=0)
        ts = np.concatenate(self.ts, axis=0)
        envs = np.concatenate(self.envs, axis=0)
        label_t = np.concatenate(self.label_t, axis=0)
        label_e = np.concatenate(self.label_e, axis=0)
        sample_idx = np.concatenate(self.sample_idx, axis=0)
        z_hat = np.concatenate(self.z_hat, axis=0)
        x_hat = np.concatenate(self.x_hat, axis=0)
        y_hat = np.concatenate(self.y_hat, axis=0)

        if len(self.label_subT) != 0: label_subT = np.concatenate(self.label_subT, axis=0)
        else: label_subT = np.array([np.nan] * len(xs))
        if len(self.cs) != 0: cs = np.concatenate(self.cs, axis=0)
        else: cs = np.array([np.nan] * len(xs))
        if len(self.ss) != 0: ss = np.concatenate(self.ss, axis=0)
        else: ss = np.array([np.nan] * len(xs))

        df = pd.DataFrame({
            'sample_idx': sample_idx,
            'X': xs.tolist(),
            'Y': ys.tolist(),
            'T': ts.tolist(),
            'E': envs.tolist(),
            'C': cs.tolist(),
            'S': ss.tolist(),
            'Index_T': label_t,
            'SubTask': label_subT.tolist(),
            'Index_E': label_e,
            'Z_hat': z_hat.tolist(),
            'X_hat': x_hat.tolist(),
            'Y_hat': y_hat.tolist(),
        })
        return df
    
def inference_variables(lossfn_eval, frame, discr, gen, data_loader, device, phase='val', ta_data_loader=None, is_toy=False, n_mc=0,
                        sample_z_draw=False, gen_probs=True):
    total_loss, total_x_recon_loss, total_y_recon_loss = 0, 0, 0
    total_c_recon_loss, total_s_recon_loss = 0, 0
    collector = InferenceCollector()

    for i_bat, data_bat in enumerate(data_loader, start=1):
        if is_toy:
            xs = data_bat['X'].to(device, dtype=tc.float32)
            ys = data_bat['Y'].to(device, dtype=tc.float32)
            ts = data_bat['T'].to(device, dtype=tc.float32)
            cs = data_bat['C'].to(device, dtype=tc.float32)
            ss = data_bat['S'].to(device, dtype=tc.float32)
            envs = data_bat['E'].to(device, dtype=tc.float32)

            if len(ys.size()) == 1:
                ys = ys.unsqueeze(1)
            data_args = (xs, ts, envs, ys, phase)           
        else:
            xs = data_bat['X'].to(device, dtype=tc.float32)
            ys = data_bat['Y'].to(device, dtype=tc.float32)
            ts = data_bat['T'].to(device, dtype=tc.float32)
            envs = data_bat['E'].to(device, dtype=tc.float32)          

            if len(ys.size()) == 1:
                ys = ys.unsqueeze(1)
            data_args = (xs, ts, envs, ys, phase)

        total_loss += lossfn_eval(*data_args, 0)

        if n_mc == 0:
            with tc.no_grad():
                if sample_z_draw:
                    z_samples = frame.qt_z1x.draw((1,), {'x':xs, 't':ts, 'e':envs})
                    z_hat = z_samples['z'].squeeze(0)
                else:
                    z_hat = frame.qt_z1x.mean({'x':xs, 't':ts, 'e':envs})['z']

                if gen_probs:
                    x_hat = frame.p_x1z.mean({'z':z_hat})['x']
                    y_hat = frame.p_y1z.mean({'z':z_hat})['y']
                else:
                    x_hat = gen(z_hat)
                    y_hat = discr(xs)

        if phase == 'test':
            collector.collect_batch(data_bat, z_hat, x_hat, y_hat)

        total_x_recon_loss += F.mse_loss(x_hat, xs, reduction='none').mean(dim=1).sum()
        total_y_recon_loss += F.mse_loss(y_hat, ys, reduction='none').mean(dim=1).sum()

        if is_toy:
            total_c_recon_loss += F.mse_loss(z_hat, cs, reduction='none').mean(dim=1).sum()
            total_s_recon_loss += F.mse_loss(z_hat, ss, reduction='none').mean(dim=1).sum()

    avg_loss = total_loss / len(data_loader)
    avg_x_recon_loss = total_x_recon_loss / len(data_loader.dataset)
    avg_y_recon_loss = total_y_recon_loss / len(data_loader.dataset)

    if is_toy:
        avg_c_recon_loss = total_c_recon_loss.item() / len(data_loader.dataset)
        avg_s_recon_loss = total_s_recon_loss.item() / len(data_loader.dataset)
    else:
        avg_c_recon_loss = None
        avg_s_recon_loss = None

    results={
        'avg_loss': avg_loss.item(),
        'avg_x_recon_loss': avg_x_recon_loss.item(),
        'avg_y_recon_loss': avg_y_recon_loss.item(),
        'avg_c_recon_loss': avg_c_recon_loss,
        'avg_s_recon_loss': avg_s_recon_loss,
    }

    if phase == 'test':
        results_df = collector.to_dataframe()
        return results, results_df
    return results

def main(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader, val_src_loader, ls_ts_tgt_loader,
        is_toy=None,
    ):
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
    pgc = ParamGroupsCollector(ag.lr)
    pgc.collect_params(discr)
    pgc.collect_params(gen)

    opt = getattr(tc.optim, ag.optim)(pgc.param_groups, weight_decay=ag.wl2)
    auto_load(locals(), 'opt', ckpt)

    res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)
    print(f"Run in mode '{ag.mode}' for {ag.n_epk} epochs:")

    early_stopping = EarlyStopping(
        patience=ag.patience if hasattr(ag, 'patience') else 10,
        verbose=True
    )

    epk0 = 1
    n_per_epk = len(tr_src_loader)
    _filename = None
    for epk in range(epk0, ag.n_epk+1):
        n_min_batch, epk_loss = 0, 0
        pbar = tqdm.tqdm(total=n_per_epk, desc=f"Train epoch = {epk}", ncols=80, leave=False)
        for i_bat, data_bat in enumerate(tr_src_loader, start=1):
            n_min_batch = len(data_bat)
            xs = data_bat['X'].to(device, dtype=tc.float32)
            ys = data_bat['Y'].to(device, dtype=tc.float32)
            if len(ys.size()) == 1:
                ys = ys.unsqueeze(1)

            ts = data_bat['T'].to(device, dtype=tc.float32)
            envs = data_bat['E'].to(device, dtype=tc.float32)                
            data_args = (xs, ts, envs, ys, 'tr')

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
                setattr(ag, 'current_epoch', epk)
                results_tr = inference_variables(lossfn_eval, frame, discr, gen, tr_src_loader, device, phase='tr',
                                                  sample_z_draw=ag.sample_z_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
                
                results_val = inference_variables(lossfn_eval, frame, discr, gen, val_src_loader, device, phase='val',
                                                  sample_z_draw=ag.sample_z_draw, gen_probs=ag.gen_probs, is_toy=is_toy)              
                
                results_test, _ = inference_variables(lossfn_eval, frame, discr, gen, ls_ts_tgt_loader, device, phase='test',
                                                      sample_z_draw=ag.sample_z_draw, gen_probs=ag.gen_probs, is_toy=is_toy)

                print(f"Mode '{ag.mode}': Epoch {epk}, Val Loss = {results_val['avg_loss']:.3f}")
                print(f"Mode '{ag.mode}': Epoch {epk}, Test Loss = {results_test['avg_loss']:.3f}")

                wandb.log({
                    'epoch': epk,
                    'Tr_avg_Loss': epk_loss,
                    'Tr_X_Recon_Loss': results_tr['avg_x_recon_loss'],
                    'Tr_Y_Recon_Loss': results_tr['avg_y_recon_loss'],

                    'Val_avg_Loss': results_val['avg_loss'],
                    'Val_X_Recon_Loss': results_val['avg_x_recon_loss'],
                    'Val_Y_Recon_Loss': results_val['avg_y_recon_loss'],

                    'Te_avg_Loss': results_test['avg_loss'],
                    'Te_X_Recon_Loss': results_test['avg_x_recon_loss'],
                    'Te_Y_Recon_Loss': results_test['avg_y_recon_loss'],
                })

                if is_toy:
                    wandb.log({
                        'epoch': epk,
                        'Tr_C_Recon_Loss': results_tr['avg_c_recon_loss'],
                        'Tr_S_Recon_Loss': results_tr['avg_s_recon_loss'],
                        'Val_C_Recon_Loss': results_val['avg_c_recon_loss'],
                        'Val_S_Recon_Loss': results_val['avg_s_recon_loss'],
                        'Te_C_Recon_Loss': results_test['avg_c_recon_loss'],
                        'Te_S_Recon_Loss': results_test['avg_s_recon_loss'],
                    })
                
                if ag.early_stop_metric == 'sup': es_metric = results_val['val_Loss_sup']
                elif ag.early_stop_metric == 'y_emb': es_metric = results_val['avg_y_recon_loss']
                elif ag.early_stop_metric == 'val_loss': es_metric = results_val['avg_loss']

                flag_early_stop, flag_update_best_model = early_stopping(es_metric)
                if not ag.no_save:
                    _filename = save_ckpt(ag, ckpt, discr, gen, frame, opt,
                                              edic(locals()) | vars(ag),
                                              is_toy=is_toy, filename=_filename)
                if flag_early_stop:
                    print("Early stopping triggered")
                    return _filename, frame
                    
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
        ls_ts_tgt_loader = None,
        is_toy = None
    ):
    print(ag)
    print_infrstru_info()
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Models
    dc_vars = edic(locals()) | vars(ag) | \
        {'dim_x':ckpt['dim_x'], 'dim_y':ckpt['dim_y'],}

    res = get_models(archtype, dc_vars, ckpt, device)
    discr, gen, frame = res

    res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)

    with tc.no_grad():
        discr.eval(); gen.eval()

        lossfn_eval = ood_methods(discr, frame, ag)

        setattr(ag, 'log_loss', True)
        setattr(ag, 'current_epoch', 0)
        results_ID, results_ID_df = inference_variables(lossfn_eval, frame, discr, gen, tr_src_loader, device,
                                                        phase='test', sample_z_draw=ag.sample_z_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
        results_OOD, results_OOD_df = inference_variables(lossfn_eval, frame, discr, gen, ls_ts_tgt_loader, device,
                                                          phase='test', sample_z_draw=ag.sample_z_draw, gen_probs=ag.gen_probs, is_toy=is_toy)
        
        print(f"Mode '{ag.mode}': ID Loss = {results_ID['avg_loss']:.3f}")
        print(f"Mode '{ag.mode}': ID X Recon Loss = {results_ID['avg_x_recon_loss']:.3f}")
        print(f"Mode '{ag.mode}': ID Y Recon Loss = {results_ID['avg_y_recon_loss']:.3f}")

        print(f"Mode '{ag.mode}': OOD Loss = {results_OOD['avg_loss']:.3f}")
        print(f"Mode '{ag.mode}': OOD X Recon Loss = {results_OOD['avg_x_recon_loss']:.3f}")
        print(f"Mode '{ag.mode}': OOD Y Recon Loss = {results_OOD['avg_y_recon_loss']:.3f}")

        wandb.log({
            'ID_Loss': results_ID['avg_loss'],
            'OOD_Loss': results_OOD['avg_loss'],
            'ID_X_Recon_Loss': results_ID['avg_x_recon_loss'],
            'OOD_X_Recon_Loss': results_OOD['avg_x_recon_loss'],
            'ID_Y_Recon_Loss': results_ID['avg_y_recon_loss'],
            'OOD_Y_Recon_Loss': results_OOD['avg_y_recon_loss'],
        })

        if is_toy:
            wandb.log({
                'ID_C_Recon_Loss': results_ID['avg_c_recon_loss'],
                'OOD_C_Recon_Loss': results_OOD['avg_c_recon_loss'],
                'ID_S_Recon_Loss': results_ID['avg_s_recon_loss'],
                'OOD_S_Recon_Loss': results_OOD['avg_s_recon_loss'],
            })
    
    results_ID_df.to_pickle(f"./results/{ag.traindom}/results_{ckpt['filename'].split('/')[0]}_{ckpt['filename'].split('/')[-1].split(".")[0]}_ID.pkl")
    results_OOD_df.to_pickle(f"./results/{ag.traindom}/results_{ckpt['filename'].split('/')[0]}_{ckpt['filename'].split('/')[-1].split(".")[0]}_OOD.pkl")

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--gpu", type=int, default = 0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type = str, default='ind')
    parser.add_argument("--exp_name", type = str, choices=["real", "toy"])
    parser.add_argument("--emb_model", type=str, default='gpt')
    parser.add_argument("--debug", type=boolstr, default=False)
    parser.add_argument("--deploy", type=boolstr, default=False)
    parser.add_argument("--verbose", type=boolstr, default=False)
    parser.add_argument("--early_stop_metric", type=str, default='val_loss', choices=['val_loss', 'sup', 'y_emb'])

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="CCL_VAE")
    parser.add_argument("--log_loss", type=boolstr, default=False)

    # Data
    parser.add_argument("--traindom", type = str)
    parser.add_argument("--testdoms", type = str)
    parser.add_argument("--tr_val_split", type = float, default = .7)
    parser.add_argument("--n_bat", type = int, default = 128)
    parser.add_argument("--y_dtype", type = str, default = 'emb')
    parser.add_argument("--y_emb_option", type = str, default='default', choices=['default', 'long', 'eq'])
    parser.add_argument("--subtask", type = boolstr, default=False)

    # Model
    parser.add_argument("--init_model", type = str, default = "rand") # or a model file name to continue running
    parser.add_argument("--discrstru", type = str)
    parser.add_argument("--genstru", type = str)
    parser.add_argument("--mlpstrufile", type = str, default = "./arch/mlpstru.json")
    parser.add_argument("--use_fc", type=boolstr, default=False)

    # Process
    parser.add_argument("--n_epk", type = int, default = 800)
    parser.add_argument("--eval_interval", type = int, default = 5)
    parser.add_argument("--patience", type = int, default = 5)
    parser.add_argument("--no_save", type=boolstr, default=True)
    parser.add_argument("--elbo_only", type=boolstr, default=False)

    # Optimization
    parser.add_argument("--optim", type = str, default="Adam")
    parser.add_argument("--lr", type = float, default=1e-3)
    parser.add_argument("--wl2", type = float, default=1e-2)
    parser.add_argument("--reduction", type = str, default = "mean")

    # Supervised loss weight
    parser.add_argument("--wsup", type = float, default = 1.)
    parser.add_argument("--wsup_expo", type = float, default = 0.75) # only when "wsup" is not 0
    parser.add_argument("--wsup_wdatum", type = float, default = 6.25e-6) # only when "wsup" and "wsup_expo" are not 0
    
    # ELBO weight
    parser.add_argument("--wgen", type = float, default = 1.)
    parser.add_argument("--wlogpi", type = float, default = 1.)
    parser.add_argument("--wrecon_x", type = float, default = 1.)
    parser.add_argument("--wrecon_z", type = float, default = 1.)

    # For Prior
    parser.add_argument("--mu_z", type = float, default = 0.01)
    parser.add_argument("--sig_z", type = float, default = -1.)
    parser.add_argument("--actv_prior", type = str, default = "xtanh")
    parser.add_argument("--cond_prior", type = boolstr, default = False)

    # For Inference models
    parser.add_argument("--n_mc_q", type = int, default = 0)

    # For Generation models
    parser.add_argument("--pstd_x", type = float, default = 3e-1)
        
    parser.add_argument("--true_sup", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--true_sup_val", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--wda", type = float, default = .25)

    # Deploy settings
    parser.add_argument("--sample_z_draw", type=boolstr, default=False)
    parser.add_argument("--gen_probs", type=boolstr, default=True)
    parser.add_argument("--actv", type = str, default = "ReLU")
    parser.add_argument("--after_actv", type = boolstr, default = False)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    ag = parser.parse_args()
    ag, ckpt = process_continue_run(ag)
    archtype = "mlp"

    set_seed_all(ag.seed)

    # Dataset
    dataset_id, dataset_ood, y_dtype= load_dataset(ag, ag.exp_name, ag.traindom, ag.testdoms, ag.emb_model, ag.y_dtype, ag.y_emb_option)

    sample = dataset_id[0]
    shape_x = sample['X'].shape
    dim_t = sample['T'].shape[-1]
    dim_e = sample['E'].shape[-1]

    if y_dtype == 'emb':
        dim_y = sample['Y'].shape[-1]
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
        setattr(ag, 'discrstru', f'{ag.traindom}_{ag.emb_model}')
    print(ag.discrstru)

    # DataLoader
    if ag.deploy:
        tr_src_loader = DataLoader(dataset_id, batch_size=ag.n_bat, shuffle=False, drop_last=False)
        ts_tgt_loader = DataLoader(dataset_ood, batch_size=ag.n_bat, shuffle=False)
        setattr(ag, 'mode', 'ind')
    else:
        total_len = len(dataset_id)
        train_len = int(total_len * ag.tr_val_split)
        val_len = total_len - train_len

        train_dataset, val_dataset = random_split(dataset_id, [train_len, val_len])

        tr_src_loader = DataLoader(train_dataset, batch_size=ag.n_bat, shuffle=True, drop_last=False)
        val_src_loader = DataLoader(val_dataset, batch_size=ag.n_bat, shuffle=True)
        ts_tgt_loader = DataLoader(dataset_ood, batch_size=ag.n_bat, shuffle=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wandb.init(
                project=ag.wandb_project,
                config=ag,
                save_code=True,
                dir=tmp_dir
            )
        
        if ag.deploy:
            main_depoly(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, ts_tgt_loader, is_toy=ag.exp_name=='toy')
        else:
            filename, frame = main(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, val_src_loader, ts_tgt_loader, is_toy=ag.exp_name=='toy')      
            print(f"checkpoint saved to '{filename}'.")
            if ag.verbose:
                device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")
                er = EvalRetrival(device, dataset_id, dataset_ood, sample_cs_draw=ag.sample_cs_draw)
                er(frame)

        wandb.finish()

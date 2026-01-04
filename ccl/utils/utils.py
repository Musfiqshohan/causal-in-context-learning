#!/usr/bin/env python3.6
import os
import sys
import wandb
import warnings
import math
import random
import torch as tc
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from itertools import product, chain
from collections import defaultdict
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.utils.utils import pairwise_kl_div_two_batches, pairwise_wasserstein_distance_two_batches, pairwise_dot_two_batches, pairwise_manhattan_distance_two_batches
from ccl.utils.utils_data import HighDimSCMRealWorldDataset

def set_seed_all(seed):
    from transformers import set_seed
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed(seed)
        tc.cuda.manual_seed_all(seed)
        tc.backends.cudnn.deterministic = True
        tc.backends.cudnn.benchmark = False
        tc.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    tc.manual_seed(seed)
    try:
        tc.set_deterministic_debug_mode(True)
    except AttributeError:
        pass

# General Utilities
def unique_filename(prefix: str="", suffix: str="", n_digits: int=2, count_start: int=0) -> str:
    fmt = "{:0" + str(n_digits) + "d}"
    if prefix and prefix[-1] not in {"/", "\\"}: prefix += "_"
    
    if count_start > 100:
        return prefix + f"{count_start}" + suffix
    else:
        while True:
            filename = prefix + fmt.format(count_start) + suffix
            if not os.path.exists(filename): return filename
            else: count_start += 1

def print_infrstru_info():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPytc: {}".format(tc.__version__))
    print("\tCUDA: {}".format(tc.version.cuda))
    print("\tCUDNN: {}".format(tc.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

class Averager:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, nrep = 1):
        self._val = val
        self._sum += val.sum()
        self._count += nrep
        self._avg = self._sum / self._count

    @property
    def val(self): return self._val
    @property
    def avg(self): return self._avg
    @property
    def sum(self): return self._sum
    @property
    def count(self): return self._count

def repeat_iterator(itr, n_repeat):
    # The built-in `itertools.cycle` stores all results over `itr` and does not initialize `itr` again.
    return chain.from_iterable([itr] * n_repeat)

class RepeatIterator:
    def __init__(self, itr, n_repeat):
        self.itr = itr
        self.n_repeat = n_repeat
        self.len = len(itr) * n_repeat

    def __iter__(self):
        return chain.from_iterable([self.itr] * self.n_repeat)

    def __len__(self): return self.len

def zip_longer(itr1, itr2):
    len_ratio = len(itr1) / len(itr2)
    if len_ratio > 1:
        return zip(itr1, repeat_iterator(itr2, math.ceil(len_ratio)))
    elif len_ratio < 1:
        return zip(repeat_iterator(itr1, math.ceil(1/len_ratio)), itr2)
    else: return zip(itr1, itr2)

def zip_longest(*itrs):
    itr_longest = max(itrs, key=len)
    len_longest = len(itr_longest)
    return zip(*[itr if len(itr) == len_longest
            else repeat_iterator(itr, math.ceil(len_longest / len(itr)))
        for itr in itrs])

class ZipLongest:
    def __init__(self, *itrs):
        self.itrs = itrs
        self.itr_longest = max(itrs, key=len)
        self.len = len(self.itr_longest)

    def __iter__(self):
        return zip(*[itr if len(itr) == self.len
                else repeat_iterator(itr, math.ceil(self.len / len(itr)))
            for itr in self.itrs])

    def __len__(self): return self.len


class CyclicLoader:
    def __init__(self, datalist: list, shuffle: bool=True, cycle: int=None):
        self.len = len(datalist[0])
        if shuffle:
            ids = np.random.permutation(self.len)
            self.datalist = [data[ids] for data in datalist]
        else: self.datalist = datalist
        self.cycle = self.len if cycle is None else cycle
        self.head = 0
    def iter(self):
        self.head = 0
        return self
    def next(self, n: int) -> tuple:
        ids = [i % self.cycle for i in range(self.head, self.head + n)]
        self.head = (self.head + n) % self.cycle
        return tuple(data[ids] for data in self.datalist)
    def back(self, n: int):
        self.head = (self.head - n) % self.cycle
        return self

def boolstr(s: str) -> bool:
    # for argparse argument of type bool
    if isinstance(s, str):
        true_strings = {'1', 'true', 'True', 'T', 'yes', 'Yes', 'Y'}
        false_strings = {'0', 'false', 'False', 'F', 'no', 'No', 'N'}
        if s not in true_strings | false_strings:
            raise ValueError('Not a valid boolean string')
        return s in true_strings
    else:
        return bool(s)

## For lists/tuples
def getlist(ls: list, ids: list) -> list:
    return [ls[i] for i in ids]

def interleave(*lists) -> list:
    return [v for row in zip(*lists) for v in row]

def flatten(ls: list, depth: int=None) -> list:
    i = 0
    while (depth is None or i < depth) and bool(ls) and all(
            type(row) is list or type(row) is tuple for row in ls):
        ls = [v for row in ls if bool(row) for v in row]
        i += 1
    return ls

class SlicesAt:
    def __init__(self, axis: int, ndim: int):
        if ndim <= 0: raise ValueError(f"`ndim` (which is {ndim}) should be a positive integer")
        if not -ndim <= axis < ndim: raise ValueError(f"`axis` (which is {axis}) should be within [{-ndim}, {ndim})")
        self._axis, self._ndim = axis % ndim, ndim

    def __getitem__(self, idx):
        slices = [slice(None)] * self._ndim
        slices[self._axis] = idx
        return tuple(slices)

# For numpy/tc
def moving_average_slim(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # `(n_win-1)` shorter. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    cum = arr.cumsum(axis) # , dtype=float)
    return concatfn([ cum[slc[n_win-1:n_win]], cum[slc[n_win:]]-cum[slc[:-n_win]] ], axis) / float(n_win)

def moving_average_full(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # Same length as `arr`. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    cum = arr.cumsum(axis) # , dtype=float)
    stem = concatfn([ cum[slc[n_win-1:n_win]], cum[slc[n_win:]]-cum[slc[:-n_win]] ], axis) / float(n_win)
    length = arr.shape[axis]
    lwid = (n_win - 1) // 2
    rwid = n_win//2 + 1
    return concatfn([
            *[ cum[slc[j-1: j]] / float(j) for i in range(min(lwid, length)) for j in [min(i+rwid, length)] ],
            stem,
            *[ (cum[slc[-1:]] - cum[slc[i-lwid-1: i-lwid]] if i-lwid > 0 else cum[slc[-1:]]) / float(length-i+lwid)
                for i in range(max(length-rwid+1, lwid), length) ]
        ], axis)

def moving_average_full_checker(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # Same length as `arr`. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    if arr.shape[axis] < 2: return arr
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    lwid = (n_win - 1) // 2
    rwid = n_win//2 + 1
    return concatfn([ arr[slc[max(0, i-lwid): (i+rwid)]].mean(axis, keepdims=True) for i in range(arr.shape[axis]) ], axis)

# Plotting Utilities
class Plotter:
    def __init__(self, var_xlab: dict, metr_ylab: dict, tab_items: list=None,
            check_var: bool=False, check_tab: bool=False, loader = tc.load):
        for var in var_xlab:
            if not var_xlab[var]: var_xlab[var] = var
        for metr in metr_ylab:
            if not metr_ylab[metr]: metr_ylab[metr] = metr
        self.var_xlab, self.metr_ylab = var_xlab, metr_ylab
        self.variables, self.metrics = list(var_xlab), list(metr_ylab)
        if tab_items is None: self.tab_items = []
        else: self.tab_items = tab_items
        self.check_var, self.check_tab = check_var, check_tab
        self.loader = loader
        self._plt_data, self._tab_data = [], []

    def _get_res(self, dataholder): # does not change `self`
        res_x = {var: [] for var in self.variables}
        res_ymean = {metr: np.array([]) for metr in self.metrics}
        res_ystd = {metr: np.array([]) for metr in self.metrics}
        res_tab = [None for item in self.tab_items]
        if type(dataholder) is not dict: # treated as a list of data file names
            resfiles = []
            for file in dataholder:
                if os.path.isfile(file): resfiles.append(file)
                else: warnings.warn(f"file '{file}' does not exist")
            dataholder = dict()
            for file in resfiles:
                ckp = self.loader(file)
                for name in self.metrics + self.variables + self.tab_items:
                    if name not in ckp: warnings.warn(f"metric or variable or item '{name}' not found in file '{file}'")
                    else:
                        if name not in dataholder: dataholder[name] = []
                        dataholder[name].append(ckp[name])
        for metr in self.metrics:
            if metr not in dataholder or not dataholder[metr]:
                warnings.warn(f"metric '{metr}' not found or empty")
                continue
            n_align = min((len(line) for line in dataholder[metr]), default=0)
            if n_align:
                vals = np.array([line[:n_align] for line in dataholder[metr]])
                res_ymean[metr] = vals.mean(0)
                res_ystd[metr] = vals.std(0)
        for var in self.variables:
            if var not in dataholder or not dataholder[var]:
                warnings.warn(f"variable '{var}' not found or empty")
                continue
            n_align = min((len(line) for line in dataholder[var]), default=0)
            if n_align:
                res_x[var] = dataholder[var][0]
                if self.check_var:
                    for line in dataholder[var][1:]:
                        if line != res_x[x]: raise RuntimeError(f"variable '{var}' not match")
        for i, item in enumerate(self.tab_items):
            if item not in dataholder or not dataholder[item]:
                warnings.warn(f"item '{item}' not found or empty")
                continue
            res_tab[i] = dataholder[item][0]
            if self.check_tab:
                for val in dataholder[item][1:]:
                    if val != res_tab[i]: raise RuntimeError(f"item '{item}' not match")
        return res_x, res_ymean, res_ystd, res_tab

    def load(self, *triplets):
        # each triplet = (legend, pltsty, [filename1, filename2]), or
        # (legend, pltsty, {var1: [val1_1, val1_2], metr2: [val2_1, val2_2, val2_3]})
        data = [(legend, pltsty, *self._get_res(dataholder)) for legend, pltsty, dataholder in triplets]
        self._plt_data += [entry[:-1] for entry in data]
        self._tab_data += [[entry[0]] + entry[-1] for entry in data]

    def clear(self):
        self._plt_data.clear()
        self._tab_data.clear()

    def plot(self, variables: list=None, metrics: list=None,
            var_xlim: dict=None, metr_ylim: dict=None,
            n_start: int=None, n_stop: int=None, n_step: int=None, n_win: int=1,
            plot_err: bool=True, ncol: int=None,
            fontsize: int=20, figheight: int=8, linewidth: int=4, alpha: float=.2, show_legend: bool=True):
        if variables is None: variables = self.variables
        if metrics is None: metrics = self.metrics
        if var_xlim is None: var_xlim = {}
        if metr_ylim is None: metr_ylim = {}
        slc = slice(n_start, n_stop, n_step)
        if ncol is None: ncol = max(2, len(variables))
        nfig = len(variables) * len(metrics)
        nrow = (nfig-1) // ncol + 1
        if nfig < ncol: ncol = nfig
        plt.rcParams.update({'font.size': fontsize})

        fig, axes0 = plt.subplots(nrow, ncol, figsize=(ncol*figheight, nrow*figheight))
        if nfig == 1: axes = [axes0]
        elif nrow > 1: axes = [ax for row in axes0 for ax in row][:nfig]
        else: axes = axes0[:nfig]
        for ax, (metr, var) in zip(axes, product(metrics, variables)):
            plotted = False
            for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data:
                y, std = res_ymean[metr], res_ystd[metr]
                x = res_x[var] if var is not None else list(range(min(len(y), len(std))))
                n_align = min(len(x), len(y), len(std))
                x, y, std = x[:n_align], y[:n_align], std[:n_align]
                if n_win > 1:
                    y = moving_average_full(y, n_win)
                    if plot_err: std = moving_average_full(std, n_win) # Not precise! std and averaging is not interchangeable, since sqrt(sum ^2) is not linear
                x, y, std = x[slc], y[slc], std[slc]
                if len(x):
                    if plot_err:
                        ax.fill_between(x, y-std, y+std, facecolor=pltsty[0], alpha=alpha, linewidth=0)
                    ax.plot(x, y, pltsty, label=legend, linewidth=linewidth)
                    plotted = True
            if show_legend and plotted: ax.legend()
            if var in var_xlim: ax.set_xlim(var_xlim[var])
            if metr in metr_ylim: ax.set_ylim(metr_ylim[metr])
            ax.set_xlabel(self.var_xlab[var] if var is not None else "index")
            ax.set_ylabel(self.metr_ylab[metr])
        return fig, axes0

    def inspect(self, metr: str, ids: list=None, var: str=None, vals: list=None,
            show_std: bool=True, **tbformat):
        if (ids is None) == (var is None and vals is None):
            raise ValueError("exactly one of `ids`, or `var` and `vals`, should be provided")
        if ids is not None:
            if not show_std:
                table = [[legend, *getlist(res_ymean[metr], ids)] for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = ["indices"] + ids, **tbformat))
            else:
                table = [[legend, *interleave(getlist(res_ymean[metr], ids), getlist(res_ystd[metr], ids))]
                        for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = ["indices"] + interleave(ids, ids), **tbformat))
        else:
            if not show_std:
                table = [[legend, *[res_ymean[metr][res_x[var].index(val)] for val in vals]]
                        for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = [var] + vals, **tbformat))
            else:
                ids_list = [[res_x[var].index(val) for val in vals] for _, _, res_x, _, _ in self._plt_data]
                table = [[legend, *interleave(getlist(res_ymean[metr], ids), getlist(res_ystd[metr], ids))]
                        for ids, (legend, pltsty, res_x, res_ymean, res_ystd) in zip(ids_list, self._plt_data)]
                print(tabulate(table, headers = [var] + interleave(vals, vals), **tbformat))
        return table

    def tabulate(self, **tbformat):
        print(tabulate(self._tab_data, headers = ["legend"] + self.tab_items, **tbformat))

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve for a given patience."""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.update_best_model = False
        self.val_loss_min = float('inf')

    def dc_state_dict(self, dc_vars, *name_list):
        return {name+"_state_dict" : dc_vars[name].state_dict()
                for name in name_list if hasattr(dc_vars[name], 'state_dict')}

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.update_best_model = False
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'\n Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f})\n')
            self.best_loss = val_loss
            self.counter = 0
            self.update_best_model = True

        return self.early_stop, self.update_best_model
    
def compute_cossim(X, Y):
    n_samples = X.shape[0]
    return np.sum(cosine_similarity(X, Y)*np.eye(n_samples, n_samples))

class EvalRetrival():
    def __init__(self, device, dataset_id, sample_cs_draw=False, y_emb_option='eq',
                 data_dir='llm_ret', emb_model='gpt'):
        from ccl.run_ccl import custom_collate_fn
        
        self.c_sample_hat = []
        self.s_sample_hat = []
        self.device = device
        self.sample_cs_draw = sample_cs_draw

        self.target_batch_size = len(dataset_id)

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

        self.dataloader_id = DataLoader(dataset_id, batch_size=512, shuffle=False, 
                                  drop_last=False, pin_memory=False, prefetch_factor=4, persistent_workers=True, 
                                  num_workers=2, collate_fn=custom_collate_fn)
        
        if y_emb_option == 'long': df_ID = pd.read_feather(f"../data/{data_dir}/embs/dataset_{emb_model}_{y_emb_option}_emb_ID.feather")
        else: df_ID = pd.read_feather(f"../data/{data_dir}/embs/dataset_{emb_model}_emb_ID.feather")

        self.t_idx_id = df_ID['Index_T'].values
        self.e_idx_id = df_ID['Index_E'].values

        self.num_task = len(np.unique(self.t_idx_id))
        self.num_env = len(np.unique(self.e_idx_id))

        del df_ID

    def infer_latent_hat(self, dataloader, frame, is_ood=False):
        for _, data_bat in enumerate(dataloader, start=1):
            xs = data_bat.X.to(self.device, dtype=tc.float32)
            ts = data_bat.T.to(self.device, dtype=tc.float32)
            envs = data_bat.E.to(self.device, dtype=tc.float32)

            with tc.no_grad():
                c_hat = frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                s_hat = frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']
                            
            self.c_sample_hat.append(c_hat.cpu().numpy())
            self.s_sample_hat.append(s_hat.cpu().numpy())

    def __call__(self, frame, metric='kmeans'):
        print("Infer Latent Embedding")
        self.infer_latent_hat(self.dataloader_id, frame, is_ood=False)

        self.c_sample_hat = np.concatenate(self.c_sample_hat, axis=0)
        self.s_sample_hat = np.concatenate(self.s_sample_hat, axis=0)

        print("Latent C")
        kmeans = KMeans(n_clusters=self.num_task, random_state=0, n_init="auto").fit(self.c_sample_hat)
        centers = kmeans.cluster_centers_
        mat_ct = distance_matrix(centers, self.c_sample_hat)
        centers_idx = np.argmin(mat_ct, axis=1)
        ct_t_idx = np.sort(self.t_idx_id[centers_idx])
        ct_e_idx = np.sort(self.e_idx_id[centers_idx])

        print("Latent S")
        kmeans = KMeans(n_clusters=self.num_task, random_state=0, n_init="auto").fit(self.s_sample_hat)
        centers = kmeans.cluster_centers_
        mat_st = distance_matrix(centers, self.s_sample_hat)
        centers_idx = np.argmin(mat_st, axis=1)
        st_t_idx = np.sort(self.t_idx_id[centers_idx])
        st_e_idx = np.sort(self.e_idx_id[centers_idx])

        print("Clustering num of Env")
        print("Latent C")
        kmeans = KMeans(n_clusters=self.num_env, random_state=0, n_init="auto").fit(self.c_sample_hat)
        centers = kmeans.cluster_centers_
        mat_ce = distance_matrix(centers, self.c_sample_hat)
        centers_idx = np.argmin(mat_ce, axis=1)
        ce_t_idx = np.sort(self.t_idx_id[centers_idx])
        ce_e_idx = np.sort(self.e_idx_id[centers_idx])

        print("Latent S")
        kmeans = KMeans(n_clusters=self.num_env, random_state=0, n_init="auto").fit(self.s_sample_hat)
        centers = kmeans.cluster_centers_
        mat_se = distance_matrix(centers, self.s_sample_hat)
        centers_idx = np.argmin(mat_se, axis=1)
        se_t_idx = np.sort(self.t_idx_id[centers_idx])
        se_e_idx = np.sort(self.e_idx_id[centers_idx])

        print("Embedding C")
        print(f"Num of Cluster: Task: {self.num_task}")
        print(f"Center sample task: {np.unique(ct_t_idx)}")
        print(f"Center sample env: {np.unique(ct_e_idx)}")
        print(f"Num of Cluster: Env: {self.num_env}")
        print(f"Center sample task: {np.unique(ce_t_idx)}")
        print(f"Center sample env: {np.unique(ce_e_idx)}")

        print("Embedding S")
        print(f"Num of Cluster: Task: {self.num_task}")
        print(f"Center sample task: {np.unique(st_t_idx)}")
        print(f"Center sample env: {np.unique(st_e_idx)}")
        print(f"Num of Cluster: Env: {self.num_env}")
        print(f"Center sample task: {np.unique(se_t_idx)}")
        print(f"Center sample env: {np.unique(se_e_idx)}")

        wandb.log(
        {
            'EMB_C_NUM_T_Task(%)': round(100*(len(np.unique(ct_t_idx)) / self.num_task), 3),
            'EMB_S_NUM_T_Task(%)': round(100*(len(np.unique(st_t_idx)) / self.num_task), 3),
            'EMB_C_NUM_E_Env(%)': round(100*(len(np.unique(ct_e_idx)) / self.num_env), 3),
            'EMB_S_NUM_E_Env(%)': round(100*(len(np.unique(st_e_idx)) / self.num_env), 3),
            'EMB_C_NUM_E_Task(%)': round(100*(len(np.unique(ce_t_idx)) / self.num_task), 3),
            'EMB_S_NUM_E_Task(%)': round(100*(len(np.unique(se_t_idx)) / self.num_task), 3),
            'EMB_C_NUM_E_Env(%)': round(100*(len(np.unique(ce_e_idx)) / self.num_env), 3),
            'EMB_S_NUM_E_Env(%)': round(100*(len(np.unique(se_e_idx)) / self.num_env), 3),
        }
        )

    def get_nearest_neighbors(self, emb_demo, emb_target, std_demo, std_target, num_neighbors, metric='knn'):
        if not metric == 'knn':
            if type(emb_demo) == np.ndarray: emb_demo = tc.from_numpy(emb_demo)
            if type(emb_target) == np.ndarray: emb_target = tc.from_numpy(emb_target)
            if type(std_demo) == np.ndarray: std_demo = tc.from_numpy(std_demo)
            if type(std_target) == np.ndarray: std_target = tc.from_numpy(std_target)

        if metric == 'knn': # Euclidean distance as a default metric
            nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
            distances, indices = nbrs.kneighbors(emb_target)
        elif metric == 'kl_td': # KL(target || demo)
            distances, indices = pairwise_kl_div_two_batches(emb_target, std_target, emb_demo, std_demo, self.target_batch_size, self.demo_batch_size, self.device)
        elif metric == 'kl_dt': # KL(demo || target)
            distances, indices = pairwise_kl_div_two_batches(emb_demo, std_demo, emb_target, std_target, self.demo_batch_size, self.target_batch_size, self.device)
            distances = distances.detach().cpu().numpy().T; indices = indices.detach().cpu().numpy().T
            sorted_order = np.argsort(distances, axis=1)
            distances = np.take_along_axis(distances, sorted_order, axis=1)
            indices = np.take_along_axis(indices, sorted_order, axis=1)
        elif metric == 'wd': # Wasserstein distance
            distances, indices = pairwise_wasserstein_distance_two_batches(emb_target, std_target, emb_demo, std_demo, self.target_batch_size, self.demo_batch_size, self.device)
        elif metric == 'dot': # dot product
            distances, indices = pairwise_dot_two_batches(emb_target, emb_demo, self.target_batch_size, self.demo_batch_size, self.device)
        elif metric == 'man': # Manhattan distance
            distances, indices = pairwise_manhattan_distance_two_batches(emb_target, emb_demo, self.target_batch_size, self.demo_batch_size, self.device)
        return distances, indices

    def calculate_accuracy(self, nei_idx, t_idx_demo, t_idx_target, topk=10, is_sim=False):
        """Calculate accuracy for nearest neighbor predictions"""
        acc = 0.0
        for i in range(len(nei_idx)):
            if is_sim:
                acc += np.where(t_idx_demo[nei_idx[:, :topk][i]]==t_idx_demo[i], 1, 0).mean()
            else:
                acc += np.where(t_idx_demo[nei_idx[:, :topk][i]]==t_idx_target[i], 1, 0).mean()
        return acc / len(nei_idx)

    def calculate_ndcg(self, nei_idx, t_idx_demo, t_idx_target, topk=10):
        """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
        ndcg = 0.0
        for i in range(len(nei_idx)):
            # Get relevance scores (1 for match, 0 for no match)
            rel = np.where(t_idx_demo[nei_idx[:, :topk][i]]==t_idx_target[i], 1, 0)
            # Calculate DCG
            dcg = rel[0] + np.sum(rel[1:] / np.log2(np.arange(2, len(rel) + 1)))
            # Calculate IDCG (sort relevance in descending order)
            idcg = 1 + np.sum(np.sort(rel[1:])[::-1] / np.log2(np.arange(2, len(rel) + 1)))
            ndcg += dcg / idcg if idcg > 0 else 0
        return ndcg / len(nei_idx)

    def calculate_f1(self, nei_idx, t_idx_demo, t_idx_target):
        """Calculate top-1 F1 score"""
        tp = fp = fn = 0
        for i in range(len(nei_idx)):
            pred = t_idx_demo[nei_idx[i]][0]  # Top-1 prediction
            true = t_idx_target[i]
            if pred == true:
                tp += 1
            else:
                fp += 1
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def eval_retrival_performance(self, nei_idx, example_labels, target_labels,topk=3):
        acc = self.calculate_accuracy(nei_idx, example_labels, target_labels, topk=topk)
        ndcg = self.calculate_ndcg(nei_idx, example_labels, target_labels, topk=topk)
        f1 = self.calculate_f1(nei_idx, example_labels, target_labels)

        print("Acc: ", round(acc, 3))
        print("NDCG: ", round(ndcg, 3))
        print("F1: ", round(f1, 3))
        return acc, ndcg, f1

def print_results(ag, result_dict):
    w_dict = {
        'log_phi_loss': ag.wlogpi,
        'Elbo': ag.wgen,
        'Loss_x': ag.wrecon_x,
        'Loss_e': ag.wrecon_e,
        'Loss_c': ag.wrecon_c,
        'Loss_s': ag.wrecon_s,
        'Reg_CS_loss': ag.wreg_cs,
        'Loss_sup': ag.wsup
    }
    state = list(result_dict.keys())[-1].split('_')[0]
    print(f"Epoch: {result_dict['Epoch']}")
    if state in ['tr', 'val']:
        print(f"State: {state}")
        
        log_list = list(result_dict.keys())[1:-1]
        for k in log_list:
            if k.split('_')[0] == 'val':
                _w = w_dict['_'.join(k.split('_')[1:])]
                print(f"{k}: {_w}*{round(result_dict[k], 6)} = ", round(_w*result_dict[k], 6))
        print(f"Total Loss: {result_dict['val_total_Loss']}")
        print("**************")

class RetrievalEval():
    def __init__(self, ag):
        if ag.exp_name == 'ood_nlp':
            LABEL2IDX={
            'sa':{'negative':0, 'positive':1, 'neutral':2},
            'nli':{'entailment':0, 'neutral':1, 'contradiction':2},
            'td':{'benign':0, 'toxic':1}}
            TASK2IDX={
            'sa':0,
            'nli':1,
            'eqa':2,
            'td':3,
            'ner':4}
        elif ag.exp_name == 'llm_ret':
            LABEL2IDX = {
                'sentiment':{'negative':0, 'positive':1},
                'commonsense':{'a':0, 'b':1},
                'coreference':{'a':0, 'b':1},
                'nli':{'yes':0, 'no':1}}
            TASK2IDX = {'sentiment':6,
                        'commonsense':1,
                        'coreference':2,
                        'nli':4}

        self.dataset_id = pd.read_feather(f"../data/{ag.exp_name}/embs/dataset_{ag.emb_model}_emb_ID.feather")
        self.dataset_ood = pd.read_feather(f"../data/{ag.exp_name}/embs/{ag.testdoms}/dataset_{ag.emb_model}_emb_OOD.feather")
        
        self.label2idx = LABEL2IDX[ag.traindom]
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.dataset_ood['Index_Y'] = self.dataset_ood['answer'].apply(lambda x: self.label2idx[x])
        self.dataset_ood = self.dataset_ood.groupby('Index_Y', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 1000)))

        print(f"Loading inference results for {ag.traindom} from {ag.ckp_id}...")
        self.results_ID = pd.read_feather(f"./results/{ag.exp_name}/results_ind_{ag.emb_model}_emb_{ag.ckp_id}_ID.feather")
        self.results_ID = self.results_ID[self.results_ID['Index_T']==TASK2IDX[ag.traindom]]
        self.results_ID = self.results_ID.sort_values(by='sample_idx', ignore_index=True)
        self.results_ID['C_norm'] = self.results_ID['C_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x))) 
        self.results_ID['S_norm'] = self.results_ID['S_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))

    def get_nearest_neighbors(self, emb_demo, emb_target, std_demo, std_target, num_neighbors, metric='knn'):
        if metric == 'knn':
            nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
            distances, indices = nbrs.kneighbors(emb_target)
        elif metric == 'cossim':
            dist_matrix = pairwise.cosine_similarity(X=emb_target, Y=emb_demo)
            distances, indices = tc.topk(tc.from_numpy(dist_matrix), k=num_neighbors, dim=-1)
            distances = distances.numpy()
            indices = indices.numpy()
        return distances, indices

    def calc_ndcg(self, exp_answer_list, gt):
        # Convert labels to binary values (1 for matching gt, 0 otherwise)
        rel_scores = [1 if ans == gt else 0 for ans in exp_answer_list]

        # Calculate DCG
        dcg = rel_scores[0] + sum(rel/np.log2(i+2) for i, rel in enumerate(rel_scores[1:]))

        # Calculate IDCG (sort relevance scores in descending order)
        idcg = 1 + sum(1/np.log2(i+2) for i in range(sum(rel_scores[1:])))

        # Calculate NDCG
        ndcg = dcg/idcg if idcg > 0 else 0
        return ndcg

    def eval_retrieval(self, results_OOD, embs, metric):
        x_id = np.stack(self.results_ID.X.values)
        c_id_hat = np.stack(self.results_ID['C_hat'].values)
        c_id_norm = np.stack(self.results_ID['C_norm'].values)
        c_id_std = np.stack(self.results_ID['C_std'].values)

        c_ood_hat = np.stack(results_OOD['C_hat'].values)
        c_ood_norm = np.stack(results_OOD['C_norm'].values)
        c_ood_std = np.stack(results_OOD['C_std'].values)

        if embs == 'c_hat': dist_ood, nei_idx_ood = self.get_nearest_neighbors(emb_demo=c_id_hat, emb_target=c_ood_hat, num_neighbors=len(x_id), metric=metric)
        elif embs == 'c_norm': dist_ood, nei_idx_ood = self.get_nearest_neighbors(emb_demo=c_id_norm, emb_target=c_ood_norm, num_neighbors=len(x_id), metric=metric)

        results = defaultdict(int)
        results_class = defaultdict(list)
        for i in range(len(results_OOD)):
            example_idx = nei_idx_ood[i]
            row_ood = self.dataset_ood.iloc[i]
            topk_exaple_idx = example_idx[:3]
            
            assert row_ood.name == results_OOD.iloc[i].sample_idx

            target_answer = row_ood['answer']
            target_label = self.label2idx[target_answer]

            counter_topk = 0
            counter_top1 = 0
            counter_error = 0
            exp_answer_list = []
            for k, i in enumerate(topk_exaple_idx):
                sample_idx = self.results_ID.iloc[i].sample_idx
                example_answer = self.dataset_id[self.dataset_id.index==str(sample_idx)]['answer'].values[0]
                example_label = self.label2idx[example_answer]
                exp_answer_list.append(example_label)

                if k == 0 and example_label == target_label:
                    counter_top1 += 1
                    counter_topk += 1
                elif example_label == target_label:
                    counter_topk += 1
                
                if example_label != target_label:
                    counter_error += 1
            
            counter_topk /= 3
            if counter_error == 3: counter_error = 1
            else: counter_error = 0

            ndcg = self.calc_ndcg(exp_answer_list, target_label)

            results['retrieval_topk'] += counter_topk
            results['retrieval_top1'] += counter_top1
            results['retrieval_error'] += counter_error
            results['retrieval_ndcg'] += ndcg

            results_class[f'{self.idx2label[target_label]}_retrieval_top1'].append(counter_top1)
            results_class[f'{self.idx2label[target_label]}_retrieval_topk'].append(counter_topk)
            results_class[f'{self.idx2label[target_label]}_retrieval_error'].append(counter_error)
            results_class[f'{self.idx2label[target_label]}_retrieval_ndcg'].append(ndcg)
        
        results['retrieval_topk'] /= len(results_OOD)
        results['retrieval_top1'] /= len(results_OOD)
        results['retrieval_error'] /= len(results_OOD)
        results['retrieval_ndcg'] /= len(results_OOD)

        for k, v in results_class.items():
            results_class[k] = np.mean(v)

        print("\n=== Overall Retrieval Results ===")
        print(f"Top-1 Accuracy: {100*(results['retrieval_top1']):.4f}")
        print(f"Top-k Accuracy: {100*(results['retrieval_topk']):.4f}")
        print(f"Error Rate: {100*(results['retrieval_error']):.4f}")
        print(f"NDCG: {100*(results['retrieval_ndcg']):.4f}")

        print("\n=== Per-Class Retrieval Results ===")
        for label, score in results_class.items():
            print(f"{label}: {100*(score):.4f}")

        return results, results_class
    
    def set_results_ood(self, results_OOD):
        results_OOD = results_OOD.sort_values(by='sample_idx').reset_index(drop=True)
        results_OOD['C_norm'] = results_OOD['C_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))
        results_OOD['S_norm'] = results_OOD['S_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))
        results_OOD = results_OOD[results_OOD['sample_idx'].isin(self.dataset_ood.index)]
        self.dataset_ood = self.dataset_ood.sort_index()
        self.results_OOD = results_OOD.sort_values(by='sample_idx')

    def __call__(self, embs='c_hat', metric='knn'):
        results, results_class = self.eval_retrieval(self.results_OOD, embs, metric)

        results.update(results_class)
        return results
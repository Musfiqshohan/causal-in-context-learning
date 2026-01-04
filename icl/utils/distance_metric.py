import time
import pandas as pd
import numpy as np

import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence

from tqdm import tqdm

def pairwise_kl_normal(target_dist, demo_dist, eps=1e-8):
    target_loc   = target_dist.loc
    target_scale = target_dist.scale # .clamp(min=eps)
    
    demo_loc   = demo_dist.loc
    demo_scale = demo_dist.scale # .clamp(min=eps)
    
    target_loc   = target_loc.unsqueeze(1)  
    target_scale = target_scale.unsqueeze(1)
    demo_loc     = demo_loc.unsqueeze(0)    
    demo_scale   = demo_scale.unsqueeze(0)
    
    kl_per_dim = torch.log(demo_scale / target_scale) + \
                 (target_scale.pow(2) + (target_loc - demo_loc).pow(2)) / (2 * demo_scale.pow(2)) - 0.5
    
    kl_avg = kl_per_dim.mean(dim=-1)  # shape: (N, M)
    return kl_avg

def pairwise_kl_div_two_batches(mean_target, std_target, mean_demo, std_demo,
                                target_batch_size=1000, demo_batch_size=1000, device='cpu'):
    N = mean_target.size(0)
    M = mean_demo.size(0)

    sorted_kls_list = []
    sorted_idx_list = []

    for t_start in tqdm(range(0, N, target_batch_size)):
        t_end = min(t_start + target_batch_size, N)

        batch_mean_target = mean_target[t_start:t_end].to(device)
        batch_std_target = std_target[t_start:t_end].to(device)
        
        batch_kl_parts = []
        batch_idx_parts = []
        
        for d_start in tqdm(range(0, M, demo_batch_size)):
            d_end = min(d_start + demo_batch_size, M)

            batch_mean_demo = mean_demo[d_start:d_end].to(device)
            batch_std_demo = std_demo[d_start:d_end].to(device)
            
            target_dist = Normal(batch_mean_target, batch_std_target)
            demo_dist = Normal(batch_mean_demo, batch_std_demo)

            kl = pairwise_kl_normal(target_dist, demo_dist)
            
            global_demo_indices = torch.arange(d_start, d_end, device=kl.device)
            global_demo_indices = global_demo_indices.unsqueeze(0).expand(kl.size(0), -1)
            
            batch_kl_parts.append(kl.cpu())
            batch_idx_parts.append(global_demo_indices.cpu())
            
            del batch_mean_demo, batch_std_demo, kl, global_demo_indices
        
        batch_kl = torch.cat(batch_kl_parts, dim=1)
        batch_idx = torch.cat(batch_idx_parts, dim=1)
        
        sorted_kl, sorted_idx_within = torch.sort(batch_kl, dim=1, descending=False)
        sorted_demo_indices = torch.gather(batch_idx, 1, sorted_idx_within)
        
        sorted_kls_list.append(sorted_kl)
        sorted_idx_list.append(sorted_demo_indices)
        
        del batch_mean_target, batch_std_target, batch_kl, batch_idx, sorted_kl, sorted_idx_within, sorted_demo_indices

    sorted_kls_full = torch.cat(sorted_kls_list, dim=0)
    sorted_demo_indices_full = torch.cat(sorted_idx_list, dim=0)
    return sorted_kls_full, sorted_demo_indices_full

def pairwise_wasserstein_distance_two_batches(mean_target, std_target, mean_demo, std_demo,
                                             target_batch_size=1000, demo_batch_size=1000, device='cpu'):
    N = mean_target.size(0)
    M = mean_demo.size(0)

    sorted_wd_list = []
    sorted_idx_list = []

    for t_start in tqdm(range(0, N, target_batch_size), desc="Target batches"):
        t_end = min(t_start + target_batch_size, N)
        batch_mean_target = mean_target[t_start:t_end].to(device)
        batch_std_target = std_target[t_start:t_end].to(device)
        
        batch_wd_parts = []
        batch_idx_parts = []
        
        for d_start in tqdm(range(0, M, demo_batch_size), desc="Demo batches", leave=False):
            d_end = min(d_start + demo_batch_size, M)
            batch_mean_demo = mean_demo[d_start:d_end].to(device)
            batch_std_demo = std_demo[d_start:d_end].to(device)
            
            diff_mean = batch_mean_target.unsqueeze(1) - batch_mean_demo.unsqueeze(0)
            diff_std  = batch_std_target.unsqueeze(1) - batch_std_demo.unsqueeze(0)

            wd_squared = (diff_mean ** 2).sum(dim=-1) + (diff_std ** 2).sum(dim=-1)
            wd = torch.sqrt(wd_squared)
            
            global_demo_indices = torch.arange(d_start, d_end, device=wd.device)
            global_demo_indices = global_demo_indices.unsqueeze(0).expand(wd.size(0), -1)
            
            batch_wd_parts.append(wd.cpu())
            batch_idx_parts.append(global_demo_indices.cpu())
        
        batch_wd = torch.cat(batch_wd_parts, dim=1)
        batch_idx = torch.cat(batch_idx_parts, dim=1)
        
        sorted_wd, sorted_idx_within = torch.sort(batch_wd, dim=1, descending=False)
        sorted_demo_indices = torch.gather(batch_idx, 1, sorted_idx_within)
        
        sorted_wd_list.append(sorted_wd)
        sorted_idx_list.append(sorted_demo_indices)

    sorted_wd_full = torch.cat(sorted_wd_list, dim=0)
    sorted_demo_indices_full = torch.cat(sorted_idx_list, dim=0)
    return sorted_wd_full, sorted_demo_indices_full

def pairwise_dot_two_batches(emb_target, emb_demo,
                             target_batch_size=1000, demo_batch_size=1000, device='cpu'):
    N = emb_target.size(0)
    M = emb_demo.size(0)

    sorted_dot_list = []
    sorted_idx_list = []

    for t_start in tqdm(range(0, N, target_batch_size), desc="Target batches"):
        t_end = min(t_start + target_batch_size, N)
        batch_emb_target = emb_target[t_start:t_end].to(device)
        
        batch_dot_parts = []
        batch_idx_parts = []
        
        for d_start in tqdm(range(0, M, demo_batch_size), desc="Demo batches", leave=False):
            d_end = min(d_start + demo_batch_size, M)
            batch_emb_demo = emb_demo[d_start:d_end].to(device)

            dot = torch.matmul(batch_emb_target, batch_emb_demo.t())

            global_demo_indices = torch.arange(d_start, d_end, device=device)
            global_demo_indices = global_demo_indices.unsqueeze(0).expand(dot.size(0), -1)
            
            batch_dot_parts.append(dot.cpu())
            batch_idx_parts.append(global_demo_indices.cpu())

        batch_dot = torch.cat(batch_dot_parts, dim=1)
        batch_idx = torch.cat(batch_idx_parts, dim=1)
        
        sorted_dot, sorted_idx_within = torch.sort(batch_dot, dim=1, descending=True)
        sorted_demo_indices = torch.gather(batch_idx, 1, sorted_idx_within)
        
        sorted_dot_list.append(sorted_dot)
        sorted_idx_list.append(sorted_demo_indices)

    sorted_dot_full = torch.cat(sorted_dot_list, dim=0)
    sorted_demo_indices_full = torch.cat(sorted_idx_list, dim=0)
    return sorted_dot_full, sorted_demo_indices_full

def pairwise_euclidean_distance_two_batches(emb_target, emb_demo,
                                            target_batch_size=1000, demo_batch_size=1000, device='cpu'):
    N = emb_target.size(0)
    M = emb_demo.size(0)

    sorted_euclidean_distance_list = []
    sorted_idx_list = []

    for t_start in tqdm(range(0, N, target_batch_size), desc="Target batches"):
        t_end = min(t_start + target_batch_size, N)
        batch_emb_target = emb_target[t_start:t_end].to(device)
        
        batch_euclidean_distance_parts = []
        batch_idx_parts = []
        
        for d_start in tqdm(range(0, M, demo_batch_size), desc="Demo batches", leave=False):
            d_end = min(d_start + demo_batch_size, M)
            batch_emb_demo = emb_demo[d_start:d_end].to(device)

            euclidean_distance = torch.norm(batch_emb_target - batch_emb_demo, dim=-1)

            global_demo_indices = torch.arange(d_start, d_end, device=device)
            global_demo_indices = global_demo_indices.unsqueeze(0).expand(euclidean_distance.size(0), -1)
            
            batch_euclidean_distance_parts.append(euclidean_distance.cpu())
            batch_idx_parts.append(global_demo_indices.cpu())

        batch_euclidean_distance = torch.cat(batch_euclidean_distance_parts, dim=1)
        batch_idx = torch.cat(batch_idx_parts, dim=1)
        
        sorted_euclidean_distance, sorted_idx_within = torch.sort(batch_euclidean_distance, dim=1, descending=True)
        sorted_demo_indices = torch.gather(batch_idx, 1, sorted_idx_within)
        
        sorted_euclidean_distance_list.append(sorted_euclidean_distance)
        sorted_idx_list.append(sorted_demo_indices)

    sorted_euclidean_distance_full = torch.cat(sorted_euclidean_distance_list, dim=0)
    sorted_demo_indices_full = torch.cat(sorted_idx_list, dim=0)
    return sorted_euclidean_distance_full, sorted_demo_indices_full

def pairwise_manhattan_distance_two_batches(emb_target, emb_demo,
                                            target_batch_size=1000, demo_batch_size=1000, device='cpu'):
    N = emb_target.size(0)
    M = emb_demo.size(0)

    sorted_manhattan_distance_list = []
    sorted_idx_list = []

    for t_start in tqdm(range(0, N, target_batch_size), desc="Target batches"):
        t_end = min(t_start + target_batch_size, N)
        batch_emb_target = emb_target[t_start:t_end].to(device)
        
        batch_manhattan_distance_parts = []
        batch_idx_parts = []
        
        for d_start in tqdm(range(0, M, demo_batch_size), desc="Demo batches", leave=False):
            d_end = min(d_start + demo_batch_size, M)
            batch_emb_demo = emb_demo[d_start:d_end].to(device)

            manhattan_distance = torch.norm(batch_emb_target - batch_emb_demo, dim=-1, p=1)

            global_demo_indices = torch.arange(d_start, d_end, device=device)
            global_demo_indices = global_demo_indices.unsqueeze(0).expand(manhattan_distance.size(0), -1)
            
            batch_manhattan_distance_parts.append(manhattan_distance.cpu())
            batch_idx_parts.append(global_demo_indices.cpu())

        batch_manhattan_distance = torch.cat(batch_manhattan_distance_parts, dim=1)
        batch_idx = torch.cat(batch_idx_parts, dim=1)
        
        sorted_manhattan_distance, sorted_idx_within = torch.sort(batch_manhattan_distance, dim=1, descending=True)
        sorted_demo_indices = torch.gather(batch_idx, 1, sorted_idx_within)
        
        sorted_manhattan_distance_list.append(sorted_manhattan_distance)
        sorted_idx_list.append(sorted_demo_indices)

    sorted_manhattan_distance_full = torch.cat(sorted_manhattan_distance_list, dim=0)
    sorted_demo_indices_full = torch.cat(sorted_idx_list, dim=0)
    return sorted_manhattan_distance_full, sorted_demo_indices_full
import re
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

from torch.distributions import Normal

from matplotlib.colors import Normalize

from scipy import stats
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccl.utils.utils import set_seed_all
from icl.utils.utils import pairwise_kl_div_two_batches, pairwise_wasserstein_distance_two_batches, pairwise_dot_two_batches, pairwise_manhattan_distance_two_batches

def get_nearest_neighbors(emb_demo, emb_target, std_demo, std_target, num_neighbors, metric='knn', device='cuda'):
    if not metric == 'knn':
        if type(emb_demo) == np.ndarray: emb_demo = torch.from_numpy(emb_demo)
        if type(emb_target) == np.ndarray: emb_target = torch.from_numpy(emb_target)
        if type(std_demo) == np.ndarray: std_demo = torch.from_numpy(std_demo)
        if type(std_target) == np.ndarray: std_target = torch.from_numpy(std_target)

    if metric == 'knn': # Euclidean distance as a default metric
        nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
        distances, indices = nbrs.kneighbors(emb_target)
    elif metric == 'kl_td': # KL(target || demo)
        distances, indices = pairwise_kl_div_two_batches(emb_target, std_target, emb_demo, std_demo, 512, 1000, device)
    elif metric == 'kl_dt': # KL(demo || target)
        distances, indices = pairwise_kl_div_two_batches(emb_demo, std_demo, emb_target, std_target, 1000, 512, device)
        distances = distances.detach().cpu().numpy().T; indices = indices.detach().cpu().numpy().T
        sorted_order = np.argsort(distances, axis=1)
        distances = np.take_along_axis(distances, sorted_order, axis=1)
        indices = np.take_along_axis(indices, sorted_order, axis=1)
    elif metric == 'wd': # Wasserstein distance
        distances, indices = pairwise_wasserstein_distance_two_batches(emb_target, std_target, emb_demo, std_demo, 512, 1000, device)
    elif metric == 'dot': # dot product
        distances, indices = pairwise_dot_two_batches(emb_target, emb_demo, 512, 1000, device)
    elif metric == 'man': # Manhattan distance
        distances, indices = pairwise_manhattan_distance_two_batches(emb_target, emb_demo, 512, 1000, device)
    return distances, indices

def calc_ndcg(exp_answer_list, gt):
    # Convert labels to binary values (1 for matching gt, 0 otherwise)
    rel_scores = [1 if ans == gt else 0 for ans in exp_answer_list]

    # Calculate DCG
    dcg = rel_scores[0] + sum(rel/np.log2(i+2) for i, rel in enumerate(rel_scores[1:]))

    # Calculate IDCG (sort relevance scores in descending order)
    idcg = 1 + sum(1/np.log2(i+2) for i in range(sum(rel_scores[1:])))

    # Calculate NDCG
    ndcg = dcg/idcg if idcg > 0 else 0
    return ndcg

def eval_retrieval(args, data_id, data_ood, results_ID, results_OOD, label2idx, idx2label):
    log_file = f"./results/{args.exp_name}/retrieval/retrieval_{args.metric[:3]}_{args.ckp}_adapt_{args.adapt}_{args.embs}_{args.testdomain}.txt"
    print(f"Results will be saved to {log_file}")

    x_id = np.stack(results_ID.X.values)
    c_id_hat = np.stack(results_ID['C_hat'].values)
    c_id_norm = np.stack(results_ID['C_norm'].values)
    
    x_ood = np.stack(data_ood.X.values)
    c_ood_hat = np.stack(results_OOD['C_hat'].values)
    c_ood_norm = np.stack(results_OOD['C_norm'].values)

    if args.metric in ['kl_td', 'kl_dt', 'wd']:
        std_id = np.stack(results_ID['C_std'].values)
        std_ood = np.stack(results_OOD['C_std'].values)
    else:
        std_id, std_ood = None, None

    if args.embs == 'x': dist_ood, nei_idx_ood = get_nearest_neighbors(emb_demo=x_id, emb_target=x_ood, std_demo=std_id, std_target=std_ood, num_neighbors=len(x_id), metric=args.metric, device=args.device)
    elif args.embs == 'c_hat': dist_ood, nei_idx_ood = get_nearest_neighbors(emb_demo=c_id_hat, emb_target=c_ood_hat, std_demo=std_id, std_target=std_ood, num_neighbors=len(x_id), metric=args.metric, device=args.device)
    elif args.embs == 'c_norm': dist_ood, nei_idx_ood = get_nearest_neighbors(emb_demo=c_id_norm, emb_target=c_ood_norm, std_demo=std_id, std_target=std_ood, num_neighbors=len(x_id), metric=args.metric, device=args.device)

    if isinstance(nei_idx_ood, torch.Tensor):
        nei_idx_ood = nei_idx_ood.cpu().numpy().tolist()

    results = defaultdict(int)
    results_class = defaultdict(list)
    for i in range(len(results_OOD)):
        example_idx = nei_idx_ood[i]
        row_ood = data_ood.iloc[i]
        topk_exaple_idx = example_idx[:3]
        
        assert row_ood.name == results_OOD.iloc[i].sample_idx

        target_task = row_ood['Index_T']
        # target_label = label2idx[target_task]

        counter_topk = 0
        counter_top1 = 0
        counter_error = 0
        exp_answer_list = []
        for k, i in enumerate(topk_exaple_idx):
            sample_idx = results_ID.iloc[i].sample_idx
            example_answer = data_id[data_id.index==sample_idx]['Index_T'].values[0]
            # try: example_label = label2idx[example_answer]
            # except: example_label = label2idx['etc']
            exp_answer_list.append(example_answer)

            if k == 0 and example_answer == target_task:
                counter_top1 += 1
                counter_topk += 1
            elif example_answer == target_task:
                counter_topk += 1
            
            if example_answer != target_task:
                counter_error += 1
        
        counter_topk /= 3
        if counter_error == 3: counter_error = 1
        else: counter_error = 0

        ndcg = calc_ndcg(exp_answer_list, target_task)

        results['retrieval_topk'] += counter_topk
        results['retrieval_top1'] += counter_top1
        results['retrieval_error'] += counter_error
        results['retrieval_ndcg'] += ndcg

        results_class[f'{target_task}_retrieval_top1'].append(counter_top1)
        results_class[f'{target_task}_retrieval_topk'].append(counter_topk)
        results_class[f'{target_task}_retrieval_error'].append(counter_error)
        results_class[f'{target_task}_retrieval_ndcg'].append(ndcg)
        # results_class[f'{idx2label[target_task]}_retrieval_top1'].append(counter_top1)
        # results_class[f'{idx2label[target_task]}_retrieval_topk'].append(counter_topk)
        # results_class[f'{idx2label[target_task]}_retrieval_error'].append(counter_error)
        # results_class[f'{idx2label[target_task]}_retrieval_ndcg'].append(ndcg)
    
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

    # Save results to log file
    with open(log_file, "w") as f:
        f.write("=== Overall Retrieval Results ===\n")
        f.write(f"Top-1 Accuracy: {100*(results['retrieval_top1']):.4f}\n")
        f.write(f"Top-k Accuracy: {100*(results['retrieval_topk']):.4f}\n") 
        f.write(f"Error Rate: {100*(results['retrieval_error']):.4f}\n\n")
        f.write(f"NDCG: {100*(results['retrieval_ndcg']):.4f}\n\n")
        
        f.write("=== Per-Class Retrieval Results ===\n")
        for label, score in results_class.items():
            f.write(f"{label}: {100*(score):.4f}\n")
    print(f"\nResults saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ood_nlp')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='sa')
    parser.add_argument('--emb_model', type=str, default='gpt')
    parser.add_argument('--embs', type=str, default='c_hat')
    parser.add_argument('--ckp', type=str, default='101')
    parser.add_argument('--testdomain', type=str, default='amazon')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--adapt_id', type=str, default='random')
    parser.add_argument('--metric', type=str, default='knn', help='knn or cossim')
    parser.add_argument('--remove_results', action='store_true')
    args = parser.parse_args()

    setattr(args, 'device', f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if args.exp_name == 'ood_nlp':
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
    elif args.exp_name == 'llm_ret':
        LABEL2IDX = {
            'sentiment':{'negative':0, 'positive':1},
            'commonsense':{'a':0, 'b':1, 'etc':2},
            'coreference':{'a':0, 'b':1, 'etc':2},
            'nli':{'yes':0, 'no':1, 'etc':2}}
        TASK2IDX = {'sentiment':6,
                    'commonsense':1,
                    'coreference':2,
                    'nli':4}

    set_seed_all(0)
    
    if args.remove_results:
        if args.adapt: path_ood = f"./results/{args.exp_name}/{args.task}/{args.testdomain}/{args.adapt_id}/results_ind_{args.emb_model}_emb_{args.task}_{args.testdomain}_{args.ckp}_ada_OOD.feather"
        else: path_ood = f"./results/{args.exp_name}/{args.task}/{args.testdomain}/results_ind_{args.emb_model}_emb_{args.task}_{args.testdomain}_{args.ckp}_OOD.feather"
        os.remove(path_ood)
        print("Removing results")
        exit()

    print(f"Loading dataset")
    data_id = pd.read_feather(f"../data/{args.exp_name}/embs/dataset_gpt_emb_ID.feather")
    if args.testdomain in ['amazon', 'mnli', 'civil_comments']: data_ood = pd.read_feather(f"../data/{args.exp_name}/embs/{args.task}/{args.testdomain}/dataset_gpt_emb_test.feather")
    else: data_ood = pd.read_feather(f"../data/{args.exp_name}/embs/{args.task}/{args.testdomain}/dataset_gpt_emb_OOD.feather")

    label2idx = LABEL2IDX[args.task]
    idx2label = {v: k for k, v in label2idx.items()}
    # data_ood['Index_Y'] = data_ood['answer'].apply(lambda x: label2idx[x])
    # data_ood = data_ood.groupby('Index_Y', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 500)))

    print(f"Loading inference results for {args.task} from {args.ckp}...")
    results_ID = pd.read_feather(f"./results/{args.exp_name}/results_ind_gpt_emb_{args.ckp}_ID.feather")
    # results_ID = results_ID[results_ID['Index_T']==TASK2IDX[args.task]]
    results_ID = results_ID.sort_values(by='sample_idx', ignore_index=True)
    results_ID['C_norm'] = results_ID['C_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x))) 
    results_ID['S_norm'] = results_ID['S_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))

    print(f"Loading inference results for {args.testdomain} from {args.ckp}...")
    if args.adapt: path_ood = f"./results/{args.exp_name}/{args.task}/{args.testdomain}/{args.adapt_id}/results_ind_{args.emb_model}_emb_{args.task}_{args.testdomain}_{args.ckp}_ada_OOD.feather"
    else: path_ood = f"./results/{args.exp_name}/{args.task}/{args.testdomain}/results_ind_{args.emb_model}_emb_{args.task}_{args.testdomain}_{args.ckp}_OOD.feather"

    results_OOD = pd.read_feather(path_ood)
    results_OOD = results_OOD.sort_values(by='sample_idx').reset_index(drop=True)
    results_OOD['C_norm'] = results_OOD['C_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))
    results_OOD['S_norm'] = results_OOD['S_hat'].apply(lambda x: np.array(x) / np.linalg.norm(np.array(x)))

    results_OOD = results_OOD[results_OOD['sample_idx'].isin(data_ood.index)]
    data_ood = data_ood.sort_index()
    results_OOD = results_OOD.sort_values(by='sample_idx')

    eval_retrieval(args, data_id, data_ood, results_ID, results_OOD, label2idx, idx2label)
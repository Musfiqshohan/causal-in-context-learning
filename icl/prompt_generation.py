import sys
import os
import torch
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.utils.utils import DemoSampler, PromptConstructor, save_selected_indices
from icl.utils.utils import get_clusters

def main(args, df_id, demo_embs, target_embs, target_queries: list, std_demo=None, std_target=None, demo_s_embs=None, target_s_embs=None):
    demo_sampler = DemoSampler(device=args.device, target_batch_size=len(target_embs),
                               demo_batch_size=args.demo_batch_size, use_posneg=args.use_posneg,
                               data_id=df_id, num_points=args.num_points, top_t=args.top_t)
    
    if args.task_name.split('/')[-1] == 'wsc': task_name = 'nli'
    else: task_name = args.task_name.split('/')[0]
    prompt_constructor = PromptConstructor(exp_name=args.exp_name,
                                           task_name=task_name,
                                           use_system_prompt=args.use_system_prompt,
                                           use_instruction=args.use_instruction,
                                           is_icl=True if not args.icl_method in ['fix', 'zs'] else args.icl_method)
    
    prompt_list = []
    if args.num_shots == 0:
        for k, q in enumerate(target_queries):
            prompt_list.append(prompt_constructor([], [], q, style_idx=0))
    else:
        if args.exp_name == 'ood_nlp':
            class_list = df_id['answer'].unique()
            tmp_idx = -1*np.ones((len(target_embs), len(class_list)), dtype=int)
            tmp_dist = -1*np.ones((len(target_embs), len(class_list)), dtype=float)
            for c_idx, c in enumerate(class_list):
                class_rows = df_id[df_id['answer']==c].index
                demo_embs_c = demo_embs[class_rows]
                c_hat_distances, c_hat_indices = demo_sampler.get_demo_samples(demo_embs_c, target_embs, std_demo, std_target, 30, args.selection_method, demo_s_embs, target_s_embs)
                
                tmp_idx[:, c_idx] = class_rows[c_hat_indices[:, 0]].tolist()
                tmp_dist[:, c_idx] = c_hat_distances[:, 0]

            sorted_idx = np.argsort(tmp_dist, axis=1)
            c_hat_indices = np.take_along_axis(tmp_idx, sorted_idx, axis=1)
        elif args.exp_name == 'llm_ret' and args.top_t == 0 and args.num_points == 0:
            class_list = df_id['Task'].unique()
            tmp_idx = -1*np.ones((len(target_embs), len(class_list)), dtype=int)
            tmp_dist = -1*np.ones((len(target_embs), len(class_list)), dtype=float)
            for c_idx, c in enumerate(class_list):
                class_rows = df_id[df_id['Task']==c].index
                demo_embs_c = demo_embs[class_rows]
                c_hat_distances, c_hat_indices = demo_sampler.get_demo_samples(demo_embs_c, target_embs, std_demo, std_target, 30, args.selection_method, demo_s_embs, target_s_embs)
                
                tmp_idx[:, c_idx] = class_rows[c_hat_indices[:, 0]].tolist()
                tmp_dist[:, c_idx] = c_hat_distances[:, 0]

            sorted_idx = np.argsort(tmp_dist, axis=1)
            c_hat_indices = np.take_along_axis(tmp_idx, sorted_idx, axis=1)
            c_hat_indices = c_hat_indices[:, :-1]
        else:
            _, c_hat_indices = demo_sampler.get_demo_samples(demo_embs, target_embs, std_demo, std_target, 30, args.selection_method, demo_s_embs, target_s_embs)
    
        demo_x, demo_y = demo_sampler.get_demo_texts(c_hat_indices, n_shot=args.num_shots, x_title=args.x_title, y_title=args.y_title, random_order=True if args.icl_method == 'ccl_r' else False)

        save_selected_indices(args, demo_sampler._idx)

        for k, q in enumerate(target_queries):
            prompt_list.append(prompt_constructor(demo_x[k], demo_y[k], q, style_idx=0))

    return prompt_list

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='mgsm', help="mgsm, ood_nlp, ...")
    parser.add_argument('--task_name', type=str, default='mgsm', help="mgsm, sa/sst5, ...")
    parser.add_argument('--domain_name', type=str, default='id', help='id => in-domain, ood => all of ood, else => specific ood domain')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--demo_batch_size', type=int, default=2000)
    parser.add_argument('--overwrite', action='store_true')

    # Embedding model
    parser.add_argument('--emb_model', type=str, default='gpt')
    parser.add_argument('--adapt_ood', action='store_true')
    parser.add_argument('--adapt_id', type=str, default='random')

    # Prompt settings
    parser.add_argument('--x_title', type=str, default='question', help='for mgsm, question; for sa, text')
    parser.add_argument('--y_title', type=str, default='answer')
    parser.add_argument('--use_system_prompt', action='store_true')
    parser.add_argument('--use_instruction', action='store_true')
    parser.add_argument('--prompt_style', type=int, default=0)

    # ICL settings
    parser.add_argument('--ckp', type=str, default='000')
    parser.add_argument('--icl_method', type=str, choices=['icl', 'ccl', 'ccl_r', 'zs', 'ccl_s', 'fix'], default='ccl')
    parser.add_argument('--use_all', action='store_true', help='use all of the ID data for examples')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--use_posneg", action='store_true')
    parser.add_argument("--selection_method", type=str, default='cossim')
    parser.add_argument("--num_points", type=int, default=50)
    parser.add_argument("--top_t", type=int, default=0)

    parser = parser.parse_args()
    return parser

if __name__ == "__main__":
    args = get_parser()
    path = f"./results/{args.exp_name}/{args.task_name}/"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    setattr(args, 'device', f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.exp_name == 'mgsm':
        df_id = pd.read_pickle(f"../data/{args.exp_name}/embs/dataset_{args.emb_model}_emb_ID.pkl")
        df_emb_id = pd.read_pickle(f"../ccl/results/{args.exp_name}/final/results_{args.ckp}_{args.emb_model}_emb_ID.pkl")
    else:
        df_id = pd.read_feather(f"../data/{args.exp_name}/embs/dataset_{args.emb_model}_emb_ID.feather")
        if not args.use_all:
            df_id = df_id.iloc[df_id[df_id.Task==args.task_name.split('/')[0]].index]
        
        if args.use_all: df_emb_id = pd.read_feather(f"../ccl/results/{args.exp_name}/final/results_{args.ckp}_{args.emb_model}_emb_ID.feather")
        else:
            df_emb_id = pd.read_feather(f"../ccl/results/{args.exp_name}/final/results_{args.ckp}_{args.emb_model}_emb_ID.feather")
            df_emb_id = df_emb_id.iloc[df_id.index].reset_index(drop=True)
            df_id.reset_index(drop=True, inplace=True) 

    if args.domain_name == 'id':
        df_target = df_id    
        df_emb_target = df_emb_id
    else:
        if args.exp_name == 'mgsm':
            df_target = pd.read_pickle(f"../data/{args.exp_name}/embs/dataset_{args.emb_model}_emb_OOD.pkl")
            df_emb_target = pd.read_pickle(f"../ccl/results/{args.task_name}/final/results_{args.ckp}_{args.emb_model}_emb_OOD.pkl")
        else:
            if args.task_name.split('/')[-1] in ['amazon', 'mnli', 'civil_comments', 'copa', 'wsc', 'rte', 'sentiment140']: path_target = f"../data/{args.exp_name}/embs/{args.task_name}/dataset_{args.emb_model}_emb_test.feather"
            else: path_target = f"../data/{args.exp_name}/embs/{args.task_name}/dataset_{args.emb_model}_emb_OOD.feather"
            df_target = pd.read_feather(path_target)
            if args.adapt_ood: df_emb_target = pd.read_feather(f"../ccl/results/{args.exp_name}/{args.task_name}/{args.adapt_id}/final/results_{args.ckp}_{args.emb_model}_emb_OOD_adapt.feather")
            else: df_emb_target = pd.read_feather(f"../ccl/results/{args.exp_name}/{args.task_name}/final/results_{args.ckp}_{args.emb_model}_emb_OOD.feather")

    if len(df_id) != len(df_emb_id) or len(df_target) != len(df_emb_target):
        raise ValueError("Length mismatch between data and embedding dataframes")

    # Get Embeddings
    std_demo, std_target = None, None
    demo_s_embs, target_s_embs = None, None
    if args.icl_method in ['icl', 'zs', 'fix']:
        demo_embs = np.stack(df_emb_id['X'].values)
        target_embs = np.stack(df_emb_target['X'].values)
    elif args.icl_method in ['ccl', 'ccl_r']:
        if args.norm:
            demo_embs = np.stack(df_emb_id['C_norm'].values)
            target_embs = np.stack(df_emb_target['C_norm'].values)
            if args.selection_method in ['kl_td', 'kl_dt', 'wd']:
                std_demo = np.stack(df_emb_id['C_std'].values)
                std_target = np.stack(df_emb_target['C_std'].values)
            if args.selection_method in ['knn_cs', 'knn_mix', 'knn_cat']:
                demo_s_embs = np.stack(df_emb_id['S_norm'].values)
                target_s_embs = np.stack(df_emb_target['S_norm'].values)
        else:
            demo_embs = np.stack(df_emb_id['C_hat'].values)
            target_embs = np.stack(df_emb_target['C_hat'].values)
            if args.selection_method in ['kl_td', 'kl_dt', 'wd']:
                std_demo = np.stack(df_emb_id['C_std'].values)
                std_target = np.stack(df_emb_target['C_std'].values)
            if args.selection_method in ['knn_cs', 'knn_mix', 'knn_cat']:
                demo_s_embs = np.stack(df_emb_id['S_hat'].values)
                target_s_embs = np.stack(df_emb_target['S_hat'].values)
    elif args.icl_method == 'ccl_s':
        if args.norm:
            demo_embs = np.stack(df_emb_id['S_norm'].values)
            target_embs = np.stack(df_emb_target['S_norm'].values)
            if args.selection_method in ['kl_td', 'kl_dt', 'wd']:
                std_demo = np.stack(df_emb_id['S_std'].values)
                std_target = np.stack(df_emb_target['S_std'].values)
        else:
            demo_embs = np.stack(df_emb_id['S_hat'].values)
            target_embs = np.stack(df_emb_target['S_hat'].values)
            if args.selection_method in ['kl_td', 'kl_dt', 'wd']:
                std_demo = np.stack(df_emb_id['S_std'].values)
                std_target = np.stack(df_emb_target['S_std'].values)
    else:
        raise ValueError(f"unknown `icl_method` '{args.icl_method}'")
    
    if not args.icl_method in ['zs','fix']: SELECT = args.selection_method.upper()
    else: SELECT = ''

    if args.selection_method in ['kmeans_x', 'kmeans_c', 'kmeans_s']:
        SELECT += f'_TOP_{args.top_t}_POINTS_{args.num_points}'
    elif args.selection_method in ['votek_x', 'votek_c']:
        SELECT += f'_POINTS_{args.num_points}'

    if args.num_shots == 0: ICL_METHOD = args.icl_method.upper()
    else: ICL_METHOD = args.icl_method.upper() + f"_{args.num_shots}shots_{SELECT}"
    if args.use_posneg: ICL_METHOD += '_pn'
    if args.use_all: ICL_METHOD = ICL_METHOD + '_all'
    else: ICL_METHOD = ICL_METHOD + '_part'
    if args.adapt_ood: ICL_METHOD = ICL_METHOD + f'_adapt_{args.adapt_id}'
    
    if args.norm and args.icl_method in ['ccl', 'ccl_s', 'ccl_r']:
        if args.exp_name == 'mgsm': _filename = os.path.join(path, f"{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}_norm")
        else: _filename = os.path.join(path, f"{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}_norm")
    else:
        if args.exp_name == 'mgsm': _filename = os.path.join(path, f"{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}")
        else: _filename = os.path.join(path, f"{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}")

    print(f"Generated prompts will saved to {_filename}.json")
    setattr(args, 'prompt_path', _filename)

    prompt_list_all = []
    for _start in tqdm(range(0, len(df_target), args.batch_size)):
        _end = min(_start + args.batch_size, len(df_target))
        df_batch = df_target.iloc[_start:_end]
        if args.task_name.split('/')[-1] in ['piqa', 'wsc273', 'copa']:
            target_queries = df_batch['TextwOpt'].tolist()
        else: target_queries = df_batch[args.x_title].tolist()
        if std_target is not None: _p_list = main(args, df_id, demo_embs, target_embs[_start:_end], target_queries, std_demo, std_target[_start:_end], demo_s_embs, target_s_embs[_start:_end] if target_s_embs is not None else None)
        else: _p_list = main(args, df_id, demo_embs, target_embs[_start:_end], target_queries, std_demo, std_target, demo_s_embs, target_s_embs[_start:_end] if target_s_embs is not None else None)
        
        prompt_list_all.extend(_p_list)
    
    # Save json file
    with open(f"{_filename}.json", "w") as f:
        json.dump(prompt_list_all, f)

    print(f"Saved to {_filename}.json")
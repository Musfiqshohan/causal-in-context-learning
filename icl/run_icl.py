import os
import sys
import json
import time
import random
import pickle
import argparse
import pandas as pd
import wandb
import tempfile
from glob import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.utils.utils import LLMAgent, OpenLLMAgent, DflotLLMAgent, set_seed_all
from icl.utils.eval_llm import EvalLLM

LABEL2IDX={'ood_nlp':{
            'sa':{'negative':0, 'positive':1, 'neutral':2},
            'nli':{'entailment':0, 'neutral':1, 'contradiction':2},
            'td':{'benign':0, 'toxic':1}
            },
            'llm_ret':{
                'commonsense':{'A':0,'B':1},
                'coreference':{'A':0,'B':1},
                'sentiment':{'negative':0,'positive':1},
                'nli':{'yes':0,'no':1}
            }
        }

TASK2IDX={
'sa':0,
'nli':1,
'eqa':2,
'td':3,
'ner':4}

TASK2MNT = {
    'sa': 256,
    'nli': 50,
    'td': 256,
    'sentiment': 50,
    'commonsense': 50,
    'coreference': 50,
}

LLM_VER = {
    'llama': "meta-llama/Llama-3.2-3B-Instruct",
    'deepseek': 'deepseek/deepseek-r1',
    'gemma': 'google/gemma-3-27b-it',
    'qwen3': 'Qwen/Qwen3-32B',
    'qwen306b': 'Qwen/Qwen3-0.6B',
    'dflot-gemma': 'DFloat11/gemma-3-27b-it-DF11',
    'dflot-qwen3': 'DFloat11/Qwen3-32B-DF11',
    'bitnet': 'microsoft/bitnet-b1.58-2B-4T',
    'phi4-mini-it': 'microsoft/Phi-4-mini-instruct',
    'deepseek-mini': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'gemma3-mini-it': 'google/gemma-3-1b-it',
}

def single_response(args, prompt_list, eval_llm, llm_agent, response_filename, subset_ids, overwrite=False):
    # Load existing responses if file exists, otherwise start with empty dict
    if os.path.exists(f"{response_filename}.json"):
        if overwrite:
            response_dict = {}
            print("Overwriting existing responses")
        else:
            with open(f"{response_filename}.json", "r") as f:
                response_dict = json.load(f)
            print(f"Found {len(response_dict)} existing responses, continuing from last index")
    else:
        response_dict = {}
        print("Starting new response collection from index 0")

    _response_list = []
    for i in tqdm(subset_ids, total=len(subset_ids)):
        if str(i) in response_dict:
            _response = response_dict[str(i)]
        else:
            messages = prompt_list[i]

            _response = llm_agent.get_response(messages)

            # Update response dict and save after each response
            response_dict[str(i)] = _response
            with open(f"{response_filename}.json", "w") as f:
                json.dump(response_dict, f)
        _response_list.append(_response)

        if i % 100 == 0:
            print(f"\nCompleted {i+1}/{len(subset_ids)} samples")
    
    print(f"\nEval {i+1}/{len(subset_ids)} samples")
    pred_and_gt = eval_llm(_response_list, subset_ids)

    with open(f"{response_filename}_pred_and_gt.pkl", "wb") as f:
        pickle.dump(pred_and_gt, f)

    return _response_list

def batch_response(args, prompt_list, eval_llm, llm_agent, response_filename, subset_ids, overwrite=False):
    # Load existing responses if file exists, otherwise start with empty dict
    if os.path.exists(f"{response_filename}.json"):
        if overwrite:
            response_dict = {}
            print("Overwriting existing responses")
        else:
            with open(f"{response_filename}.json", "r") as f:
                response_dict = json.load(f)
            print(f"Found {len(response_dict)} existing responses, continuing from last index")
    else:
        response_dict = {}
        print("Starting new response collection from index 0")

    _response_list = []
    batch_size = llm_agent.batch_size

    current_completed_samples = 0
    for batch_start in tqdm(range(0, len(subset_ids), batch_size), total=(len(subset_ids) // batch_size) + 1):
        batch_end = min(batch_start + batch_size, len(subset_ids))
        batch_indices = subset_ids[batch_start:batch_end]

        batch_messages = []
        batch_processed = []
        for i in batch_indices:
            if str(i) in response_dict:
                _response_list.append(response_dict[str(i)])
                batch_processed.append(True)
            else:
                batch_messages.append(prompt_list[i])
                batch_processed.append(False)

        if batch_messages:
            batch_responses = llm_agent.get_response(batch_messages)

            for i, response in zip(batch_indices, batch_responses):
                if not batch_processed[batch_indices.index(i)]:
                    response_dict[str(i)] = response
                    _response_list.append(response)

            with open(f"{response_filename}.json", "w") as f:
                json.dump(response_dict, f)

        current_completed_samples += len(batch_responses)
        print(f"\nCompleted {current_completed_samples}/{len(subset_ids)} samples")
    
    print(f"\nEval {current_completed_samples}/{len(subset_ids)} samples")
    pred_and_gt = eval_llm(_response_list, subset_ids)
    with open(f"{response_filename}_pred_and_gt.pkl", "wb") as f:
        pickle.dump(pred_and_gt, f)

    return _response_list

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='mgsm')
    parser.add_argument('--n_subset', type=int, default=1000)
    parser.add_argument('--task_name', type=str, default='mgsm')
    parser.add_argument('--domain_name', type=str, default='id', help='id => in-domain, ood => all of ood, else => specific ood domain')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing responses')
    
    # ICL settings
    parser.add_argument('--emb_model', type=str, default='gpt')
    parser.add_argument('--icl_method', type=str, choices=['icl', 'ccl', 'ccl_s', 'ccl_r', 'zs', 'fix', 'llmr'], default='ccl')
    parser.add_argument('--ckp', type=str, default='000')
    parser.add_argument('--adapt_ood', action='store_true')
    parser.add_argument('--adapt_id', type=str, default='random')
    parser.add_argument('--use_all', action='store_true', help='use all of the ID data for examples')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--num_shots', type=int, default=3)
    parser.add_argument("--selection_method", type=str, default='cossim')
    parser.add_argument("--use_posneg", action='store_true')
    parser.add_argument("--num_points", type=int, default=50)
    parser.add_argument("--top_t", type=int, default=1)

    # LLM model
    parser.add_argument('--llm_model', type=str, default='gpt')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--frequency_penalty', type=float, default=0)
    parser.add_argument('--presence_penalty', type=float, default=0)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    
    if args.icl_method == 'llmr':
        ICL_METHOD = args.icl_method.upper()
        _filename = f"./results/llm_ret/llmr/{args.task_name}/ICL_8shots_LLMR"
    else:
        if args.emb_model in ['gpt', "intfloat"]:
            if not args.icl_method in ['zs', 'fix', 'llmr']: SELECT = args.selection_method.upper()
            if args.selection_method in ['kmeans_x', 'kmeans_c', 'kmeans_s']:
                SELECT += f'_TOP_{args.top_t}_POINTS_{args.num_points}'
            if args.selection_method in ['votek_x', 'votek_c']:
                SELECT += f'_POINTS_{args.num_points}'
                
            if args.num_shots == 0: ICL_METHOD = args.icl_method.upper()
            else: ICL_METHOD = args.icl_method.upper() + f"_{args.num_shots}shots_{SELECT}"
            if args.use_posneg: ICL_METHOD += '_pn'
            if args.use_all: ICL_METHOD = ICL_METHOD + '_all'
            else: ICL_METHOD = ICL_METHOD + '_part'
            if args.adapt_ood: ICL_METHOD = ICL_METHOD + f'_adapt_{args.adapt_id}'

            if args.norm and args.icl_method in ['ccl', 'ccl_s', 'ccl_r']:
                _filename = f"./results/{args.exp_name}/{args.task_name}/{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}_norm"
            else:
                if args.icl_method in ['ccl', 'ccl_s', 'ccl_r']:
                    _filename = f"./results/{args.exp_name}/{args.task_name}/{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}"
                else:
                    _filename = f"./results/{args.exp_name}/{args.task_name}/{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}"
                    if len(glob(f"{_filename}.json")) == 0:
                        _filename = f"./results/{args.exp_name}/{args.task_name}/{args.ckp}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}"
        else:
            _filename = f'./results/{args.exp_name}/{args.emb_model}/{args.task_name}/{args.ckp}_ICL_{args.num_shots}shots_{args.emb_model.upper()}'
            
    print(f"Load {_filename}")

    with open(f"{_filename}.json", "r") as f:
        prompt_list_all = json.load(f)
    
    print(f"Total Test Samples: {len(prompt_list_all)}")

    set_seed_all(args.seed)

    if args.task_name.split('/')[-1] in ['amazon', 'mnli', 'civil_comments', 'copa', 'wsc', 'rte', 'sentiment140']:
            dataset_ood = pd.read_feather(f"../data/{args.exp_name}/embs/{args.task_name}/dataset_gpt_emb_test.feather")
    else: dataset_ood = pd.read_feather(f"../data/{args.exp_name}/embs/{args.task_name}/dataset_gpt_emb_OOD.feather")

    if args.task_name.split('/')[-1] == 'wsc': gt_answers = dataset_ood['answer'].tolist()
    elif args.task_name.split('/')[0] in ['commonsense', 'coreference']: gt_answers = dataset_ood['label'].tolist()
    else: gt_answers = dataset_ood['answer'].tolist()

    print(f"Seed: {args.seed}")
    if args.n_subset > 0:
        print(f"Test Subset: {args.n_subset}/{len(prompt_list_all)}")
    else:
        print(f"Test Subset: All")
    label2idx = LABEL2IDX[args.exp_name][args.task_name.split('/')[0]]
    if args.task_name.split('/')[-1] == 'wsc': label2idx = LABEL2IDX[args.exp_name]['nli']
    idx2label = {v: k for k, v in label2idx.items()}
    
    if args.task_name.split('/')[-1] == 'wsc': dataset_ood['Index_Y'] = dataset_ood['answer'].apply(lambda x: label2idx[x])
    elif args.task_name.split('/')[0] in ['commonsense', 'coreference']: dataset_ood['Index_Y'] = dataset_ood['label'].apply(lambda x: label2idx[x])
    else: dataset_ood['Index_Y'] = dataset_ood['answer'].apply(lambda x: label2idx[x])
    
    if args.n_subset > 0:
        dataset_ood = dataset_ood.groupby('Index_Y', group_keys=False).apply(lambda x: x.sample(n=min(len(x), args.n_subset//len(idx2label)), random_state=args.seed))
        subset_ids = dataset_ood.index.tolist()
    else:
        subset_ids = list(range(len(prompt_list_all)))

    # LLM agent
    LLM_NAME = f"{args.llm_model}_{LLM_VER[args.llm_model]}"
    print("LLM Agent: ", LLM_NAME)

    if args.llm_model != 'gpt':
        use_dfloat = args.llm_model.split('-')[0] == 'dflot'
        if use_dfloat:
            llm_agent = DflotLLMAgent(LLM_NAME,
                            seed=args.seed,
                            max_tokens=args.max_tokens,
                            max_new_tokens=TASK2MNT[args.task_name.split('/')[0]],
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            batch_size=args.batch_size)
        else:
            llm_agent = OpenLLMAgent(LLM_NAME,
                            seed=args.seed,
                            max_tokens=args.max_tokens,
                            max_new_tokens=TASK2MNT[args.task_name.split('/')[0]],
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            batch_size=args.batch_size)
    else:
        llm_agent = LLMAgent(LLM_NAME,
                         seed=args.seed,
                         max_tokens=args.max_tokens,
                         temperature=args.temperature,
                         frequency_penalty=args.frequency_penalty,
                         presence_penalty=args.presence_penalty,
                         batch_size=args.batch_size)

    # Run LLM
    if args.exp_name == 'mgsm':
        if args.norm and args.icl_method=='ccl': response_filename = f"./response/{args.exp_name}/{args.llm_model}/response_{args.llm_model}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}_norm"
        else: response_filename = f"./response/{args.exp_name}/{args.llm_model}/response_{args.llm_model}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}"
    else:
        if args.icl_method in ['ccl', 'ccl_s', 'ccl_r']: response_path = f"./response/{args.exp_name}/{args.task_name}/{args.llm_model}/{args.ckp}"
        elif args.emb_model == 'llmr': response_path = f"./response/{args.exp_name}/{args.task_name}/{args.llm_model}"
        else: response_path = f"./response/{args.exp_name}/{args.task_name}/{args.llm_model}"

        if args.norm: response_filename = f"{response_path}/response_{args.llm_model}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}_norm"
        elif args.emb_model == 'llmr': response_filename = f"{response_path}/response_{args.llm_model}_prompt_{args.domain_name}_emb_{args.emb_model}"
        else: response_filename = f"{response_path}/response_{args.llm_model}_{ICL_METHOD}_prompt_{args.domain_name}_emb_{args.emb_model}"
    
    print(f"Response result will be saved at {response_filename}")
    if not os.path.exists(response_path):
        os.makedirs(response_path, exist_ok=True)
    
    # Log configuration to config.txt
    config_path = os.path.join(response_path, 'config.txt')
    with open(config_path, 'a') as f:
        f.write(f"\n\n=== Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(str(vars(args)) + "\n")
    
    eval_llm = EvalLLM(task=args.task_name.split('/')[0], domain=args.task_name.split('/')[1], llm=args.llm_model, gt_answers=gt_answers, benchmark=args.exp_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wandb.init(
                project='CCL_LLM_Results',
                config=args,
                save_code=True,
                dir=tmp_dir,
            )

        if llm_agent.batch_size == 1:
            response_list = single_response(args,prompt_list_all, eval_llm, llm_agent, response_filename, subset_ids, overwrite=args.overwrite)
        else:
            response_list = batch_response(args, prompt_list_all, eval_llm, llm_agent, response_filename, subset_ids, overwrite=args.overwrite)

        wandb.finish()
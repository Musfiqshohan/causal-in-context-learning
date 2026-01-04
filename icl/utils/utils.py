import os
import json
import time
import torch
import copy
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors

from openai import OpenAI
from google import genai
from google.genai import types

from dfloat11 import DFloat11Model
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from ratelimit import limits, sleep_and_retry
CALLS = 1
RATE_LIMIT_PERIOD = 5

from icl.utils.distance_metric import pairwise_kl_div_two_batches, pairwise_wasserstein_distance_two_batches, pairwise_dot_two_batches, pairwise_euclidean_distance_two_batches, pairwise_manhattan_distance_two_batches

def set_seed_all(seed):
    from transformers import set_seed
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    try:
        torch.set_deterministic_debug_mode(True)
    except AttributeError:
        pass

def get_clusters(df, var, n_clusters):
    emb = np.stack(df[var].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(emb)
    return kmeans.labels_.tolist()

def save_selected_indices(args, idx):
    # Load existing indices if file exists
    _indices = copy.deepcopy(idx)
    _indices = _indices.tolist()

    indices_dict = {}
    if os.path.exists(args.prompt_path + "_indices.json") and not args.overwrite:
        with open(args.prompt_path + "_indices.json", "r") as f:
            indices_dict = json.load(f)
        start_idx = max(map(int, indices_dict.keys())) + 1
    else:
        start_idx = 0
        indices_dict = {}

    # Create dictionary with new indices
    for i, indices in enumerate(_indices, start=start_idx):
        indices_dict[str(i)] = indices
        
    # Save updated indices
    with open(args.prompt_path + "_indices.json", "w") as f:
        json.dump(indices_dict, f)

    setattr(args, 'overwrite', False)
    
class DemoSampler:
    def __init__(self, device, target_batch_size=1000, demo_batch_size=1000, use_posneg=False, data_id=None, num_points=50, top_t=1):
        self.device = device
        self.target_batch_size = target_batch_size
        self.demo_batch_size = demo_batch_size
        self.use_posneg = use_posneg
        self.data_id = data_id
        self.metric = None
        self.num_points = num_points
        self.top_t = top_t

    def get_demo_samples(self, emb_demo, emb_target, std_demo, std_target, num_neighbors, metric='knn', demo_s_embs=None, target_s_embs=None):
        self.metric = metric
        self.emb_demo = emb_demo
        self.emb_target = emb_target
        if not metric.split("_")[0] in ['knn', 'votek']:
            if type(emb_demo) == np.ndarray: emb_demo = torch.from_numpy(emb_demo)
            if type(emb_target) == np.ndarray: emb_target = torch.from_numpy(emb_target)
            if type(std_demo) == np.ndarray: std_demo = torch.from_numpy(std_demo)
            if type(std_target) == np.ndarray: std_target = torch.from_numpy(std_target)

        if metric in ['knn', 'kmeans_c', 'kmeans_s', 'kmeans_x']: # Euclidean distance as a default metric
            nbrs = NearestNeighbors(n_neighbors=len(emb_demo), algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
            distances, indices = nbrs.kneighbors(emb_target)
        elif metric == 'knn_cs':
            c_nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
            _, indices = c_nbrs.kneighbors(emb_target)
            candidate_embs = demo_s_embs[indices]
            dist_candidate = np.sum((candidate_embs - target_s_embs[:, np.newaxis, :])**2, axis=2)
            order = np.argsort(dist_candidate, axis=1)
            sorted_indices = np.take_along_axis(indices, order, axis=1)
            return dist_candidate, sorted_indices
        elif metric == 'knn_mix':
            emb_demo_mixup = (0.9 * emb_demo) + (0.1 * demo_s_embs)
            emb_target_mixup = (0.9 * emb_target) + (0.1 * target_s_embs)
            nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo_mixup)
            distances, indices = nbrs.kneighbors(emb_target_mixup)
        elif metric == 'knn_cat':
            emb_demo_cat = np.concatenate([emb_demo, demo_s_embs], axis=1)
            emb_target_cat = np.concatenate([emb_target, target_s_embs], axis=1)
            nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo_cat)
            distances, indices = nbrs.kneighbors(emb_target_cat)
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
        elif metric in ['dot']: # dot product
            distances, indices = pairwise_dot_two_batches(emb_target, emb_demo, self.target_batch_size, self.demo_batch_size, self.device)
        elif metric == 'man': # Manhattan distance
            distances, indices = pairwise_manhattan_distance_two_batches(emb_target, emb_demo, self.target_batch_size, self.demo_batch_size, self.device)
        elif metric == 'votek_x':
            with open("./utils/selected_indices_x_list.json", "r") as f:
                selected_indices = json.load(f)[str(self.num_points)]
            emb_demo = emb_demo[np.array(selected_indices)]
            cos_sim = pairwise.cosine_similarity(emb_target, emb_demo)
            indices = np.argsort(-cos_sim, axis=1)
            distances = -cos_sim
        elif metric == 'votek_c':
            with open("./utils/selected_indices_c_list.json", "r") as f:
                selected_indices = json.load(f)[str(self.num_points)]
            emb_demo = emb_demo[np.array(selected_indices)]
            cos_sim = pairwise.cosine_similarity(emb_target, emb_demo)
            indices = np.argsort(-cos_sim, axis=1)
            distances = -cos_sim

        return distances, indices
    
    def get_diverse_y_indices(self, indices, num_clusters):
        clusters = self.data_id.loc[indices.flatten(), 'cluster_y'].values.reshape(indices.shape)

        selected_indices = []
        for row, cluster_row in zip(indices, clusters):
            _, unique_indices = np.unique(cluster_row, return_index=True)
            selected = row[unique_indices[:num_clusters]]
            selected_indices.append(selected.tolist())

        return selected_indices
    
    def get_diverse_c_indices(self, indices, num_clusters):
        clusters = self.data_id.loc[indices.flatten(), 'cluster_c'].values.reshape(indices.shape)

        selected_indices = []

        for row, cluster_row in zip(indices, clusters):
            _, unique_indices = np.unique(cluster_row, return_index=True)
            selected = row[unique_indices[:num_clusters]]
            selected_indices.append(selected.tolist())

        return selected_indices

    def get_centroid_idices(self, indices, metric, n_shot):
        N = indices.shape[0]
        center_idx_per_sample = np.empty((N, n_shot), dtype=int)
        p = self.num_points

        print(f"{n_shot} clusters with {p} samples")
        if metric == 'c':
            for i in range(N):
                demo_idx = indices[i, :p] # 50, 100
                demo_embs = self.emb_demo[demo_idx]

                kmeans = KMeans(n_clusters=n_shot, random_state=0, n_init="auto").fit(demo_embs)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_

                for cluster_id in range(n_shot):
                    cluster_mask = (labels == cluster_id)
                    if np.any(cluster_mask):
                        cluster_embs = demo_embs[cluster_mask]
                        dists = np.linalg.norm(cluster_embs - centers[cluster_id], axis=1)
                        min_idx_within_cluster = np.argmin(dists)
                        original_idx = np.where(cluster_mask)[0][min_idx_within_cluster]
                        center_idx_per_sample[i, cluster_id] = demo_idx[original_idx]
                    else:
                        center_idx_per_sample[i, cluster_id] = demo_idx[0]
        elif metric == 's':
            for i in range(N):
                demo_idx = indices[i, :p] # 50, 100
                demo_embs = np.stack(self.data_id['X'].iloc[demo_idx].values)

                kmeans = KMeans(n_clusters=n_shot, random_state=0, n_init="auto").fit(demo_embs)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_

                for cluster_id in range(n_shot):
                    cluster_mask = (labels == cluster_id)
                    if np.any(cluster_mask):
                        cluster_embs = demo_embs[cluster_mask]
                        dists = np.linalg.norm(cluster_embs - centers[cluster_id], axis=1)
                        min_idx_within_cluster = np.argmin(dists)
                        original_idx = np.where(cluster_mask)[0][min_idx_within_cluster]
                        center_idx_per_sample[i, cluster_id] = demo_idx[original_idx]
                    else:
                        center_idx_per_sample[i, cluster_id] = demo_idx[0]
        elif metric == 'x':
            for i in range(N):
                demo_idx = indices[i, :p] # 50, 100
                demo_embs = np.stack(self.data_id['X'].iloc[demo_idx].values)

                kmeans = KMeans(n_clusters=n_shot, random_state=0, n_init="auto").fit(demo_embs)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_

                for cluster_id in range(n_shot):
                    cluster_mask = (labels == cluster_id)
                    if np.any(cluster_mask):
                        cluster_embs = demo_embs[cluster_mask]
                        dists = np.linalg.norm(cluster_embs - centers[cluster_id], axis=1)
                        min_idx_within_cluster = np.argmin(dists)
                        original_idx = np.where(cluster_mask)[0][min_idx_within_cluster]
                        center_idx_per_sample[i, cluster_id] = demo_idx[original_idx]
                    else:
                        center_idx_per_sample[i, cluster_id] = demo_idx[0]
        return center_idx_per_sample

    def get_demo_texts(self, indices, n_shot=3, x_title='question', y_title='answer', random_order=False):
        if n_shot == 0: return [], []
        
        if self.metric == None: raise ValueError("Metric is not set")
        if isinstance(indices, torch.Tensor): indices = indices.cpu().numpy()

        if self.use_posneg:
            n_shot += (n_shot % 2)
            # Get top n_shot//2 and bottom n_shot//2 indices
            top_indices = indices[:, :n_shot//2]  # Top half
            bottom_indices = indices[:, -n_shot//2:]  # Bottom half
            self._idx = np.concatenate([bottom_indices, top_indices], axis=1)  # Combine them
        if random_order:
            top1_idx = indices[:, :1]
            random_indices =  np.array([np.random.choice(row[1:20], size=n_shot-1, replace=False) for row in indices])
            _idx = np.concatenate([top1_idx, random_indices], axis=1)
            self._idx = _idx[:, :n_shot][:, ::-1]
        elif self.top_t and self.top_t < n_shot:
            t = self.top_t
            print(f"Use Top-{t} shots")
            if self.metric == 'kmeans_c':
                if t > 0: top1_idx = indices[:, :t]
                self.centroid_idx = self.get_centroid_idices(indices[:, t:], 'c', n_shot-t)
                if t > 0: self._idx = np.concatenate([self.centroid_idx, top1_idx], axis=1)
                else: self._idx = self.centroid_idx
            elif self.metric == 'kmeans_s':
                if t > 0: top1_idx = indices[:, :t]
                self.centroid_idx = self.get_centroid_idices(indices[:, t:], 's', n_shot-t)
                if t > 0: self._idx = np.concatenate([self.centroid_idx, top1_idx], axis=1)
                else: self._idx = self.centroid_idx
            elif self.metric == 'kmeans_x':
                if t > 0: top1_idx = indices[:, :t]
                self.centroid_idx = self.get_centroid_idices(indices[:, t:], 'x', n_shot-t)
                if t > 0: self._idx = np.concatenate([self.centroid_idx, top1_idx], axis=1)
                else: self._idx = self.centroid_idx
            else: self._idx = indices[:, :n_shot][:, ::-1]
        else: self._idx = indices[:, ::-1]

        demo_x, demo_y = [], []
        for l in self._idx:
            demo_x.append(self.data_id[x_title].iloc[l].tolist())
            demo_y.append(self.data_id[y_title].iloc[l].tolist())
        return demo_x, demo_y

class PromptConstructor:
    def __init__(self, exp_name, task_name,
                 use_system_prompt=False,
                 use_instruction=False,
                 is_icl=True):

        self.exp_name = exp_name
        self.task_name = task_name
        self.use_system_prompt = use_system_prompt
        self.use_instruction = use_instruction

        with open("./prompts/templates.json", "r") as f:
            self.templates = json.load(f)

        self.input_prompt = ""
        if use_system_prompt or use_instruction:
            with open("./prompts/default_prompts.json", "r") as f:
                self.default_prompts = json.load(f)
            if use_system_prompt: self.system_prompt = self.default_prompts[exp_name][task_name]["system"]
            if use_instruction and is_icl == True: self.instruction_prompt = self.default_prompts[exp_name][task_name]["instruction"]['icl']
            else: self.instruction_prompt = self.default_prompts[exp_name][task_name]["instruction"][is_icl]
            self.input_prompt = self.default_prompts[exp_name][task_name]["input"]
    
    def set_system_prompt(self):
        return [{"role": "system", "content": self.system_prompt}]

    def set_instruction_prompt(self):
        return self.instruction_prompt
    
    def sanitize_text(self, text: str) -> str:
        return text.replace('\n', ' ').replace('  ', ' ')

    def set_query_prompt(self, demo_samples_x, demo_samples_y, query_x, style_idx=0):
        n_demo = len(demo_samples_x)
        
        prompt=''
        if n_demo > 0:
            for i in range(n_demo):
                prompt += self.templates[self.exp_name][self.task_name]["X"][style_idx].format(self.sanitize_text(demo_samples_x[i]))
                prompt += self.templates[self.exp_name][self.task_name]["Y"][style_idx].format(demo_samples_y[i])
            # prompt += "\n"
        
        if self.input_prompt != "": prompt += self.input_prompt
        prompt += self.templates[self.exp_name][self.task_name]["X"][style_idx].format(query_x)
        if self.templates[self.exp_name][self.task_name]["A"][style_idx] != "":
            prompt += self.templates[self.exp_name][self.task_name]["A"][style_idx]
        
        return prompt

    def __call__(self, demo_samples_x, demo_samples_y, query_x, style_idx=0):
        if self.use_system_prompt: sys_prompt = self.set_system_prompt()
        else: sys_prompt = None
        if self.use_instruction: inst_prompt = self.set_instruction_prompt()
        else: inst_prompt = ''

        query_prompt = self.set_query_prompt(demo_samples_x, demo_samples_y, query_x, style_idx)

        query_prompt = [{'role':'user',
                         'content':inst_prompt + query_prompt}]
        
        if sys_prompt is not None: return sys_prompt + query_prompt
        else: return query_prompt

class OpenLLMAgent():
    def __init__(self, llm_name, seed=0, max_tokens=4096, max_new_tokens=30, temperature=0.0, top_p=1, n=1, frequency_penalty=0, presence_penalty=0, batch_size=256):
        from transformers import pipeline

        self.llm = llm_name.split('_')[0]
        self.version = llm_name.split('_')[1]

        if self.llm == 'llama':
            self.model = AutoModelForCausalLM.from_pretrained(self.version, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.version, padding_side="left", truncation_side="left")
        elif self.llm in ['bitnet']:
            self.model = AutoModelForCausalLM.from_pretrained(self.version, torch_dtype=torch.bfloat16, device_map="auto", cache_dir='/data1/hoyun/models', trust_remote_code=False)
            self.tokenizer = AutoTokenizer.from_pretrained(self.version, padding_side="left", truncation_side="left", cache_dir='/data1/hoyun/models')
        elif self.llm in ['gemma3-mini-it']:
            self.model = AutoModelForCausalLM.from_pretrained(self.version, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.version, padding_side="left", truncation_side="left", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.version, torch_dtype=torch.bfloat16, device_map="auto", cache_dir='/data1/hoyun/models', trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.version, padding_side="left", truncation_side="left", cache_dir='/data1/hoyun/models')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_tokens = min(self.tokenizer.model_max_length, max_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.seed = seed
        self.batch_size = batch_size

        print(f"LLM Generation Setting: max_tokens={self.max_tokens}, max_new_tokens={self.max_new_tokens}")

    def get_response(self, messages: list):
        responses = []
        if self.llm in ['qwen3', 'qwen306b']:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)            
        else:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id    
            )

        batch_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses.extend(batch_responses)
        return responses
    
class DflotLLMAgent(OpenLLMAgent):
    def __init__(self, llm_name, seed=0, max_tokens=4096, max_new_tokens=30, temperature=0.0, top_p=1, n=1, frequency_penalty=0, presence_penalty=0, batch_size=256):
        super().__init__(llm_name, seed, max_tokens, max_new_tokens, temperature, top_p, n, frequency_penalty, presence_penalty, batch_size)

        self.llm = llm_name.split('_')[0]
        self.version = llm_name.split('_')[1]

        print(f"Loading {self.version} with torch_dtype=torch.bfloat16")
        self.model = DFloat11Model.from_pretrained(self.version, device_map="auto", cache_dir='/data1/hoyun/models')
        self.tokenizer = AutoTokenizer.from_pretrained(self.version, cache_dir='/data1/hoyun/models')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

class LLMAgent():
    def __init__(self, llm_name, seed=0, max_tokens=1024, temperature=0, top_p=1, n=1, frequency_penalty=0, presence_penalty=0, batch_size=1):
        self.llm = llm_name.split('_')[0]
        self.version = llm_name.split('_')[1]
    
        if self.llm in ['gpt', 'turbo']:
            api_key = os.environ.get('OPENAI_API_KEY')
            self.client = OpenAI(
                api_key=api_key,
            )

        elif self.llm == 'deepseek':
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get('OPENROUTER_API_KEY')
            )

        elif self.llm == 'llama':
            api_key = os.environ.get('NIM_API_KEY')
            self.client = OpenAI(
                                base_url = "https://integrate.api.nvidia.com/v1",
                                api_key=api_key
            )

        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.n = n
        self.seed = seed
    
    def get_response(self, messages: list):
        response = self.client.chat.completions.create(
            seed=self.seed,
            model=self.version,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.n,
        )

        if self.n > 1:
            all_responses = [response.choices[i].message.content for i in range(len(response.choices))]
            return all_responses
        return [response.choices[0].message.content]
    
    @sleep_and_retry
    @limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
    def get_response_w_limiter(self, messages: list, is_zs=False):
        response = self.client.chat.completions.create(
            seed=self.seed,
            model=self.version,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.n,
        )

        try:
            return [response.text]
        except: print("Error in get_response_w_limiter"); return ['']
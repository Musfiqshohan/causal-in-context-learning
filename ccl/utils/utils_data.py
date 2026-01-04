import numpy as np
import torch as tc
from torch.utils.data import Dataset
from types import SimpleNamespace

class HighDimSCMRealWorldDataset(Dataset):
    def __init__(self, df, subtask=False, y_dtype='emb', y_emb_option='default', use_fc=False, is_ood=False, env_idx=None):
        super(HighDimSCMRealWorldDataset, self)
        
        # if env_idx is None: self.df = df
        # else: self.df = df[df['Index_E'] == env_idx]
        self.df = df.query("Index_E == @env_idx") if env_idx is not None else df
        
        self.X = tc.FloatTensor(np.stack(self.df['X'].values))
        self.E = tc.FloatTensor(np.stack(self.df['E'].values))
        self.T = tc.FloatTensor(np.stack(self.df['T'].values))

        if subtask:
            self.T = tc.FloatTensor(np.stack(self.df['ST'].values))
            self.label_subT = tc.LongTensor(self.df['Index_ST'].values)

        if not is_ood:
            if y_dtype == 'emb':
                if y_emb_option == 'long': self.Y = tc.FloatTensor(np.stack(self.df['Y_long'].values))
                elif y_emb_option == 'eq': self.Y = tc.FloatTensor(np.stack(self.df['Y_eq'].values))
                else: self.Y = tc.FloatTensor(np.stack(self.df['Y'].values))
                if use_fc: self.Y_num = tc.FloatTensor(np.stack(self.df['answer'].values))

            elif y_dtype == 'regression':
                self.Y = tc.FloatTensor(np.stack(self.df['answer'].values))
            elif y_dtype == 'classification':
                self.Y = tc.LongTensor(tc.from_numpy(self.df['Index_Y'].values))
                self.num_class = self.df['Index_Y'].max() + 1
        else: self.Y = tc.FloatTensor(np.stack(self.df['Y'].values))

        self.label_T = tc.LongTensor(tc.from_numpy(self.df['Index_T'].values))
        self.label_E = tc.LongTensor(tc.from_numpy(self.df['Index_E'].values))
        self.index = tc.LongTensor(tc.from_numpy(self.df.index.astype(int).values))

        self.subtask = subtask
        self.use_fc = use_fc
        self.is_ood = is_ood

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return SimpleNamespace(
            X=self.X[idx],
            Y=self.Y[idx],
            T=self.T[idx],
            E=self.E[idx],
            label_T=self.label_T[idx],
            label_E=self.label_E[idx],
            index=self.index[idx],
            label_subT=self.label_subT[idx] if self.subtask else None,
            Y_num=self.Y_num[idx] if self.use_fc else None
        )
    
    # def __getitem__(self, idx):
    #     x = self.X[idx]
    #     y = self.Y[idx]
    #     e = self.E[idx]
    #     t = self.T[idx]

    #     label_t = self.label_T[idx]
    #     label_e = self.label_E[idx]
    #     index = self.index[idx]

    #     _dict = {'X': x, 'Y': y, 'T':t, 'E':e, 'label_T':label_t, 'label_E':label_e, 'index':index}

    #     if self.subtask:
    #         label_subT = self.label_subT[idx]
    #         _dict['label_subT'] = label_subT
    #     if self.use_fc:
    #         y_num = self.Y_num[idx]
    #         _dict['Y_num'] = y_num

    #     return _dict

class HighDimSCMSyntheticDataset(Dataset):
    def __init__(self, df):
        super(HighDimSCMSyntheticDataset, self)

        self.df = df
        self.X = tc.FloatTensor(np.stack(self.df['X'].values))
        self.Y = tc.FloatTensor(np.stack(self.df['Y'].values))
        self.E = tc.FloatTensor(np.stack(self.df['E'].values))
        self.T = tc.FloatTensor(np.stack(self.df['T'].values))
        self.C = tc.FloatTensor(np.stack(self.df['C'].values))
        self.S = tc.FloatTensor(np.stack(self.df['S'].values))
        self.label_T = tc.LongTensor(self.df['Index_T'].values)
        self.label_E = tc.LongTensor(self.df['Index_E'].values)
        self.index = tc.LongTensor(tc.from_numpy(self.df.index.astype(int).values))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'Y': self.Y[idx],
            'C': self.C[idx],
            'S': self.S[idx],
            'T': self.T[idx],
            'E': self.E[idx],
            'label_T': self.label_T[idx],
            'label_E': self.label_E[idx],
            'index': self.index[idx]
        }
    
def load_dataset(ag, exp_name, data_id_dir, data_ood_dir, emb_model=None, y_dtype=None, y_emb_option='default'):

    import pandas as pd
    env_idx = None

    if exp_name == 'toy':
        df_ID = pd.read_pickle(f"../data/synthetic_dataset/dataset_{emb_model}_ID.pickle")
        df_OOD = pd.read_pickle(f"../data/synthetic_dataset/dataset_{emb_model}_OOD.pickle")
    
    if exp_name == 'mgsm':
        if len(data_ood_dir.split("_")) == 2: env_idx = int(data_ood_dir.split("_")[1])

        if y_emb_option == 'long': df_ID = pd.read_pickle(f"../data/{data_id_dir}/embs/dataset_{emb_model}_{y_emb_option}_emb_ID.pkl")
        else: df_ID = pd.read_pickle(f"../data/{data_id_dir}/embs/dataset_{emb_model}_emb_ID.pkl")

    if exp_name == 'ood_nlp':
        df_ID = pd.read_feather(f"../data/ood_nlp/embs/dataset_{emb_model}_emb_ID.feather")

        if ag.deploy_test: df_OOD = pd.read_feather(f"../data/ood_nlp/embs/{data_ood_dir}/dataset_{emb_model}_emb_test.feather")
        else: df_OOD = pd.read_feather(f"../data/ood_nlp/embs/{data_ood_dir}/dataset_{emb_model}_emb_OOD.feather")

    if exp_name == 'llm_ret':
        df_ID = pd.read_feather(f"../data/llm_ret/embs/dataset_{emb_model}_emb_ID.feather")

        if ag.deploy_test: df_OOD = pd.read_feather(f"../data/llm_ret/embs/{data_ood_dir}/dataset_{emb_model}_emb_test.feather")
        else: df_OOD = pd.read_feather(f"../data/llm_ret/embs/{data_ood_dir}/dataset_{emb_model}_emb_OOD.feather")

    if exp_name != 'toy':
        dataset_ID = HighDimSCMRealWorldDataset(df_ID, y_dtype=y_dtype, y_emb_option=y_emb_option, use_fc=ag.use_fc, subtask=ag.subtask)
        if ag.deploy or ag.verbose: dataset_OOD = HighDimSCMRealWorldDataset(df_OOD, y_dtype=y_dtype, y_emb_option=y_emb_option if y_emb_option == 'eq' else 'default', use_fc=ag.use_fc, subtask=ag.subtask, is_ood=exp_name=='ood_nlp', env_idx=env_idx)
        else: dataset_OOD = None

    else:
        dataset_ID = HighDimSCMSyntheticDataset(df_ID)
        dataset_OOD = HighDimSCMSyntheticDataset(df_OOD)

    return dataset_ID, dataset_OOD, y_dtype
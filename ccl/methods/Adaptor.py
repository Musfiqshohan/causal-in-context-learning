import copy
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from arch import mlp
from methods import SemVar
from distr import edic

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import kl_divergence


from ccl.utils.utils import EarlyStopping

class Adaptor():
    def __init__(self, ag, discr, gen):
        self.ag = ag
        self.discr = discr
        self.gen = gen

        self.original_discr_params = copy.deepcopy(discr.state_dict())
        self.original_gen_params = copy.deepcopy(gen.state_dict())

        for param in self.discr.parameters():
            param.requires_grad = False
        for param in self.gen.parameters():
            param.requires_grad = False

        self.sample_idx, self.xs, self.xs_hat, self.index_t, self.index_e = [], [], [], [], []
        self.cs, self.ss, self.c_std, self.s_std = [], [], [], []
    
    def set_optimizer(self):
        params = [
            {'params': self.discr.parameters(), 'lr': self.ag.lr_discr, 'weight_decay': self.ag.wl2_discr},
            # {'params': self.gen.parameters(), 'lr': self.ag.lr_gen, 'weight_decay': self.ag.wl2_gen},
        ]

        self.optim = getattr(torch.optim, self.ag.optim)(params)

    def set_learning_params(self):
        discr_params_list = ['f_xt2prev', 'f_xe2prev', 'f_prev2c', 'f_prevs2s']
                            #  'nn_std_c', 'f_std_c', 'nn_std_s', 'f_std_s']
        # gen_params_list = ['f_vparas2x', 'f_vparas2e']

        for name, param in self.discr.named_parameters():
            if any(key in name for key in discr_params_list):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # for name, param in self.gen.named_parameters():
        #     if any(key in name for key in gen_params_list):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def load_original_params(self):
        self.discr.load_state_dict(self.original_discr_params)
        self.gen.load_state_dict(self.original_gen_params)

    def get_frame(self, discr, gen, dc_vars, device = None, discr_src = None):
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
            # print("Conditional prior std")
            # print(prior_std)
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
    
    def adapt_methods(self):
        lossobj = self.frame.get_lossfn(self.ag.n_mc_q, self.ag.reduction, 'adapt', recon=self.ag.recon)

        if self.ag.wreg_cs > -1.0:
            if self.ag.reg_cs == "cossim": loss_reg_cs = self.frame.reg_cossim_cs()
            elif self.ag.reg_cs == "mmd": loss_reg_cs = self.frame.reg_mmd_cs()
        else: loss_reg_cs = None

        def lossfn(*x_y_maybext_niter):
            log_x_recon_loss, log_e_recon_loss, expc_val_c, expc_val_s = lossobj(*x_y_maybext_niter[:4])
            if loss_reg_cs is not None: reg_cs_loss = loss_reg_cs(*x_y_maybext_niter[:4])

            adapt_loss = torch.mean(-1*self.ag.wada_x*log_x_recon_loss + -1*self.ag.wada_e*log_e_recon_loss + -1*self.ag.lam_c*expc_val_c + -1*self.ag.lam_s*expc_val_s)
            if self.ag.wreg_cs != -1: adapt_loss += self.ag.wreg_cs * reg_cs_loss.mean()
            return adapt_loss
        
        return lossfn

    def get_dataframe(self):
        def safe_concat(lst): return np.vstack(lst) if len(lst) > 0 else np.array([])
        sample_idx = np.concatenate(self.sample_idx, axis=0)
        label_t = np.concatenate(self.index_t, axis=0)
        label_e = np.concatenate(self.index_e, axis=0)
        
        xs = safe_concat(self.xs)
        c_hat = safe_concat(self.cs)
        s_hat = safe_concat(self.ss)
        c_std = safe_concat(self.c_std)
        s_std = safe_concat(self.s_std)
        xs_hat = safe_concat(self.xs_hat)
        

        df = pd.DataFrame({
            'sample_idx': sample_idx.flatten(),
            'X': xs.tolist(),
            'C_hat': c_hat.tolist(),
            'S_hat': s_hat.tolist(),
            'C_std': c_std.tolist(),
            'S_std': s_std.tolist(),
            'X_hat': xs_hat.tolist(),
            'Index_T': label_t.tolist(),
            'Index_E': label_e.tolist()
        })
        return df
    
    def adapt_latent(self, dataloader, device):
        self.load_original_params()
        self.discr.eval(); self.gen.eval()
        self.frame = self.get_frame(self.discr, self.gen, vars(self.ag), device)

        mse_loss = nn.MSELoss(reduction='none').to(device)
        self.total_x_recon_loss, self.total_e_recon_loss, self.total_y_recon_loss = 0, 0, 0
        with tqdm(total=len(dataloader), leave=True, desc="Extract Latent Variables") as outer_progress:
            for i_bat, data_bat in enumerate(dataloader, start=1):
                xs = data_bat.X.to(device, dtype=torch.float32)
                ts = data_bat.T.to(device, dtype=torch.float32)
                envs = data_bat.E.to(device, dtype=torch.float32)

                self.sample_idx.append(data_bat.index.cpu().numpy())
                self.xs.append(data_bat.X.cpu().numpy())
                self.index_t.append(data_bat.label_T.cpu().numpy())
                self.index_e.append(data_bat.label_E.cpu().numpy())

                # Initialize learnable parameters
                with torch.no_grad():
                    c_init = self.frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                    s_init = self.frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_init})['s']
                    c_std = self.frame.qt_c1x.std({'x':xs, 't':ts, 'e':envs})['c']
                    s_std = self.frame.qt_s1x.std({'x':xs, 't':ts, 'e':envs, 'c':c_init})['s']

                    print(f"Latent Variable Difference (Before Adaptation):")
                    normal1 = dist.Normal(c_init, c_std)
                    normal2 = dist.Normal(s_init, s_std)
                    kl_div = kl_divergence(normal1, normal2).mean().item()
                    cossim = F.cosine_similarity(c_init, s_init, dim=1).mean().item()
                    print("KL Divergence:", kl_div)
                
                    print("Avg Cosine similarity: ", cossim)
                    wandb.log({
                        "i_bat": i_bat,
                        "CS_KL_Div_Before": kl_div,
                        "CS_Cosine_Sim_Before": cossim
                    })

                # Create learnable parameters initialized with c_init and s_init
                c_hat = torch.nn.Parameter(c_init.clone(), requires_grad=True)
                s_hat = torch.nn.Parameter(s_init.clone(), requires_grad=True)
                
                params = [
                        {'params': c_hat, 'lr': self.ag.lr_l_c, 'weight_decay': self.ag.wd_l_c},
                        {'params': s_hat, 'lr': self.ag.lr_l_s, 'weight_decay': self.ag.wd_l_s},
                    ]

                opt = torch.optim.Adam(params)
                early_stopping = EarlyStopping(patience=self.ag.patience, verbose=False)
                total_loss = 0
                with tqdm(total=self.ag.n_epk, leave=False, desc=f"Batch {i_bat}/{len(dataloader)}: Adapt Latent Variables") as inner_progress:
                    for e in range(self.ag.n_epk):
                        xs = data_bat.X.to(device, dtype=torch.float32)
                        ys = data_bat.Y.to(device, dtype=torch.float32)
                        ts = data_bat.T.to(device, dtype=torch.float32)
                        envs = data_bat.E.to(device, dtype=torch.float32)

                        # Adapt latent variables to the new domain dataset
                        x_hat = self.frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                        e_hat = self.frame.p_e1s.mean({'s':s_hat})['e']

                        x_recon_loss = mse_loss(x_hat, xs).mean()
                        e_recon_loss = mse_loss(e_hat, envs).mean()

                        cs_cos_sim = F.cosine_similarity(c_hat, s_hat, dim=1).mean()

                        opt.zero_grad()
                        # adapt_loss = self.ag.wada_x * x_recon_loss + self.ag.wada_e * e_recon_loss + \
                        #     self.ag.lam_c * torch.norm(torch.abs(c_init-c_hat), p=2) + self.ag.lam_s * torch.norm(torch.abs(s_init-s_hat), p=2)
                        # Implementing L2 regularization terms: ||c_hat||_2^2 and ||s_hat||_2^2
                        adapt_loss = self.ag.wada_x * x_recon_loss + \
                            self.ag.lam_c * torch.norm(c_hat, p=2)**2 + self.ag.lam_s * torch.norm(s_hat, p=2)**2 + \
                            self.ag.wreg_cs * cs_cos_sim
                        adapt_loss.backward()
                        opt.step()
                        
                        # Calculate gradient norms
                        c_grad_norm = torch.norm(c_hat.grad).item()
                        s_grad_norm = torch.norm(s_hat.grad).item()
                        
                        # Log gradient norms
                        wandb.log({
                            "C_Gradient_Norm": c_grad_norm,
                            "S_Gradient_Norm": s_grad_norm
                        })
                        
                        wandb.log({"epoch": e, "Adapt_Loss": adapt_loss.detach().cpu()})
                        total_loss += adapt_loss.detach().cpu()

                        with torch.no_grad():
                            x_hat = self.frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                            e_hat = self.frame.p_e1s.mean({'s':s_hat})['e']

                            x_recon_loss = mse_loss(x_hat, xs).mean()
                            e_recon_loss = mse_loss(e_hat, envs).mean()

                            if e == 0:
                                init_x_recon_loss = x_recon_loss.clone().detach()
                                init_e_recon_loss = e_recon_loss.clone().detach()
                            else:
                                x_recon_loss /= init_x_recon_loss
                                e_recon_loss /= init_e_recon_loss
                            total_recon_loss = x_recon_loss + e_recon_loss

                        # flag_early_stop, flag_update_best_model = early_stopping(total_recon_loss)
                        # if flag_update_best_model or e == 0:
                            best_latent_c = c_hat.detach().clone()
                            best_latent_s = s_hat.detach().clone()

                        # if flag_early_stop:
                        #     break

                        inner_progress.update(1)
                
                print(f"Latent Variable Difference (After Adaptation):")
                normal1 = dist.Normal(best_latent_c, c_std)
                normal2 = dist.Normal(best_latent_s, s_std)
                kl_div = kl_divergence(normal1, normal2).mean().item()
                cossim = F.cosine_similarity(best_latent_c, best_latent_s, dim=1).mean().item()
                # Diff between best_latent_c and c_init
                diff_c = torch.norm(torch.abs(c_init-best_latent_c), p=2).mean().item()
                # Diff between best_latent_s and s_init
                diff_s = torch.norm(torch.abs(s_init-best_latent_s), p=2).mean().item()
                print("KL Divergence:", kl_div)
                print("Avg Cosine similarity: ", cossim)
                print("Diff between best_latent_c and c_init:", diff_c)
                print("Diff between best_latent_s and s_init:", diff_s)
                wandb.log({
                    "i_bat": i_bat,
                    "CS_KL_Div_After": kl_div,
                    "CS_Cosine_Sim_After": cossim,
                    "Diff_C_Before_After": diff_c,
                    "Diff_S_Before_After": diff_s
                })

                print(f"{i_bat}/{len(dataloader)}: Adapt_Avg_Loss: {total_loss/self.ag.n_epk}")
                wandb.log({"Adapt_Avg_Loss": total_loss/self.ag.n_epk})

                self.cs.append(best_latent_c.cpu().numpy())
                self.ss.append(best_latent_s.cpu().numpy())
                self.c_std.append(c_std.cpu().numpy())
                self.s_std.append(s_std.cpu().numpy())

                with torch.no_grad():
                    x_hat = self.frame.p_x1cs.mean({'c':best_latent_c.to(device), 's':best_latent_s.to(device)})['x']
                    e_hat = self.frame.p_e1s.mean({'s':best_latent_s.to(device)})['e']
                    y_hat = self.frame.p_y1c.mean({'c':best_latent_c.to(device)})['y']

                    self.total_x_recon_loss += mse_loss(x_hat, xs).mean(dim=1).sum()
                    self.total_e_recon_loss += mse_loss(e_hat, envs).mean(dim=1).sum()
                    self.total_y_recon_loss += mse_loss(y_hat, ys).mean(dim=1).sum()

                    self.xs_hat.append(x_hat.cpu().numpy())

                outer_progress.update(1)

            self.total_x_recon_loss /= len(dataloader.dataset)
            self.total_e_recon_loss /= len(dataloader.dataset)
            self.total_y_recon_loss /= len(dataloader.dataset)

            wandb.log({
                "Adapt_Avg_X_Recon_Loss": self.total_x_recon_loss.detach().cpu(),
                "Adapt_Avg_E_Recon_Loss": self.total_e_recon_loss.detach().cpu(),
                "Adapt_Avg_Y_Recon_Loss": self.total_y_recon_loss.detach().cpu()
            })
            

    def adapt_para(self, dataloader, device):
        mse_loss = nn.MSELoss(reduction='none').to(device)
        self.total_x_recon_loss, self.total_y_recon_loss, self.total_e_recon_loss = 0, 0, 0
        with tqdm(total=len(dataloader), leave=True, desc="Batch") as outer_progress:
            for i_bat, data_bat in enumerate(dataloader, start=1):
                self.sample_idx.append(data_bat['index'].cpu().numpy())
                self.xs.append(data_bat['X'].cpu().numpy())
                self.index_t.append(data_bat['label_T'].cpu().numpy())
                self.index_e.append(data_bat['label_E'].cpu().numpy())

                # print(f"{i_bat}/{len(dataloader)}: Load original params and set learning params")
                self.load_original_params()
                self.set_learning_params()
                self.set_optimizer()

                self.frame = self.get_frame(self.discr, self.gen, vars(self.ag), device)
                self.lossfn = self.adapt_methods()

                early_stopping = EarlyStopping(patience=self.ag.patience, verbose=False)
                self.gen.eval()
                with tqdm(total=self.ag.n_epk, leave=False, desc=f"Batch {i_bat}/{len(dataloader)}: Adapt OOD") as inner_progress:
                    for e in range(self.ag.n_epk):
                        self.discr.train()

                        xs = data_bat['X'].to(device, dtype=torch.float32)
                        ys = data_bat['Y'].to(device, dtype=torch.float32)
                        ts = data_bat['T'].to(device, dtype=torch.float32)
                        envs = data_bat['E'].to(device, dtype=torch.float32)

                        data_args = (xs, ts, envs, ys, 'adapt')

                        self.optim.zero_grad()
                        losses = self.lossfn(*data_args, 0)
                        losses.backward()
                        self.optim.step()

                        losses = losses.detach()

                        wandb.log({"epoch": e, "Adapt_Loss": losses.detach().cpu()})

                        with torch.no_grad():
                            self.discr.eval() # ; self.gen.eval()
                            c_hat = self.frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                            s_hat = self.frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']
                            x_hat = self.frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                            e_hat = self.frame.p_e1s.mean({'s':s_hat})['e']

                            x_recon_loss = mse_loss(x_hat, xs).mean()
                            e_recon_loss = mse_loss(e_hat, envs).mean()

                            if e == 0:
                                init_x_recon_loss = x_recon_loss.clone().detach()
                                init_e_recon_loss = e_recon_loss.clone().detach()
                            else:
                                x_recon_loss /= init_x_recon_loss
                                e_recon_loss /= init_e_recon_loss
                            total_recon_loss = x_recon_loss + e_recon_loss

                        if e % 5 == 0:
                            flag_early_stop, flag_update_best_model = early_stopping(total_recon_loss)
                            if flag_update_best_model or e == 0:
                                # print(f"\nUpdating best model at epoch {e}\n")
                                best_model_discr = self.discr.state_dict()
                                # best_model_gen = self.gen.state_dict()

                            if flag_early_stop:
                                break

                        inner_progress.update(1)
                    
                    self.discr.load_state_dict(best_model_discr)
                    # self.gen.load_state_dict(best_model_gen)
                    self.discr.eval() # ; self.gen.eval()
                    self.frame = self.get_frame(self.discr, self.gen, vars(self.ag), device)

                    with torch.no_grad():
                        c_hat = self.frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                        s_hat = self.frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']
                        c_std = self.frame.qt_c1x.std({'x':xs, 't':ts, 'e':envs})['c']
                        s_std = self.frame.qt_s1x.std({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']

                        x_hat = self.frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                        y_hat = self.frame.p_y1c.mean({'c':c_hat})['y']
                        e_hat = self.frame.p_e1s.mean({'s':s_hat})['e']

                        self.xs_hat.append(x_hat.cpu().numpy())
                        self.cs.append(c_hat.cpu().numpy())
                        self.ss.append(s_hat.cpu().numpy())
                        self.c_std.append(c_std.cpu().numpy())
                        self.s_std.append(s_std.cpu().numpy())

                        self.total_x_recon_loss += mse_loss(x_hat, xs).mean(dim=1).sum()
                        self.total_y_recon_loss += mse_loss(y_hat, ys).mean(dim=1).sum()
                        self.total_e_recon_loss += mse_loss(e_hat, envs).mean(dim=1).sum()
                
                outer_progress.update(1)
            
            self.total_x_recon_loss /= len(dataloader.dataset)
            self.total_e_recon_loss /= len(dataloader.dataset)
            self.total_y_recon_loss /= len(dataloader.dataset)

            total_x_recon_loss = self.total_x_recon_loss.detach().cpu()
            total_e_recon_loss = self.total_e_recon_loss.detach().cpu()
            total_y_recon_loss = self.total_y_recon_loss.detach().cpu()

            if self.ag.exp_name == 'llm_ret': val_x_recon = 0.0004214328364469111; val_e_recon = 0.0000000092037177879
            else: val_x_recon = 0.0005243898485787213; val_e_recon = 0.0000000089571550177

            wandb.log({
                "Adapt_Avg_XE_Recon_Loss": total_x_recon_loss + total_e_recon_loss,
                "Adapt_Avg_X_Recon_Loss": total_x_recon_loss,
                "Diff_X_Recon_Loss": np.abs(total_x_recon_loss - val_x_recon),
                "Diff_E_Recon_Loss": np.abs(total_e_recon_loss - val_e_recon),
                "Adapt_Avg_E_Recon_Loss": total_e_recon_loss,
                "Adapt_Avg_Y_Recon_Loss": total_y_recon_loss
            })

            print(f"Adapt_Avg_XE_Recon_Loss: {total_x_recon_loss + total_e_recon_loss}")
            print(f"Adapt_Avg_X_Recon_Loss: {total_x_recon_loss}")
            print(f"Diff_X_Recon_Loss: {np.abs(total_x_recon_loss - val_x_recon)}")
            print(f"Diff_E_Recon_Loss: {np.abs(total_e_recon_loss - val_e_recon)}")
            print(f"Adapt_Avg_E_Recon_Loss: {total_e_recon_loss}")
            print(f"Adapt_Avg_Y_Recon_Loss: {total_y_recon_loss}")

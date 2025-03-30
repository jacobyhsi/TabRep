import torch
import numpy as np
import pandas as pd
import os
import json
import time

from tabrep_ddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from tabrep_ddpm.models.modules import MLPDiffusion

import src
from utils_train import make_dataset

def bits_needed(categories):
    return 2 * np.ones_like(categories)

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]
    syn_cat = torch.tensor(syn_cat, dtype=torch.uint8)

    if num_inverse is not None:
        syn_num = num_inverse(syn_num).astype(np.float32)
    if cat_inverse is not None:
        syn_cat = cat_inverse(syn_cat)

    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    model_save_path,
    sample_save_path,
    real_data_path,
    batch_size = 2000,
    num_samples = 0,
    task_type = 'binclass',
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    change_val = False,
    ddim = False,
    steps = 1000,
):
    T = src.Transformations(**T_dict)

    D = make_dataset(
        real_data_path,
        T,
        task_type = task_type,
        change_val = False,
    )

    K = np.array(D.get_category_sizes('train'))
    num_numerical_features = D.X_num['train'].shape[1] if D.X_num is not None else 0
    num_bits_per_cat_feature = bits_needed(K) if len(K) > 0 else np.array([0])
    d_in = np.sum(num_bits_per_cat_feature) + num_numerical_features
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=D.get_category_sizes('train')
    )
    
    # for model_iter in range(2000, 100001, 2000): #
        # model_path =f'{model_save_path}/model_{model_iter}.pt' #
    model_path =f'{model_save_path}/model.pt' #
    print("Sampling", model_path)

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=steps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device,
        num_bits_per_cat_feature=num_bits_per_cat_feature
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps = steps)
    
    syn_data = x_gen

    num_inverse = None
    cat_inverse = None
    if num_numerical_features > 0:
        num_inverse = D.num_transform.inverse_transform
    if num_bits_per_cat_feature.sum() > 0:
        cat_inverse = D.cat_transform.inverse_transform
    info_path = f'{real_data_path}/info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)
    
    # base_path, base_filename = os.path.split(sample_save_path) #
    # filename, ext = os.path.splitext(base_filename) #
    # new_filename = f"{filename}_500{ext}" #
    # save_path = os.path.join(base_path, new_filename) #
    
    save_path = sample_save_path #

    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))

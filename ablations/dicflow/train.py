import os
import time
import torch
import numpy as np
import pandas as pd

from copy import deepcopy

import src
from utils_train import make_dataset
from ablations.dicflow.models.modules import MLPDiffusion, Model
from ablations.dicflow.models.flow_matching import ConditionalFlowMatcher

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

class Trainer:
    def __init__(self, cfm_model, train_iter, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1')):
        self.model = cfm_model
        self.ema_model = deepcopy(self.model.flow_net)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'loss'])
        self.model_save_path = model_save_path

        columns = list(np.arange(5)*200)
        columns[0] = 1
        columns = ['step'] + columns
 

        self.log_every = 50
        self.print_every = 1
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)

        self.optimizer.zero_grad()

        loss = self.model(x)

        loss.backward()
        self.optimizer.step()

        return loss


    def run_loop(self):
        step = 0
        curr_loss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print('Steps: ', self.steps)
        while step < self.steps:
            start_time = time.time()
            x = next(self.train_iter)[0]
            
            batch_loss = self._run_step(x)

            # self._anneal_lr(step)

            curr_count += len(x)
            curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                loss = np.around(curr_loss / curr_count, 4)
                if np.isnan(loss):
                    print('Finding Nan')
                    break
                
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, loss]

                np.set_printoptions(suppress=True)
          
                curr_count = 0
                curr_loss = 0.0

                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.flow_net.state_dict(), os.path.join(self.model_save_path, 'model.pt'))
  
                if (step + 1) % 10000 == 0:
                    torch.save(self.model.flow_net.state_dict(), os.path.join(self.model_save_path, f'model_{step+1}.pt'))

            # update_ema(self.ema_model.parameters(), self.model.flow_net.parameters())

            step += 1
            # end_time = time.time()
            # print('Time: ', end_time - start_time)

def train(
    model_save_path,
    real_data_path,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    task_type = 'binclass',
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False
):
    real_data_path = os.path.normpath(real_data_path)

    T = src.Transformations(**T_dict)
    
    dataset = make_dataset(
        real_data_path,
        T,
        task_type = task_type,
        change_val = False,
    )

    K = np.array(dataset.get_category_sizes('train'))
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    emb_dim = 1
    cat_len = 0
    for item in K:
        cat_len += 1 if item <=2 else emb_dim
        
    d_in = num_numerical_features + cat_len
    model_params['d_in'] = d_in
    print('d_in: ', d_in)

    train_loader = src.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    flow_net = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    cfm = ConditionalFlowMatcher(sigma=0.0, pred_x1=False)
    model = Model(
        flow_net,
        cfm,
        num_numerical_features,
        K,
        emb_dim
    )
    model.to(device)
    model.train()

    trainer = Trainer(
        model,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device
    )
    trainer.run_loop()

    torch.save(model.flow_net.state_dict(), os.path.join(model_save_path, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))

    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)



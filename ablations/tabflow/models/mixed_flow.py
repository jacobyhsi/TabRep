import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ablations.tabflow.models.discrete_flow import DiscreteFlowMatcher
from ablations.tabflow.models.continuous_flow import ContinuouslFlowMatcher

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1).float()
    return x_onehot

class ContinuousDiscreteFlow(nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            flow_net,
            device=torch.device('cpu')
        ):
        super().__init__()

        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features
        self.flow_net = flow_net
        self.device = device

        self.continuous_flow = ContinuouslFlowMatcher()
        self.discrete_flow = DiscreteFlowMatcher(num_classes)

    def mixed_loss(self, x):
        
        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]

        t = None
        if x_num.shape[1] > 0:
            x_num_1 = x_num
            x_num_0 = torch.randn_like(x_num_1)
            t, x_num_t, u_num_t = self.continuous_flow.sample_location_and_conditional_flow(x_num_0, x_num_1)
            # vt = self.flow_net(xt, t)
            # loss = (vt - ut) ** 2
        if x_cat.shape[1] > 0:
            x_cat_1 = x_cat
            x_cat_t = self.discrete_flow.sample_location_and_conditional_flow(x_cat_1, t)
            x_cat_t = index_to_log_onehot(x_cat_t, self.num_classes)

        x_in = torch.cat([x_num_t, x_cat_t], dim=1)
        model_out = self.flow_net(x_in, t)
        
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_continuous = torch.zeros((1,)).float()
        loss_discrete = torch.zeros((1,)).float()
        
        if x_num.shape[1] > 0:
            loss_continuous = self.continuous_flow_matching_loss(model_out_num, u_num_t)
        
        if x_cat.shape[1] > 0:
            loss_discrete = self.discrete_flow_matching_loss(model_out_cat, x_cat_1)

        return loss_continuous + loss_discrete

    def continuous_flow_matching_loss(self, vt, ut):
        return ((vt - ut) ** 2).mean(-1).mean()
    
    def discrete_flow_matching_loss(self, logits, x1):
        loss = 0
        for i, S in enumerate(self.num_classes):
            star_index = sum(self.num_classes[:i])
            end_index = sum(self.num_classes[:i+1])
            loss += F.cross_entropy(logits[:, star_index:end_index], x1[:, i].long(), reduction='mean')
        return loss

    @torch.no_grad()
    def mixed_sample(self, num_samples, N=1000, device='cuda:0', use_tqdm=False):
        b = num_samples
        tq = tqdm if use_tqdm else lambda x: x

        z_num = torch.randn((b, self.num_numerical_features), device=device)
        z_cat = torch.cat([
                torch.randint(0, K, (b, 1), device=device) 
                for K in self.num_classes], dim=1)
        
        dt = 1. / N
        for i in tq(range(N)):
            t = (i / N) * torch.ones((num_samples,)).to(device)
            z_cat_onehot = index_to_log_onehot(z_cat, self.num_classes)
            z = torch.cat([z_num, z_cat_onehot], dim=1)

            model_out = self.flow_net(z, t)
            num_model_out = model_out[:, :self.num_numerical_features]
            cat_model_out = model_out[:, self.num_numerical_features:]
            z_num = self.contunuous_flow_sampling_step(z_num, num_model_out, dt)
            z_cat = self.discrete_flow_sampling_step(z_cat, cat_model_out, i/N, dt)
        
        return torch.cat([z_num, z_cat], dim=1).cpu().numpy()


    def contunuous_flow_sampling_step(self, z, vt, dt):
        return z.detach().clone() + vt * dt

    def discrete_flow_sampling_step(self, z_cat, model_out, t, dt):
        noise = 1
        
        z_out = []
        for i, S in enumerate(self.num_classes):
            star_index = sum(self.num_classes[:i])
            end_index = sum(self.num_classes[:i+1])
            logits = model_out[:, star_index:end_index]
            
            xt = z_cat[:, i]    # (B,)
            x1_probs = F.softmax(logits, dim=-1)    # (B, S)
            x1_probs_at_xt = torch.gather(x1_probs, -1, xt[:, None]) # (B, 1)
            
            # Don't add noise on the final step
            if t + dt < 1.0:
                N = noise
            else:
                N = 0

            # Calculate the off-diagonal step probabilities
            step_probs = (
                dt * ((1 + N + N * (S - 1) * t ) / (1-t)) * x1_probs + 
                dt * N * x1_probs_at_xt
            ).clamp(max=1.0) # (B, S)

            # Calculate the on-diagnoal step probabilities
            # 1) Zero out the diagonal entries
            step_probs.scatter_(-1, xt[:, None], 0.0)
            # 2) Calculate the diagonal entries such that the probability row sums to 1
            step_probs.scatter_(-1, xt[:, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0))

            xt = Categorical(step_probs).sample() # (B,) 
            z_out.append(xt)
        
        return torch.stack(z_out, dim=1)

            

import os
import argparse
import time
import pdb
from contextlib import nullcontext

import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(NoMaD, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        # Runtime compute policy only; parameters and state_dict remain FP32.
        self.use_bf16 = False
    
    def forward(self, func_name, **kwargs):
        # Enter autocast inside forward so each DataParallel worker uses BF16,
        # rather than inheriting another thread's default autocast dtype.
        # getattr also supports old pickled NoMaD objects without this attribute.
        use_bf16 = getattr(self, "use_bf16", False)
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with precision:
            if func_name == "vision_encoder":
                output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
            elif func_name == "noise_pred_net":
                output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
            elif func_name == "dist_pred_net":
                output = self.dist_pred_net(kwargs["obsgoal_cond"])
            else:
                raise NotImplementedError
        # Keep losses, diffusion scheduler arithmetic and NumPy visualization in
        # FP32 while retaining gradients through the cast for training.
        return output.float() if use_bf16 else output


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output




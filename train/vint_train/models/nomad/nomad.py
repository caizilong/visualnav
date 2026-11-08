import os
import argparse
import time
import pdb

import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net,
                       goal_relation_head=None):
        super(NoMaD, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        if goal_relation_head is not None:
            self.goal_relation_head = goal_relation_head
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            extra = {"return_r1": True} if kwargs.get("return_r1", False) else {}
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"],
                                         input_goal_mask=kwargs["input_goal_mask"], **extra)
        elif func_name == "spatial_tokens":
            if not hasattr(self.vision_encoder, "extract_spatial_tokens"):
                raise ValueError("Spatial outputs require an R1 NoMaD_ViNT model")
            output = self.vision_encoder.extract_spatial_tokens(kwargs["obs_img"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        elif func_name == "goal_relation_head":
            if not hasattr(self, "goal_relation_head"):
                raise ValueError("This checkpoint has no R3 goal relation head")
            output = self.goal_relation_head(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


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




from typing import Union
import numpy as np
import torch
import math

def setup_optimizers(
        network:torch.nn.Module, opt_type,
        lr=1e-3, regularization=0.0005
):
    if regularization>0.0:
        optim_params = []
        for k, v in network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        if opt_type == 'Adam':
            # optimizer = torch.optim.Adam(optim_params, lr=lr)
            optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=regularization)
        elif opt_type == 'AdamW':
            optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=regularization)
        else:
            raise NotImplementedError
        # optimizer.add_param_group({'params': optim_params, 'weight_decay': regularization})

    else:
        if opt_type == 'Adam':
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        elif opt_type == 'AdamW':
            optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
        else:
            raise NotImplementedError

    return optimizer

class Last_Saver(object):
    def __init__(self, path, meta):
        self.path = path
        self.meta = meta

    def save(self, model):
        checkpoint = {
            'meta': self.meta,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, self.path)

class Best_Saver(object):
    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self.best_score = math.inf

    def save(self, model, score, epoch):
        if score<self.best_score:
            self.best_score = score

            checkpoint = {
                'epoch': epoch,
                'meta': self.meta,
                'state_dict': model.state_dict()
            }
            torch.save(checkpoint, self.path)

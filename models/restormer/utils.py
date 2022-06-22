from typing import Union
import torch

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

class Last_Saver:
    def __init__(self, path, meta):
        self.path = path
        self.meta = meta

    def save(self, model):
        checkpoint = {
            'meta': self.meta,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, self.path)

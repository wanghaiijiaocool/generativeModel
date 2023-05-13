#encoding=utf-8

import torch.nn as nn

def init_weights(m,):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
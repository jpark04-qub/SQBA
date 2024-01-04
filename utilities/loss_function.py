import torch
import torch.nn as nn


def list(type='cross entropy'):

    if type == 'cross entropy':
        lossF = nn.CrossEntropyLoss()

    return lossF

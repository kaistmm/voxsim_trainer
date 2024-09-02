#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def score2class2(score_list, thresholds=None):

    if thresholds is None:
        thresholds = [1/2]

    assert len(thresholds) == 1

    newlist = []

    for score in score_list:
        if score < thresholds[0]:
            newlist.append(1)
        else:
            newlist.append(2)
    
    return np.array(newlist)

def score2class4(score_list, thresholds=None):

    if thresholds is None:
        thresholds = [1/6, 1/2, 5/6]

    assert len(thresholds) == 3
    
    newlist = []

    for score in score_list:
        if score < 0 or score > 1:
            newlist.append(-1)
        elif score < thresholds[0]:
            newlist.append(1)
        elif score < thresholds[1]:
            newlist.append(2)
        elif score < thresholds[2]:
            newlist.append(3)
        else:
            newlist.append(4)
    
    return np.array(newlist)

def score2class6(score_list, thresholds=None):

    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    assert len(thresholds) == 5

    newlist = []

    for score in score_list:
        if score < 0 or score > 1:
            newlist.append(-1)
        elif score < thresholds[0]:
            newlist.append(1)
        elif score < thresholds[1]:
            newlist.append(2)
        elif score < thresholds[2]:
            newlist.append(3)
        elif score < thresholds[3]:
            newlist.append(4)
        elif score < thresholds[4]:
            newlist.append(5)
        else:
            newlist.append(6)
    
    return np.array(newlist)
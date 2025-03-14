import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch.nn as nn

def FocalLoss(predict, target, gamma=2, alpha=0.25, coef=.03):
    loss_config = {}
    loss_config['alpha'] = alpha
    loss_config['gamma'] = gamma
    ori_loss = F.binary_cross_entropy_with_logits(predict.view(-1), target.float().view(-1), reduction='none')
    pred_sigmoid = predict.view(-1).sigmoid()
    target = target.float().view(-1)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (loss_config['alpha'] * target + (1 - loss_config['alpha']) * (1 - target)) * pt.pow(loss_config['gamma'])
    loss = torch.mean(ori_loss * focal_weight)       
    return loss 

def getLossFunction(name = 'BCE'):
    if name == 'BCE':
        return BCEWithLogitsLoss()
    elif name == 'FL':
        return FocalLoss
    else:
        raise NotImplementedError
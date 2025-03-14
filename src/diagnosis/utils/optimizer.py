import transformers
import torch

def model_optimizer(model,opt):
    param_optimizer = None
    if hasattr(model,'longformer_layer'):
        bert_params = set(model.longformer_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.longformer_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.longformer_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif hasattr(model,'ernie_layer'):
        bert_params = set(model.ernie_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.ernie_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.ernie_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif hasattr(model,'bert_layer'):
        bert_params = set(model.bert_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.bert_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.bert_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    elif hasattr(model,'base_layer'):
        bert_params = set(model.base_layer.word_embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.base_layer.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-2},
            {'params': [p for n, p in model.base_layer.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.other_lr,
             'weight_decay': 0}
        ]
    if param_optimizer is not None:
        optimizer = transformers.AdamW(param_optimizer, lr=opt.other_lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr = opt.other_lr,weight_decay=0.01)
    return optimizer
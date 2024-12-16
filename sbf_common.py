import torch
import torch.optim
from torch.utils.data import Dataset
from transformers import (PreTrainedTokenizer,
                          EvalPrediction
                          )
from datasets import load_dataset
import numpy as np

MAX_LENGTH = 256
HF_DS = 'allenai/social_bias_frames'

tok_kwargs = {
    'padding': 'max_length',
    'max_length': MAX_LENGTH,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True,
    'add_special_tokens': True,
}

def cls_metrics(ep: EvalPrediction):
    logits = torch.tensor(ep.predictions[1]).squeeze(dim=1)
    preds = logits.softmax(0).round().numpy().astype(np.uint8)
    labels = np.rint(ep.label_ids).astype(np.uint8)
    pos_idx = preds == 1
    neg_idx = preds == 0
    tp = np.count_nonzero(preds[pos_idx] == labels[pos_idx]) * 1.0
    fp = np.count_nonzero(preds[pos_idx] != labels[pos_idx]) * 1.0
    fn = np.count_nonzero(preds[neg_idx] != labels[neg_idx]) * 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 

def cls_loss(outputs, labels, *, num_items_in_batch):
    if isinstance(outputs, dict) and "loss" not in outputs:
        if labels is not None:
            logits = outputs["logits"].squeeze(dim=1)
            bce = torch.nn.BCEWithLogitsLoss()
            return bce(logits, labels)
        
def get_dataset(split: str, feature: str, tokenizer: PreTrainedTokenizer):
    ds = load_dataset('anoaky/sbf-collated', feature, split=split)
    def tokenize(x):
        post = tokenizer(x['post'], **tok_kwargs)
        label = torch.tensor(x[feature]).round().item()
        return {
            'input_ids': post.input_ids.view(-1),
            'attention_mask': post.attention_mask.view(-1),
            'labels': label,
        }
    ds = ds.map(tokenize)
    return ds
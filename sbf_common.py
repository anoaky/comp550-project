import torch
import torch.optim
from torch.utils.data import Dataset
from transformers import (PreTrainedTokenizer,
                          EvalPrediction
                          )
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

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
    logits, labels = ep
    preds = logits.softmax(-1).argmax(-1).numpy().astype(np.uint8)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 

def cls_loss(outputs, labels, *, num_items_in_batch):
    if isinstance(outputs, dict) and "loss" not in outputs:
        if labels is not None:
            logits = outputs["logits"]
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(logits, labels)
        
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
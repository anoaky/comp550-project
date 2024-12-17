import wandb
import torch
import torch.optim
import torch.nn.functional as F
from transformers import (PreTrainedTokenizer,
                          EvalPrediction,
                          Trainer,
                          TrainingArguments,
                          )
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
from argparse import ArgumentParser

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

def unpack_ep(ep: EvalPrediction):
    if len(ep.predictions) == 2:
        (logits, _), labels = ep
    else:
        logits, labels = ep
    return logits, labels

def cls_metrics(ep: EvalPrediction):
    logits, labels = unpack_ep(ep)
    preds = torch.tensor(logits).softmax(-1).argmax(-1).numpy().astype(np.uint8)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    (tn, fp, fn, tp) = confusion_matrix(labels, preds).ravel()
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    } 

def cls_loss(outputs, labels, *, num_items_in_batch):
    if isinstance(outputs, dict) and "loss" not in outputs:
        if labels is not None:
            logits = outputs["logits"]
            return F.cross_entropy(logits, labels)
        
def get_dataset(split: str, label: str, tokenizer: PreTrainedTokenizer):
    ds = load_dataset('anoaky/sbf-collated', label, split=split)
    def tokenize(x):
        post = tokenizer(x['post'], **tok_kwargs)
        y = torch.tensor(x[label]).round().long()
        return {
            'input_ids': post.input_ids.view(-1),
            'attention_mask': post.attention_mask.view(-1),
            'labels': y,
        }
    ds = ds.map(tokenize)
    return ds

def wandb_setup(args):
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_PROJECT'] = 'COMP550'
    proj = 'COMP550'
    wandb.init(
        project=proj,
        name=args.run,
        tags=args.tags,
        group=args.label,
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--label', required=True, choices=['offensive', 'sex', 'intent', 'speakerMinority'])
    parser.add_argument('-o', '--output_dir', default='/outputs/model', type=str)
    parser.add_argument('-e', '--epochs', default=2, type=int)
    args = parser.parse_args()
    return args

def train(model, tokenizer, hub_model_id, args):
    wandb_token = os.environ['WANDB_TOKEN']
    assert len(wandb_token) == 40
    login_succ = wandb.login(key=wandb_token, verify=True)
    assert login_succ
    wandb_setup(args)
    label = f'{args.label}YN'
    out_dir = args.output_dir
    targs = TrainingArguments(output_dir=out_dir,
                              overwrite_output_dir=True,
                              run_name=args.experiment_name,
                              do_train=True,
                              do_eval=True,
                              do_predict=True,
                              eval_strategy='epoch',
                              eval_on_start=True,
                              save_strategy='epoch',
                              save_only_model=True,
                              tf32=True,
                              bf16=True,
                              num_train_epochs=float(args.epochs),
                              dataloader_prefetch_factor=4,
                              dataloader_num_workers=4,
                              torch_empty_cache_steps=75,
                              per_device_train_batch_size=16,
                              per_device_eval_batch_size=16,
                              gradient_accumulation_steps=4,
                              logging_steps=10,
                              report_to=['wandb'],
                              push_to_hub=True,
                              hub_model_id=hub_model_id,
                              hub_strategy='all_checkpoints',)
    trainer = Trainer(model,
                      args=targs,
                      train_dataset=get_dataset('train', label, tokenizer),
                      eval_dataset=get_dataset('validation', label, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.train()
    trainer.evaluate()
    trainer.predict(get_dataset('test', label, tokenizer))
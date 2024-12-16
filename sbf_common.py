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
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from argparse import ArgumentParser

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
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 

def cls_loss(outputs, labels, *, num_items_in_batch):
    if isinstance(outputs, dict) and "loss" not in outputs:
        if labels is not None:
            logits = outputs["logits"]
            return F.cross_entropy(logits, labels)
        
def get_dataset(split: str, feature: str, tokenizer: PreTrainedTokenizer):
    ds = load_dataset('anoaky/sbf-collated', feature, split=split)
    def tokenize(x):
        post = tokenizer(x['post'], **tok_kwargs)
        label = torch.tensor(x[feature]).round().long()
        return {
            'input_ids': post.input_ids.view(-1),
            'attention_mask': post.attention_mask.view(-1),
            'labels': label,
        }
    ds = ds.map(tokenize)
    return ds

def wandb_setup(args):
    proj = os.environ['WANDB_PROJECT']
    wandb.init(
        project=proj,
        name=args.run,
        tags=[args.tag],
        group=args.problem,
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex', 'intent', 'speakerMinority'])
    parser.add_argument('-o', '--output_dir', default='/outputs/model', type=str)
    args = parser.parse_args()
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_PROJECT'] = 'COMP550'
    return args

def train(model, tokenizer, hub_model_id, args):
    wandb_token = os.environ['WANDB_TOKEN']
    assert len(wandb_token) == 40
    login_succ = wandb.login(key=wandb_token, verify=True)
    assert login_succ
    wandb_setup(args)
    feature = f'{args.problem}YN'
    out_dir = args.output_dir
    targs = TrainingArguments(output_dir=out_dir,
                              overwrite_output_dir=True,
                              run_name=args.experiment_name,
                              do_train=True,
                              do_eval=True,
                              do_predict=True,
                              eval_strategy='steps',
                              eval_steps=0.25,
                              eval_on_start=True,
                              save_strategy='epoch',
                              save_only_model=True,
                              num_train_epochs=5.0,
                              bf16=torch.cuda.is_bf16_supported(),
                              torch_empty_cache_steps=10,
                              per_device_train_batch_size=8,
                              per_device_eval_batch_size=8,
                              gradient_accumulation_steps=8,
                              logging_steps=10,
                              report_to=['wandb'],
                              push_to_hub=True,
                              hub_model_id=hub_model_id,
                              hub_strategy='all_checkpoints',)
    trainer = Trainer(model,
                      args=targs,
                      train_dataset=get_dataset('train', feature, tokenizer),
                      eval_dataset=get_dataset('validation', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.train()
    trainer.evaluate()
    trainer.predict(get_dataset('test', feature, tokenizer))
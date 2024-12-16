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
    (logits, _), labels = ep
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

def train(model, tokenizer, hub_model_id, args):
    wandb_token = os.environ['WANDB_TOKEN']
    assert len(wandb_token) == 40
    login_succ = wandb.login(key=wandb_token, verify=True)
    assert login_succ
    feature = f'{args.problem}YN'
    out_dir = args.output_dir
    targs = TrainingArguments(output_dir=out_dir,
                              overwrite_output_dir=True,
                              run_name=args.experiment_name,
                              do_train=True,
                              do_eval=True,
                              do_predict=True,
                              eval_strategy='epoch',
                              eval_on_start=False,
                              save_strategy='epoch',
                              num_train_epochs=args.epochs,
                              bf16=torch.cuda.is_bf16_supported(),
                              torch_empty_cache_steps=10,
                              per_device_train_batch_size=8,
                              per_device_eval_batch_size=8,
                              gradient_accumulation_steps=8,
                              logging_steps=10,
                              report_to=['wandb'],
                              push_to_hub=False,
                              hub_model_id=hub_model_id,)
    trainer = Trainer(model,
                      args=targs,
                      train_dataset=get_dataset('train', feature, tokenizer),
                      eval_dataset=get_dataset('validation', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.train()
    trainer.evaluate()
    trainer.predict(get_dataset('test', feature, tokenizer))
    model.push_to_hub(hub_model_id,
                      use_temp_dir=False,
                      revision=f'{args.problem}_{args.epochs}')
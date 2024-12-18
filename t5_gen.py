import wandb
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import torch
import numpy as np
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

def unpack_ep(ep):
    if len(ep.predictions) == 2:
        (logits, _), labels = ep
    else:
        logits, labels = ep
    return logits, labels

class GenMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, ep):
        logits, labels = unpack_ep(ep)
        preds = torch.tensor(logits).softmax(-1).argmax(-1).numpy().astype(np.uint8)
        yh = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        y = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        y = [s.split() for s in y]
        yh = [[s.split()] for s in yh]
        return {
            'bleu_score': corpus_bleu(yh, y)
        }

def wandb_setup():
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'false'
    os.environ['WANDB_PROJECT'] = 'COMP550'
    proj = 'COMP550'
    wandb.init(
        project=proj,
        name='t5-gen',
        tags=['t5', 'gen'],
        group='gen',
    )

def cls_loss(outputs, labels, *, num_items_in_batch):
    if isinstance(outputs, dict):
        if "loss" not in outputs:
            if labels is not None:
                logits = outputs["logits"]
                return F.cross_entropy(logits, labels)
        else:
            return outputs['loss']

def tokenize_ds(tokenizer, split):
    ds = load_dataset(HF_DS, split=split, trust_remote_code=True)
    ds = ds.select_columns(['post', 'targetStereotype'])
    def helper(row):
        tok_inputs = tokenizer(row['post'], **tok_kwargs)
        tok_outputs = tokenizer(row['targetStereotype'], **tok_kwargs)
        return {
            'input_ids': tok_inputs['input_ids'].view(-1),
            'attention_mask': tok_inputs['attention_mask'].view(-1),
            'labels': tok_outputs['input_ids'].view(-1),
        }
    ds = ds.map(helper, remove_columns=['post', 'targetStereotype'])
    return ds

def train(args):
    wandb_token = os.environ['WANDB_TOKEN']
    assert len(wandb_token) == 40
    login_succ = wandb.login(key=wandb_token, verify=True)
    assert login_succ
    wandb_setup()
    out_dir = args.output_dir
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')
    model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base')
    train_ds = tokenize_ds(tokenizer, 'train')
    val_ds = tokenize_ds(tokenizer, 'validation')
    cls_metrics = GenMetrics(tokenizer)
    targs = TrainingArguments(output_dir=out_dir,
                              overwrite_output_dir=True,
                              run_name='t5-gen',
                              do_train=True,
                              do_eval=True,
                              do_predict=True,
                              eval_strategy='epoch',
                              logging_strategy='epoch',
                              eval_on_start=True,
                              save_strategy='epoch',
                              save_only_model=True,
                              tf32=True,
                              bf16=True,
                              num_train_epochs=5.0,
                              dataloader_prefetch_factor=2,
                              dataloader_num_workers=2,
                              torch_empty_cache_steps=10,
                              auto_find_batch_size=True,
                              gradient_accumulation_steps=8,
                              report_to=['wandb'],
                              push_to_hub=True,
                              hub_model_id='anoaky/sbf-t5-gen',
                              hub_strategy='all_checkpoints',)
    trainer = Trainer(model,
                        args=targs,
                        train_dataset=train_ds,
                        eval_dataset=val_ds,
                        compute_metrics=cls_metrics,)
    trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='/outputs/model', type=str)
    parser.add_argument('-e', '--epochs', default=2, type=int)
    args = parser.parse_args()
    train(args)
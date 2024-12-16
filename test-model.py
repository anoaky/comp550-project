import torch
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sbf_common import get_dataset, cls_loss, cls_metrics, unpack_ep, wandb_setup
from argparse import ArgumentParser
import numpy as np
import wandb
import huggingface_hub as hf

AVAILABLE = {
    'bart': {
        'offensive': [555, 1110, 1665, 2220, 2770],
    }
}

def get_test_queue(args):
    tests = {}
    if args.model is None:
        for model in AVAILABLE:
            tests[model] = {}
    else:
        assert args.model in AVAILABLE
        tests[args.model] = {}
    
    if args.label is None:
        for model in tests:
            for label in AVAILABLE[model]:
                tests[model][label] = []
    else:
        for model in tests:
            assert args.label in AVAILABLE[model]
            tests[model][args.label] = []
    
    if args.checkpoint is None:
        for model in tests:
            for label in tests[model]:
                tests[model][label] += AVAILABLE[model][label]
    else:
        for model in tests:
            for label in tests[model]:
                assert args.checkpoint in AVAILABLE[model][label]
                tests[model][label] += [args.checkpoint]
    return tests

def cm_metric(ep):
    logits, labels = unpack_ep(ep)
    preds = torch.tensor(logits).softmax(-1).argmax(-1).numpy().astype(np.uint8)
    cm = wandb.plot.confusion_matrix(y_true=labels,
                                     preds=preds,
                                     class_names=['Positive', 'Negative'])
    return {
        'confusion_matrix': cm,
    }
            

def main(args):
    wandb_setup(args)
    repo_path = hf.snapshot_download(f'anoaky/sbf-{args.model}-{args.label}',
                                     repo_type='model')
    model_path = repo_path + f'/checkpoint-{args.checkpoint}'
    model = BartForSequenceClassification.from_pretrained(model_path,
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    feature = f'{args.label}YN'
    out_dir = '.test'
    targs = TrainingArguments(output_dir=out_dir,
                              resume_from_checkpoint=False,
                              run_name=f'{args.model}-{args.label}',
                              do_train=False,
                              do_eval=True,
                              eval_strategy='epoch',
                              eval_on_start=False,
                              save_strategy='no',
                              num_train_epochs=0.0,
                              torch_empty_cache_steps=10,
                              per_device_eval_batch_size=8,
                              report_to=['wandb'],
                              push_to_hub=False,)
    trainer = Trainer(model,
                      args=targs,
                      eval_dataset=get_dataset('test', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cm_metric,)
    trainer.evaluate(metric_key_prefix='test')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=int)
    args = parser.parse_args()
    tests = get_test_queue(args)
    for model in tests:
        for label in tests[model]:
            for checkpoint in tests[model][label]:
                args.model = model
                args.label = label
                args.checkpoint = checkpoint
                args.run = f'ttt'
                args.tags = [model, label]
                main(args)
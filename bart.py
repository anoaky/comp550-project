import comet_ml
import torch
import torch.optim
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from argparse import ArgumentParser
from sbf_common import get_dataset, cls_loss, cls_metrics
import os

def main(args):
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli',
                                                          num_labels=1,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model_id = f'anoaky/sbf-bart-{args.problem}'
    feature = f'{args.problem}YN'
    out_dir = f"./{model_id}"
    targs = TrainingArguments(output_dir=out_dir,
                              run_name=args.experiment_name,
                              do_train=True,
                              do_eval=True,
                              eval_strategy='epoch',
                              eval_on_start=True,
                              num_train_epochs=args.epochs,
                              bf16=torch.cuda.is_bf16_supported(),
                              per_device_train_batch_size=8,
                              per_device_eval_batch_size=8,
                              gradient_accumulation_steps=8,
                              logging_steps=50,
                              report_to=['comet_ml'],
                              push_to_hub=True,
                              hub_model_id=model_id,
                              hub_strategy='end',)
    trainer = Trainer(model,
                      args=targs,
                      train_dataset=get_dataset('train', feature, tokenizer),
                      eval_dataset=get_dataset('validation', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.create_model_card(language='en',
                              model_name=f'sbf-bart-{args.problem}',
                              finetuned_from='facebook/bart-large-mnli',
                              tasks='Text Classification',
                              dataset='allenai/social-bias-frames')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--experiment_name', required=True, type=str)
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex'])
    parser.add_argument('-e', '--epochs', default=1, type=int)
    args = parser.parse_args()
    main(args)
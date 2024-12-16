import neptune
import torch
import torch.optim
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from argparse import ArgumentParser
from sbf_common import get_dataset, cls_loss, cls_metrics
import os

def main(args):
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model_id = f'anoaky/sbf-bart'
    feature = f'{args.problem}YN'
    out_dir = args.output_dir
    targs = TrainingArguments(output_dir=out_dir,
                              overwrite_output_dir=True,
                              run_name=args.experiment_name,
                              do_train=True,
                              do_eval=True,
                              eval_strategy='epoch',
                              eval_on_start=False,
                              save_strategy='epoch',
                              num_train_epochs=args.epochs,
                              bf16=torch.cuda.is_bf16_supported(),
                              torch_empty_cache_steps=10,
                              per_device_train_batch_size=8,
                              per_device_eval_batch_size=8,
                              gradient_accumulation_steps=8,
                              logging_steps=50,
                              report_to=['neptune'],
                              push_to_hub=False,
                              hub_model_id=model_id,)
    trainer = Trainer(model,
                      args=targs,
                      train_dataset=get_dataset('train', feature, tokenizer),
                      eval_dataset=get_dataset('validation', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.train()
    trainer.evaluate()
    model.push_to_hub(model_id,
                      revision=f'{args.problem}_{args.epochs}')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex'])
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-o', '--output_dir', default='/tmp/model', type=str)
    args = parser.parse_args()
    args.experiment_name = f'sbf-bart-{args.problem}_{args.epochs}'
    os.environ['COMET_PROJECT_NAME'] = 'comp-550-project'
    main(args)
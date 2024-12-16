from transformers import BartTokenizer, BartForSequenceClassification
from argparse import ArgumentParser
from sbf_common import train
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex', 'intent'])
    parser.add_argument('-o', '--output_dir', default='/outputs/model', type=str)
    args = parser.parse_args()
    args.experiment_name = f'sbf-bart-{args.problem}'
    
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    hub_model_id = f'anoaky/sbf-bart-{args.problem}'
    os.environ['WANDB_LOG_MODEL'] = 'end'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_PROJECT'] = 'COMP550'
    args.run = f'bart-{args.problem}'
    args.grp = 'bart'
    train(model, tokenizer, hub_model_id, args)
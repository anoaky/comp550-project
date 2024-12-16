from transformers import BartTokenizer, BartForSequenceClassification
from argparse import ArgumentParser
from sbf_common import train

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex', 'intent'])
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-o', '--output_dir', default='/outputs/model', type=str)
    args = parser.parse_args()
    args.experiment_name = f'sbf-bart-{args.problem}_{args.epochs}'
    
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    hub_model_id = 'anoaky/sbf-bart'
    train(model, tokenizer, hub_model_id, args)
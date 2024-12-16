from transformers import BartTokenizer, BartForSequenceClassification
from sbf_common import train, parse_args

if __name__ == '__main__':
    args = parse_args()
    args.experiment_name = f'sbf-bart-{args.problem}'
    
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    hub_model_id = f'anoaky/sbf-bart-{args.problem}'
    
    args.run = f'bart-{args.problem}'
    args.tag = 'bart'
    train(model, tokenizer, hub_model_id, args)
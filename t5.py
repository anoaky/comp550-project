from transformers import T5Tokenizer, T5ForSequenceClassification
from sbf_common import train, parse_args

if __name__ == '__main__':
    args = parse_args()
    args.experiment_name = f'sbf-t5-{args.label}'
    
    model = T5ForSequenceClassification.from_pretrained('google-t5/t5-large',
                                                         num_labels=2,
                                                         ignore_mismatched_sizes=True,)
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large')
    hub_model_id = f'anoaky/sbf-t5-{args.label}'
    
    args.run = f't5-{args.label}'
    args.tags = ['t5', args.label]
    train(model, tokenizer, hub_model_id, args)
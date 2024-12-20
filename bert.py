from transformers import BertTokenizer, BertForSequenceClassification
from sbf_common import parse_args, train

if __name__ == '__main__':
    args = parse_args()
    args.experiment_name = f'sbf-bert-{args.label}'
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    hub_model_id = f'anoaky/sbf-bert-{args.label}'
    
    args.run = f'bert-{args.label}'
    args.tags = ['bert', args.label]
    train(model, tokenizer, hub_model_id, args)
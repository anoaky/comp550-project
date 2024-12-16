from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sbf_common import get_dataset, cls_loss, cls_metrics
from argparse import ArgumentParser

AVAILABLE = {
    'sbf-bart': {
        'offensive': [1, 2, 5],
    }
}

MODELS = ['sbf-bart']
LABELS = ['offensive']
EPOCHS = [1]

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
    
    if args.epochs == 0:
        for model in tests:
            for label in tests[model]:
                tests[model][label] += AVAILABLE[model][label]
    else:
        for model in tests:
            for label in tests[model]:
                assert args.epochs in AVAILABLE[model][label]
                tests[model][label] += [args.epochs]
    return tests
            

def main(args):
    model = BartForSequenceClassification.from_pretrained(f'anoaky/{args.model}',
                                                          revision=f'{args.label}_{args.epochs}',
                                                          num_labels=2,
                                                          ignore_mismatched_sizes=True,)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    feature = f'{args.label}YN'
    out_dir = '.test'
    targs = TrainingArguments(output_dir=out_dir,
                              run_name=f'{args.model}-{args.label}_{args.epochs}',
                              do_train=False,
                              do_eval=True,
                              eval_strategy='epoch',
                              eval_on_start=False,
                              save_strategy='no',
                              num_train_epochs=args.epochs,
                              torch_empty_cache_steps=10,
                              per_device_eval_batch_size=8,
                              report_to=['neptune'],
                              push_to_hub=False,)
    trainer = Trainer(model,
                      args=targs,
                      eval_dataset=get_dataset('test', feature, tokenizer),
                      compute_loss_func=cls_loss,
                      compute_metrics=cls_metrics,)
    trainer.evaluate(metric_key_prefix='test')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    parser.add_argument('--epochs', default=0, type=int)
    args = parser.parse_args()
    tests = get_test_queue(args)
    print(tests)
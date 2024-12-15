import comet_ml
import torch
import torch.optim
import huggingface_hub as hf
from transformers import BartTokenizer
from sbf_common import SBFModel, SBFTrainer
from argparse import ArgumentParser

def get_model(args):
    match args.problem:
        case 'offensive':
            return BartOffensive()
        case 'sex':
            return BartSex()
        case 'category':
            return BartCategory()

class BartOffensive(SBFModel):
    def __init__(self):
        super().__init__(model_path='facebook/bart-large-mnli',
                         base_name='sbf-bart-offensive',
                         num_labels=3,)
        self.label_key = 'offensive'
    
class BartSex(SBFModel):
    def __init__(self):
        super().__init__(model_path='facebook/bart-large-mnli',
                         base_name='sbf-bart-sex',
                         num_labels=3,)
        self.label_key = 'sex'

class BartCategory(SBFModel):
    def __init__(self):
        super().__init__(model_path='facebook/bart-large-mnli',
                         base_name='sbf-bart-category',
                         num_labels=8)
        self.label_key = 'category'

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    expconf = comet_ml.ExperimentConfig(#disabled=True,
                                        name=args.experiment_name,
                                        parse_args=False)
    experiment = comet_ml.start(workspace='anoaky',
                                project_name='comp-550-project',
                                experiment_config=expconf)
    experiment.disable_mp()
    model = get_model(args)
    trainer = SBFTrainer(max_epochs=args.max_epochs,
                         step_every=args.step_every,
                         experiment=experiment)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'prefetch_factor': 2,
    }
    train_loader, val_loader, test_loader = model.dataloaders(tokenizer, **dataloader_kwargs)
    try:
        trainer.fit(model,
                    device,
                    train_loader=train_loader,
                    val_loader=val_loader,)
    except Exception as e:
        experiment_key = experiment.get_key()
        experiment.send_notification(args.experiment_name,
                                     status='failed')
        experiment.end()
        comet_ml.APIExperiment(previous_experiment=experiment_key).archive()
        raise e
    experiment.end()
    experiment.send_notification(args.experiment_name,
                                 status='finished')
    model.push_to_hub(f'anoaky/{model.base_name}',
                      branch=args.branch_name)

if __name__ == '__main__':
    if torch.cuda.is_bf16_supported():
        torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_argument('-n', '--experiment_name', required=True, type=str)
    parser.add_argument('-p', '--problem', required=True, choices=['offensive', 'sex', 'category'])
    parser.add_argument('-e', '--max_epochs', default=3, type=int)
    parser.add_argument('-g', '--step_every', default=4, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-w', '--num_workers', default=0, type=int)
    parser.add_argument('-c', '--checkout', action="store_true")
    args = parser.parse_args()
    if args.checkout:
        args.branch_name = args.experiment_name
    else:
        args.branch_name = 'main'
    print(f'EXPERIMENT NAME: {args.experiment_name}',
          f'SELECTED PROBLEM: {args.problem}',
          f'EPOCHS: {args.max_epochs}',
          f'WORKERS: {args.num_workers}',
          f'BATCH {args.batch_size} ACCUMULATE FOR {args.step_every}',
          )
    main(args)
import comet_ml
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BartTokenizer, BartConfig, BartForSequenceClassification
from trainer import prep_dataset, Trainer
from argparse import ArgumentParser

tok_kwargs = {
    'padding': 'max_length',
    'max_length': 256,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True
}


class BartSBFDataset(Dataset):
    def __init__(self, hf_data, tok):
        super().__init__()
        self.hf_data = hf_data
        self.tok = tok

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, index):
        row = self.hf_data[index]
        inputs = self.tok(row['post'], **tok_kwargs)
        target = row['targetCategory']
        return [
            inputs['input_ids'].view(-1),
            inputs['attention_mask'].view(-1),
            target
        ]


class BartForBiasClassification(nn.Module):
    def __init__(self, config: BartConfig, experiment, args, lr=1e-5):
        self.bart = BartForSequenceClassification.from_pretrained(
            'facebook/bart-large', config=config)
        self.lr = lr
        self.experiment = experiment
        self.args = args
        self.base_name = 'bart-large'

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loader(self, tok, /, split, shuffle, **dataloader_kwargs):
        loader = load_dataset('allenai/social_bias_frames', split=split)
        loader = prep_dataset(loader)
        loader = BartSBFDataset(loader, tok)
        loader = DataLoader(loader, shuffle=shuffle, **dataloader_kwargs)

    def train_loader(self, tok, **dataloader_kwargs):
        return self.loader(tok, split='train', shuffle=True, **dataloader_kwargs)

    def val_loader(self, tok, **dataloader_kwargs):
        return self.loader(tok, split='validation', shuffle=False, **dataloader_kwargs)

    def forward(self, input_ids, attention_mask, labels):
        out = self.bart(
            input_ids, attention_mask=attention_mask, labels=labels)
        return out

    def train_step(self, idx, batch):
        out = self.forward(*batch)
        return out.loss

    def val_step(self, idx, batch):
        out = self.forward(*batch)
        logits = out.logits
        yh = torch.argmax(logits, dim=1)
        accuracy = torch.sum(yh == batch[2]).item() / (len(yh) * 1.0)
        return accuracy


def main(args):
    if torch.cuda.is_bf16_supported():
        print('MEDIUM PRECISION')
        torch.set_float32_matmul_precision('medium')
    config = BartConfig(num_labels=8)
    expconfig = comet_ml.ExperimentConfig(disabled=True,
                                          name=args.experiment_name,
                                          tags=['bart'])
    experiment = comet_ml.start(api_key=os.environ['COMET_API_KEY'],
                                workspace='anoaky',
                                project_name='comp-550-project',
                                experiment_config=expconfig)
    experiment.disable_mp()
    model = BartForBiasClassification(config, experiment, args, lr=1e-5)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'pin_memory': True
    }

    train_loader = model.train_loader(tokenizer, **dataloader_kwargs)
    val_loader = model.val_loader(tokenizer, **dataloader_kwargs)

    trainer = Trainer(experiment, args.max_epochs, args.log_every)
    trainer.fit(model, train_loader=train_loader, val_loader=val_loader)
    experiment.send_notification(args.experiment_name, status='finished')
    experiment.end()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()
    main(args)

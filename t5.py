from argparse import ArgumentParser
import comet_ml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
import torch.multiprocessing as mp
from transformers import T5Tokenizer, T5Config, T5Model
import os
from tqdm import tqdm
from trainer import prep_dataset, FabricTrainer, CometCallback, FabricSummary
from old_bart import SBFDataset
import lightning as L

tok_kwargs = {
    'padding': 'max_length',
    'max_length': 256,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True
}

class T5Dataset(Dataset):
    def __init__(self, hf_data, tok, /):
        super().__init__()
        self.hf_data = hf_data
        self.tok = tok
    
    def __len__(self):
        return len(self.hf_data)
    
    def __getitem__(self, index):
        row = self.hf_data[index]
        inputs = self.tok(row['post'], **tok_kwargs)
        labels = self.tok(row['targetStereotype'])
        

class T5Bias(L.LightningModule):
    def __init__(self, config: T5Config):
        super().__init__()
        self.t5 = T5Model.from_pretrained(
            't5-3b')
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
        self.t5.train()

    def forward(self, input_ids, attention_mask, labels, /):
        x = self.t5(input_ids, 
                       attention_mask=attention_mask,
                       labels=labels)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def training_step(self, input_ids, attention_mask, labels, /):
        out = self.forward(input_ids, attention_mask, labels)
        return out.loss

    def validation_step(self, input_ids, attention_mask, y):
        out = self.forward(input_ids, attention_mask)
        val_loss = out.loss
        yh = torch.argmax(out.logits, dim=1)
        val_acc = torch.sum(y == yh).item() / (y.shape[0] * 1.0)
        return val_loss, val_acc

    def train_dataloader(self, tokenizer, **kwargs):
        train_set = load_dataset(
            'allenai/social_bias_frames', split='train', trust_remote_code=True)
        train_set = prep_dataset(train_set)
        train_set = SBFDataset(train_set, tokenizer)
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)
        return train_loader

    def val_dataloader(self, tokenizer, **kwargs):
        val_set = load_dataset('allenai/social_bias_frames',
                               split='validation', trust_remote_code=True)
        val_set = prep_dataset(val_set)
        val_set = SBFDataset(val_set, tokenizer)
        val_loader = DataLoader(val_set, shuffle=False, **kwargs)
        return val_loader


def main(args):
    expconfig = comet_ml.ExperimentConfig(#disabled=True,
                                          name=args.experiment_name)
    experiment = comet_ml.start(workspace='anoaky',
                                project_name='comp-550-project',
                                experiment_config=expconfig)
    experiment.disable_mp()
    experiment_key = experiment.get_key()
    experiment.end()
    comet_cb = CometCallback(prefix='t5-3B',
                             experiment_key=experiment_key)
    fabric_summary = FabricSummary()
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'pin_memory': True,
        'prefetch_factor': 8
    }
    
    config = T5Config(num_labels=8)
    model = T5Bias(config)
    torch.compile(model)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    train_loader = model.train_dataloader(tokenizer, **dataloader_kwargs)
    val_loader = model.val_dataloader(tokenizer, **dataloader_kwargs)
    fabric = L.Fabric(callbacks=[comet_cb, fabric_summary],
                      loggers=[],
                      precision="bf16-mixed",
                      strategy="fsdp")
    trainer = FabricTrainer(fabric,
                            max_epochs=args.max_epochs,
                            log_every=args.log_every)
    trainer.fit(model, train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--log_every', default=50, type=int)
    args = parser.parse_args()
    main(args)

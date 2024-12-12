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
from transformers import T5Tokenizer, T5Config, AutoModel
import os
from tqdm import tqdm
from trainer import prep_dataset, FabricTrainer, CometCallback, FabricSummary
from bart import SBFDataset
import lightning as L


class T5Bias(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.t5 = AutoModel.from_pretrained(
            't5-11b', ignore_mismatched_sizes=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
        self.t5.train()

    def forward(self, input_ids, attention_mask, /):
        _, x = self.t5(input_ids, attention_mask=attention_mask,
                       return_dict=False)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def training_step(self, input_ids, attention_mask, y, /):
        yh = self.forward(input_ids, attention_mask)
        train_loss = F.binary_cross_entropy_with_logits(yh, y)
        return train_loss

    def validation_step(self, input_ids, attention_mask, y):
        yh = self.forward(input_ids, attention_mask)
        val_loss = F.binary_cross_entropy_with_logits(yh, y)
        yh = torch.argmax(yh, dim=1)
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
    expconfig = comet_ml.ExperimentConfig(disabled=True,
                                          name=args.experiment_name)
    comet_cb = CometCallback(prefix='t5-11B',
                             api_key=os.environ['COMET_API_KEY'],
                             workspace='anoaky',
                             project_name='comp-550-project',
                             experiment_config=expconfig)
    fabric_summary = FabricSummary()
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'pin_memory': True,
        'prefetch_factor': 8
    }

    model = T5Bias()
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

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
import torch.multiprocessing as mp
from transformers import T5Tokenizer, T5ForSequenceClassification, T5Config
import os
from tqdm import tqdm
from trainer import prep_dataset
from bart import SBFDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
class T5Bias(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.t5 = T5ForSequenceClassification.from_pretrained('google-t5/t5-11b', config=config)
    
    def forward(self, input_ids, attention_mask, labels):
        out = self.t5(input_ids, attention_mask=attention_mask, labels=labels)
        return out
    
    def train_step(self, idx, input_ids, attention_mask, labels):
        out = self.forward(input_ids, attention_mask, labels)
        return out.loss
    

def shard_model(rank, model):
    model = model.to(rank)
    sharded_model = FullyShardedDataParallel(model, cpu_offload=CPUOffload(offload_params=True))
    optim = torch.optim.Adam(sharded_model.parameters(), lr=1e-5)
    return sharded_model, optim
                
def main(rank, world_size):
    setup(rank, world_size)
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
    train_set = load_dataset('allenai/social_bias_frames', split='train', trust_remote_code=True)
    train_set = prep_dataset(train_set)
    train_set = SBFDataset(train_set, tokenizer)
    train_sampler = DistributedSampler(train_set, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, sampler=train_sampler)
    
    config = T5Config(num_labels=8)
    model = T5Bias(config)
    model, optimizer = shard_model(rank, model)
    
    trainer = FSDPTrainer(world_size)
    trainer.fit(model, optimizer, 1, train_loader)
                
class FSDPTrainer:
    def __init__(self, world_size):
        self.world_size = world_size
    def fit(self, model, optimizer, max_epochs, train_loader):
        with tqdm() as t:
            for epoch in range(max_epochs):
                train_loader.sampler.set_epoch(epoch)
                t.reset(total=len(train_loader))
                t.set_description(f'Training epoch {epoch}')
                for idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = model(idx, *batch)
                    loss.backward()
                    optimizer.step()
                    t.set_postfix(loss=loss.item())
                    t.update()
        
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f'WORLD_SIZE: {world_size}')
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    cleanup()
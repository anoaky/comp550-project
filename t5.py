import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
import torch.multiprocessing as mp
from transformers import T5Tokenizer, T5Config, T5Model
import os
from tqdm import tqdm
from trainer import prep_dataset
from bart import SBFDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45355'
    dist.init_process_group('nccl', init_method='file:///tmp/sharedfile', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
class T5Bias(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.t5 = T5Model.from_pretrained('t5-11b', config=config)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
    
    def forward(self, input_ids, attention_mask, labels):
        _, x = self.t5(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_step(self, idx, input_ids, attention_mask, labels):
        yh = self.forward(input_ids, attention_mask, labels)
        loss = F.binary_cross_entropy_with_logits(yh, labels)
        return loss
    

def shard_model(rank, model):
    model = model.to(rank)
    sharded_model = FullyShardedDataParallel(model, cpu_offload=CPUOffload(offload_params=True))
    optim = torch.optim.Adam(sharded_model.parameters(), lr=1e-5)
    return sharded_model, optim
                
def main(rank, world_size):
    setup(rank, world_size)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
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
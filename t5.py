import comet_ml
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
        self.t5 = T5Model.from_pretrained('t5-11b', config=config, ignore_mismatched_sizes=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
    
    def forward(self, input_ids, attention_mask):
        _, x = self.t5(input_ids, attention_mask=attention_mask, return_dict=False)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_step(self, idx, input_ids, attention_mask, labels):
        yh = self.forward(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(yh, labels)
        return loss
    

def shard_model(rank, model):
    sharded_model = FullyShardedDataParallel(model, cpu_offload=CPUOffload(offload_params=True))
    optim = torch.optim.Adam(sharded_model.parameters(), lr=1e-5)
    return sharded_model, optim
                
def main(rank, world_size, experiment_key):
    setup(rank, world_size)
    experiment = comet_ml.start(api_key=os.environ['COMET_API_KEY'],
                                experiment_key=experiment_key)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    train_set = load_dataset('allenai/social_bias_frames', split='train', trust_remote_code=True)
    train_set = prep_dataset(train_set)
    train_set = SBFDataset(train_set, tokenizer)
    train_sampler = DistributedSampler(train_set, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, sampler=train_sampler)
    
    config = T5Config(num_labels=8)
    model = T5Bias(config)
    model = model.to(f'cuda:{rank}')
    model, optimizer = shard_model(rank, model)
    
    trainer = FSDPTrainer(world_size)
    trainer.fit(model, optimizer, experiment, 1, train_loader)
                
class FSDPTrainer:
    def __init__(self, world_size):
        self.world_size = world_size
    def fit(self, model, optimizer, experiment, max_epochs, train_loader):
        with tqdm() as t:
            model.train()
            for epoch in range(max_epochs):
                train_loader.sampler.set_epoch(epoch)
                t.reset(total=len(train_loader))
                t.set_description(f'Training epoch {epoch}')
                for idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = model.train_step(idx, *batch)
                    loss.backward()
                    optimizer.step()
                    if idx % 50 == 49:
                        experiment.log_metrics({'loss': loss.item()}, 
                                               prefix='t5-11b',
                                               step=idx+1,
                                               epoch=epoch)
                        print(f'batch {idx+1} loss {loss.item()}')
                    t.set_postfix(loss=loss.item())
                    t.update()
        
if __name__ == '__main__':
    expconfig = comet_ml.ExperimentConfig(name='t5-11b-32')
    experiment = comet_ml.start(api_key=os.environ['COMET_API_KEY'],
                                workspace='anoaky',
                                project_name='comp-550-project',
                                experiment_config=expconfig)
    experiment.disable_mp()
    experiment.log_parameters({'batch_size': 32, 'max_epochs': 10})
    experiment.end()
    world_size = torch.cuda.device_count()
    print(f'WORLD_SIZE: {world_size}')
    mp.spawn(main,
             args=(world_size,experiment.get_key()),
             nprocs=world_size,
             join=True)
    cleanup()
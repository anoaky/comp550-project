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
from trainer import prep_dataset
from bart import SBFDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45355'
    dist.init_process_group('nccl', init_method='file:///tmp/sharedfile', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
class T5Bias(nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5Model.from_pretrained('t5-11b', ignore_mismatched_sizes=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
    
    def forward(self, input_ids, attention_mask):
        _, x = self.t5(input_ids, attention_mask=attention_mask, return_dict=False)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def prepare_model(rank):
    model = T5Bias()
    model = model.to(rank)
    model = DDP(model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True)
    return model
    
def get_loaders_ddp():
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    train_set = load_dataset('allenai/social_bias_frames', split='train', trust_remote_code=True)
    train_set = prep_dataset(train_set)
    train_set = SBFDataset(train_set, tokenizer)
    train_sampler = DistributedSampler(train_set, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, sampler=train_sampler)
    
    val_set = load_dataset('allenai/social_bias_frames', split='validation', trust_remote_code=True)
    val_set = prep_dataset(val_set)
    val_set = SBFDataset(val_set, tokenizer)
    val_sampler = DistributedSampler(val_set, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, sampler=val_sampler)
    
    return train_loader, val_loader
                
def main(rank, world_size, experiment_key):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    experiment = comet_ml.start(api_key=os.environ['COMET_API_KEY'],
                                experiment_key=experiment_key)
    experiment.disable_mp()
    train_loader, val_loader = get_loaders_ddp()
    
    model = prepare_model(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    t = tqdm()
    for epoch in range(10):
        train_loader.sampler.set_epoch(epoch)
        t.reset(total=len(train_loader))
        t.set_description(f'Training epoch {epoch}')
        with experiment.train():
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch[0].to(rank)
                attention_mask = batch[1].to(rank)
                yh = batch[2].to(rank)
                y = model(input_ids, attention_mask)
                loss = F.binary_cross_entropy_with_logits(y, yh)
                loss.backward()
                optimizer.step()
                if idx % 10 == 9:
                    gloss = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
                    dist.all_gather(gloss, loss)
                    loss = torch.mean(gloss)
                    experiment.log_metrics({f'loss': loss.item()}, 
                                           prefix='t5-11b',
                                           step=idx+1,
                                           epoch=epoch)
                    t.set_postfix(loss=loss.item())
                t.update()
    t.close()
        
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
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
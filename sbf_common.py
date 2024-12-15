import comet_ml
import torch
import torch.nn as nn
import datasets
import typing
from torch.utils.data import Dataset, DataLoader
from transformers import (BartTokenizer,
                          BertTokenizer,
                          T5Tokenizer, 
                          AutoModelForSequenceClassification,
                          AutoConfig,
                          )
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import PyTorchModelHubMixin

MAX_LENGTH = 256
HF_DS = 'allenai/social_bias_frames'

tok_kwargs = {
    'padding': 'max_length',
    'max_length': MAX_LENGTH,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True,
    'add_special_tokens': True,
}

class SBFPreprocessed(Dataset):
    def __init__(self, 
                 hf_data: datasets.Dataset, 
                 tok: typing.Union[BartTokenizer | BertTokenizer | T5Tokenizer],
                 /,):
        super().__init__()
        self.hf_data = hf_data
        self.tok = tok
        self.preprocess()
    
    def preprocess(self):
        def remove_blanks(row):
            if len(row['targetCategory']) == 0:
                row['targetCategory'] = 'none'
            return row
        selected_columns = [
            'post',
            'intentYN',
            'offensiveYN',
            'sexYN',
            'speakerMinorityYN',
            'targetCategory'
        ]
        class_encode = [
            'intentYN',
            'offensiveYN',
            'sexYN',
            'speakerMinorityYN',
            'targetCategory'
        ]
        self.hf_data = self.hf_data.select_columns(selected_columns).map(remove_blanks)
        for col in class_encode:
            self.hf_data = self.hf_data.class_encode_column(col)
    
    def __len__(self):
        return len(self.hf_data)
    
    def __getitem__(self, idx) -> typing.Dict[str, torch.Tensor]:
        row = self.hf_data[idx]
        tok_post = self.tok.encode_plus(row['post'],
                                        **tok_kwargs)
        intent = torch.tensor(row['intentYN'])
        offensive = torch.tensor(row['offensiveYN'])
        sex = torch.tensor(row['sexYN'])
        minority = torch.tensor(row['speakerMinorityYN'])
        category = torch.tensor(row['targetCategory'])
        return {
            'ids': tok_post.input_ids.view(-1),
            'attn': tok_post.attention_mask.view(-1),
            'intent': intent,
            'offensive': offensive,
            'sex': sex,
            'minority': minority,
            'category': category,
        }
        
class SBFModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 *,
                 model_path: str,
                 base_name: str,
                 num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                                        config=config,
                                                                        ignore_mismatched_sizes=True)
        self.base_name = base_name
        self.model.train()
        
    def forward(self, ids, attns, labels):
        return self.model(ids, attention_mask=attns, labels=labels)
        
    def training_step(self,
                      ids: torch.LongTensor,
                      attn: torch.LongTensor,
                      labels: torch.LongTensor,
                      ) -> torch.FloatTensor:
        out = self.forward(ids, attn, labels)
        return out.loss
    
    def validation_step(self, 
                        ids: torch.LongTensor,
                        attn: torch.LongTensor,
                        labels: torch.LongTensor,
                        ) -> torch.FloatTensor:
        return self.training_step(ids, attn, labels) # TODO
    
    def test_step(self,
                  batch: typing.Dict[str, torch.LongTensor],
                  label_key: str,
                  /,
                  ) -> torch.FloatTensor:
        ids = batch['ids']
        attns = batch['attn']
        labels = batch[label_key]
        out = self.forward(ids, attns, labels)
        return out.logits
    
    def get_loader(self, 
                   tokenizer,
                   /,
                   *,
                   split: str,
                   shuffle: bool,
                   **kwargs) -> DataLoader:
        hf_data = load_dataset(HF_DS, split=split, trust_remote_code=True)
        hf_set = SBFPreprocessed(hf_data, tokenizer)
        hf_loader = DataLoader(hf_set, shuffle=shuffle, **kwargs)
        return hf_loader
    
    def dataloaders(self, tokenizer, **kwargs):
        return (self.train_dataloader(tokenizer, **kwargs),
                self.val_dataloader(tokenizer, **kwargs),
                self.test_dataloader(tokenizer, **kwargs))
    
    def train_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='train', shuffle=True, **kwargs)
    
    def val_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='validation', shuffle=False, **kwargs)
    
    def test_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='test', shuffle=False, **kwargs)
    
class SBFTrainer:
    def __init__(self,
                 *,
                 max_epochs: int,
                 step_every: int,
                 experiment: comet_ml.Experiment,
                 ):
        self.max_epochs = max_epochs
        self.step_every = step_every
        self.experiment = experiment
        
    def move_all(device, *args):
        for i in len(args):
            args[i] = args[i].to(device)
        return args
        
    def fit(self, 
            model: SBFModel,
            device: str,
            *,
            label_key: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            ):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = model.to(device)
        t = tqdm()
        for epoch in range(self.max_epochs):
            train_len = len(train_loader)
            val_len = len(val_loader)
            t.reset(total=train_len)
            t.set_description(f'Training epoch {epoch}')
            model.train()
            running_loss = torch.tensor(0.0, device=device)
            for idx, batch in enumerate(train_loader):
                ids = batch['ids']
                attn = batch['attn']
                labels = batch[label_key]
                loss = model.training_step(ids.to(device), attn.to(device), labels.to(device))
                running_loss = running_loss + loss
                acc_norm = torch.min(torch.tensor([4, train_len - idx])) # this accounts for number of batches not evenly dividing step_every
                loss = loss / acc_norm
                loss.backward()
                
                if (idx + 1) % self.step_every == 0 or (idx + 1) == train_len:
                    optimizer.step()
                    optimizer.zero_grad()
                t.update()
            train_loss = running_loss / train_len
            self.experiment.log_metrics({'train_loss': train_loss.item()},
                                        prefix=model.base_name,
                                        epoch=epoch)
            
            model.eval()
            running_loss = torch.tensor(0.0, device=device)
            t.reset(total=val_len)
            t.set_description(f'Validation epoch {epoch}')
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch['ids']
                    attn = batch['attn']
                    labels = batch[label_key]
                    loss = model.validation_step(ids.to(device), attn.to(device), labels.to(device))
                    running_loss = running_loss + loss
                    
                    t.update()
        t.clear()
import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import lightning as L
import typing
from torch.utils.data import Dataset, DataLoader
from transformers import (BartTokenizer, BartForSequenceClassification,
                          BertTokenizer, BertForSequenceClassification,
                          T5Tokenizer, T5ForSequenceClassification
                          )
from datasets import load_dataset

MAX_LENGTH = 256
HF_PATH = 'allenai/social_bias_frames'

type ProjectModel = typing.Union[BertForSequenceClassification | BartForSequenceClassification | T5ForSequenceClassification]
type ProjectTokenizer = typing.Union[BertTokenizer | BartTokenizer | T5Tokenizer]

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
            if len(row['targetStereotype']) == 0:
                row['targetStereotype'] = 'not applicable'
            if len(row['targetCategory'] == 0):
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
        
class SBFModel(L.LightningModule):
    def __init__(self, 
                 model: typing.Union[BertForSequenceClassification 
                                     | BartForSequenceClassification 
                                     | T5ForSequenceClassification],
                 /,
                 *,
                 lr: float=1e-4):
        self.model = model
        self.lr = lr
        self.model.train()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def training_step(self, 
                      batch: typing.Dict[str, torch.LongTensor],
                      label_key: str,
                      /,
                      ) -> torch.FloatTensor:
        ids = batch['ids']
        attns = batch['attn']
        labels = batch[label_key]
        out = self.model.forward(ids, attention_mask=attns, labels=labels)
        return out.loss
    
    def validation_step(self, 
                        batch: typing.Dict[str, torch.LongTensor],
                        label_key: str,
                        /,
                        ) -> torch.FloatTensor:
        return self.training_step(batch, label_key) # TODO
    
    def test_step(self,
                  batch: typing.Dict[str, torch.LongTensor],
                  label_key: str,
                  /,
                  ) -> torch.FloatTensor:
        ids = batch['ids']
        attns = batch['attn']
        labels = batch[label_key]
        out = self.model.forward(ids, attention_mask=attns, labels=labels)
        return out.logits
    
    def get_loader(self, 
                   tokenizer,
                   /,
                   *,
                   split: str,
                   shuffle: bool,
                   **kwargs) -> DataLoader:
        hf_data = load_dataset(HF_PATH, split=split, trust_remote_code=True)
        hf_set = SBFPreprocessed(hf_data, tokenizer)
        hf_loader = DataLoader(hf_set, shuffle=shuffle, **kwargs)
        return hf_loader
    
    def train_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='train', shuffle=True, **kwargs)
    
    def val_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='validation', shuffle=False, **kwargs)
    
    def test_dataloader(self, tokenizer, **kwargs):
        return self.get_loader(tokenizer, split='test', shuffle=False, **kwargs)
    
class SBFTrainer(L.Fabric):
    def __init__(self,
                 *,
                 max_epochs: int,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.max_epochs = max_epochs
        
    def run(self, 
            model: SBFModel,
            *,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            ):
        optimizer = model.configure_optimizers()
        model, optimizer = self.setup(model, optimizer)
        [train_loader, val_loader, test_loader] = self.setup_dataloaders(train_loader,
                                                                         val_loader,
                                                                         test_loader)
        state = {
            'model': model,
            'optimizer': optimizer,
        }
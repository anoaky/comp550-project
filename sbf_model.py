import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, AutoModel, T5Config, T5ForConditionalGeneration
from transformers.generation import GenerateEncoderDecoderOutput
from datasets import load_dataset
import datasets
import lightning as L
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
from typing import List
from trainer import CometCallback, FabricSummary
from tqdm import tqdm
from argparse import ArgumentParser
import os

MAX_LENGTH = 256

tok_kwargs = {
    'padding': 'max_length',
    'max_length': MAX_LENGTH,
    'truncation': True,
    'return_tensors': 'pt',
    'return_attention_mask': True,
    'add_special_tokens': True,
}

class SBFPreprocessed(Dataset):
    def __init__(self, hf_data: datasets.Dataset, tok: T5Tokenizer):
        super().__init__()
        self.hf_data = hf_data
        self.tok = tok
        self.preprocess()
    
    def preprocess(self):
        def remove_blanks(row):
            if len(row['targetStereotype']) == 0:
                row['targetStereotype'] = 'not applicable'
            return row
        self.hf_data = self.hf_data.select_columns(['post', 'targetStereotype', 'intentYN', 'offensiveYN']) \
                                   .class_encode_column('intentYN') \
                                   .class_encode_column('offensiveYN') \
                                   .map(remove_blanks)
    
    def __len__(self):
        return len(self.hf_data)
    
    def __getitem__(self, idx):
        row = self.hf_data[idx]
        tok_post = self.tok.encode_plus(row['post'],
                                        **tok_kwargs)
        tok_stype = self.tok.encode_plus(row['targetStereotype'],
                                         **tok_kwargs)
        return {
            'post_ids': tok_post.input_ids.view(-1),
            'post_attn': tok_post.attention_mask.view(-1),
            'stype_ids': tok_stype.input_ids.view(-1)
        }

class SBFTransformer(L.LightningModule):
    base_name = 'sbf-transformer'
    
    def __init__(self, tokenizer: T5Tokenizer):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')
        self.t5.train()
        self.tokenizer = tokenizer
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    
    def forward(self, input_ids, attention_mask, tgt, /):
        out = self.t5(input_ids, attention_mask=attention_mask, labels=tgt)
        return out
    
    def generate(self, input_ids, /) -> GenerateEncoderDecoderOutput:
        out = self.t5.generate(input_ids, 
                               return_dict_in_generate=True,
                               num_return_sequences=1)
        return out
    
    def training_step(self, input_ids, attn_mask, tgt_seq):
        out = self.forward(input_ids, attn_mask, tgt_seq)
        return out.loss
    
    def validation_step(self, input_ids, tgt_seq):
        out = self.generate(input_ids, output_logits=True)
        out_seq = out.sequences
        logits = torch.stack(out.logits, dim=1)
        loss = F.cross_entropy(logits, tgt_seq)
        tgt_str = self.tokenizer.batch_decode(tgt_seq, # this needs to be a tensor, so we have to encode then decode it T.T
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
        out_str = self.tokenizer.batch_decode(out_seq,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
        p, r, f1 = bertscore(tgt_str, out_str)
        return loss, p, r, f1
    
    def train_dataloader(self, tokenizer: T5Tokenizer, **kwargs):
        train_ds = load_dataset('allenai/social_bias_frames', 
                                split='train', 
                                trust_remote_code=True)
        train_set = SBFPreprocessed(train_ds, tokenizer)
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)
        return train_loader
    
    def val_dataloader(self, tokenizer: T5Tokenizer, **kwargs):
        val_ds = load_dataset('allenai/social_bias_frames', 
                                split='train', 
                                trust_remote_code=True)
        val_set = SBFPreprocessed(val_ds, tokenizer)
        val_loader = DataLoader(val_set, shuffle=False, **kwargs)
        return val_loader
    
class SBFTrainer:
    def __init__(self, *, max_epochs: int, log_every: int):
        self.max_epochs = max_epochs
        self.log_every = log_every
        
    def fit(self, fabric: L.Fabric, model: L.LightningModule, train_loader, val_loader, /):
        if fabric.is_global_zero:
            fabric.call("print_summary", module=model)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, 
                                        optimizer, 
                                        _reapply_compile=False)
        [train_loader, val_loader] = fabric.setup_dataloaders(train_loader,
                                                              val_loader)
        t = tqdm()
        for epoch in range(self.max_epochs):
            t.reset(total=len(train_loader))
            t.set_description(f'Training epoch {epoch}')
            model.train()
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['post_ids']
                attn_mask = batch['post_attn']
                stype = batch['stype_ids']
                loss = model.training_step(input_ids, attn_mask, stype)
                fabric.backward(loss)
                optimizer.step()
                if idx % self.log_every == self.log_every - 1:
                    all_loss = fabric.all_reduce(loss)
                    if fabric.is_global_zero:
                            fabric.call("log_metrics", 
                                        train_loss=all_loss.item(),
                                        step=idx+1,
                                        epoch=epoch)
                    t.set_postfix(loss=all_loss.item())
                t.update()
            t.reset(total=len(val_loader))
            t.set_description(f'Validation epoch {epoch}')
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    input_ids = batch['post_ids']
                    tgt_seq = batch['stype_ids']
                    loss, p, r, f1 = model.validation_step(input_ids, tgt_seq)
                    loss = fabric.all_reduce(loss)
                    p = fabric.all_reduce(p)
                    r = fabric.all_reduce(r)
                    f1 = fabric.all_reduce(f1)
                    if fabric.is_global_zero:
                            fabric.call("log_metrics",
                                        val_loss=loss.item(),
                                        precision=p.item(),
                                        recall=r.item(),
                                        f1=f1.item(),
                                        epoch=epoch)
                    t.set_postfix(p=p.item(),
                                  r=r.item(),
                                  f1=f1.item())
                    t.update()
        t.close()
        fabric.call("on_fit_end")

def main(args):
    expconfig = comet_ml.ExperimentConfig(#disabled=True,
                                          name=args.experiment_name,
                                          parse_args=False)
    if 'EXPERIMENT_KEY' in os.environ:
        experiment = comet_ml.start(experiment_key=os.environ['EXPERIMENT_KEY'])
    else:
        experiment = comet_ml.start(workspace='anoaky',
                                    project_name='comp-550-project',
                                    experiment_config=expconfig)
        experiment_key = experiment.get_key()
        os.environ['EXPERIMENT_KEY'] = experiment_key
    experiment.disable_mp()
    comet_cb = CometCallback(experiment, prefix='sbf-transformer')
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'drop_last': True,
        'pin_memory': True,
        'prefetch_factor': 8
    }
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = SBFTransformer(tokenizer)
    train_loader, val_loader = model.train_dataloader(tokenizer, **dataloader_kwargs), model.val_dataloader(tokenizer, **dataloader_kwargs)
    fabric_summary = FabricSummary()
    fabric = L.Fabric(callbacks=[comet_cb, fabric_summary],
                      loggers=[],
                      precision="bf16-mixed",
                      strategy="fsdp")
    if fabric.is_global_zero:
        experiment.log_parameters({'batch_size': args.batch_size, 'deterministic': args.deterministic, 'seed': args.seed if args.deterministic else None})
    if args.deterministic:
        fabric.seed_everything(args.seed,
                               workers=True)
        torch.backends.cudnn.benchmark = False
    trainer = SBFTrainer(max_epochs=args.max_epochs,
                         log_every=args.log_every)
    fabric.launch(trainer.fit,
                  model,
                  train_loader,
                  val_loader)
    
if __name__ == '__main__':
    comet_ml.login()
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_argument('-n', '--experiment_name', required=True, type=str)
    parser.add_argument('-s', '--seed', default=-1, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--max_epochs', default=10, type=int)
    parser.add_argument('-w', '--num_workers', default=0, type=int)
    parser.add_argument('-l', '--log_every', default=100, type=int)
    args = parser.parse_args()
    if args.seed < 0:
        print("NON-DETERMINISTIC")
        args.deterministic = False
    else:
        print("DETERMINISTIC")
        args.deterministic = True
    main(args)
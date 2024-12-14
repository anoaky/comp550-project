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
from lightning.fabric.strategies.fsdp import FSDPStrategy
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
from typing import List
from trainer import CometCallback, FabricSummary
from tqdm import tqdm
from argparse import ArgumentParser
import evaluate
import os
import torch.distributed as dist
import torch.distributed.fsdp

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
        self.trained_epochs = 0
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
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
    
    def validation_step(self, input_ids, attention_mask, tgt):
        out = self.forward(input_ids, attention_mask, tgt)
        return out.loss
    
    def test_step(self, input_ids):
        out = self.generate(input_ids)
        out_seqs = out.sequences
        return out_seqs
    
    def train_dataloader(self, tokenizer: T5Tokenizer, **kwargs):
        train_ds = load_dataset('allenai/social_bias_frames', 
                                split='train', 
                                trust_remote_code=True)
        train_set = SBFPreprocessed(train_ds, tokenizer)
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)
        return train_loader
    
    def val_dataloader(self, tokenizer: T5Tokenizer, **kwargs):
        val_ds = load_dataset('allenai/social_bias_frames', 
                                split='validation', 
                                trust_remote_code=True)
        val_set = SBFPreprocessed(val_ds, tokenizer)
        val_loader = DataLoader(val_set, shuffle=False, **kwargs)
        return val_loader
    
    def test_dataloader(self, tokenizer: T5Tokenizer, **kwargs):
        test_ds = load_dataset('allenai/social_bias_frames', 
                                split='test', 
                                trust_remote_code=True)
        test_set = SBFPreprocessed(test_ds, tokenizer)
        test_loader = DataLoader(test_set, shuffle=False, **kwargs)
        return test_loader
    
class SBFTrainer:
    def __init__(self, *, max_epochs: int):
        self.max_epochs = max_epochs
        
    def fit(self, 
            fabric: L.Fabric, 
            model: L.LightningModule, 
            tokenizer: T5Tokenizer, 
            *, 
            train_loader, 
            val_loader, 
            test_loader,
            load_path=None):
        if fabric.is_global_zero:
            fabric.call("print_summary", module=model)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, 
                                        optimizer, 
                                        _reapply_compile=False)
        [train_loader, val_loader, test_loader] = fabric.setup_dataloaders(train_loader,
                                                                           val_loader,
                                                                           test_loader)
        state = {
            'model': model,
            'optimizer': optimizer
        }
        if load_path is not None:
            fabric.load(load_path, state) 
        t = tqdm()
        for epoch in range(self.max_epochs):
            model.trained_epochs += 1
            t.reset(total=len(train_loader))
            t.set_description(f'Training epoch {epoch}')
            model.train()
            with fabric.init_tensor():
                running_loss = torch.tensor(0.0)
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['post_ids']
                attn_mask = batch['post_attn']
                stype = batch['stype_ids']
                loss = model.training_step(input_ids, attn_mask, stype)
                fabric.backward(loss)
                running_loss += (loss / len(train_loader))
                optimizer.step()
                t.update()
            avg_loss = fabric.all_reduce(running_loss)
            if fabric.is_global_zero:
                fabric.call("log_metrics", 
                            train_loss=avg_loss.item(),
                            epoch=epoch)
            fabric.barrier()
            t.reset(total=len(val_loader))
            t.set_description(f'Validation epoch {epoch}')
            model.eval()
            with torch.no_grad():
                with fabric.init_tensor():
                    val_loss = torch.tensor(0.0)
                for idx, batch in enumerate(val_loader):
                    input_ids = batch['post_ids']
                    attn_mask = batch['post_attn']
                    stype = batch['stype_ids']
                    loss = model.validation_step(input_ids, attn_mask, stype)
                    val_loss += (loss / len(val_loader))
                    t.update()
                val_loss = fabric.all_reduce(val_loss)
                if fabric.is_global_zero:
                    fabric.call("log_metrics",
                                val_loss=val_loss.item(),
                                epoch=epoch)
        fabric.barrier()
        fabric.save(f't5-test-1-{model.trained_epochs}.ckpt', state)
        t.reset(total=len(test_loader))
        t.set_description(f'Testing')
        model.eval()
        with torch.no_grad():
            preds = []
            refs = []
            for idx, batch in enumerate(test_loader):
                input_ids = batch['post_ids']
                tgt_seqs = batch['stype_ids']
                out_seqs = model.test_step(input_ids)
                preds.append(out_seqs)
                refs.append(tgt_seqs)
                t.update()
            preds = torch.cat(preds, dim=1)
            refs = torch.cat(refs, dim=1)
            preds = fabric.all_gather(preds).view(-1, MAX_LENGTH)
            refs = fabric.all_gather(refs).view(-1, MAX_LENGTH)
            if fabric.is_global_zero:
                preds = tokenizer.batch_decode(preds,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)
                refs = tokenizer.batch_decode(refs,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
                refs = [[s] for s in refs]
                bleu = evaluate.load("bleu")
                results = bleu.compute(predictions=preds,
                                       referenes=refs)
                bleu_score = results['bleu']
                precisions = torch.tensor(results['precisions'])
                avg_precision = torch.mean(precisions)
                fabric.call("log_metrics",
                            precision=avg_precision.item(),
                            bleu=bleu_score)
        t.close()
        fabric.call("on_fit_end")

def main(args):
    expconfig = comet_ml.ExperimentConfig(#disabled=True,
                                          name=args.experiment_name,
                                          parse_args=False)
    if 'EXPERIMENT_KEY' in os.environ:
        experiment = comet_ml.start(experiment_key=os.environ['EXPERIMENT_KEY'])
    elif args.existing_experiment:
        experiment = comet_ml.start(experiment_key=args.experiment_key)
    elif args.no_experiment:
        dummyconfig = comet_ml.ExperimentConfig(disabled=True)
        experiment = comet_ml.start(experiment_config=dummyconfig)
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
    train_loader, val_loader, test_loader = (model.train_dataloader(tokenizer, **dataloader_kwargs), 
                                             model.val_dataloader(tokenizer, **dataloader_kwargs),
                                             model.test_dataloader(tokenizer, **dataloader_kwargs))
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
    trainer = SBFTrainer(max_epochs=args.max_epochs)
    fabric.launch(trainer.fit,
                  model,
                  tokenizer,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  test_loader=test_loader,
                  load_path=args.load_path)
    
if __name__ == '__main__':
    comet_ml.login()
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    excl_group = parser.add_mutually_exclusive_group(required=True)
    excl_group.add_argument('-n', '--experiment_name', default=None, type=str)
    excl_group.add_argument('-k', '--experiment_key', default=None, type=str)
    excl_group.add_argument('--no_experiment', action="store_true")
    parser.add_argument('-s', '--seed', default=-1, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--max_epochs', default=10, type=int)
    parser.add_argument('-w', '--num_workers', default=0, type=int)
    parser.add_argument('-l', '--load_path', default=None, type=str)
    args = parser.parse_args()
    if args.experiment_key is not None:
        args.existing_experiment = True
    else:
        args.existing_experiment = False
    if args.seed < 0:
        print("NON-DETERMINISTIC")
        args.deterministic = False
    else:
        print("DETERMINISTIC")
        args.deterministic = True
    main(args)
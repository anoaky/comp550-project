import comet_ml
from comet_ml.integration.pytorch import log_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prep_dataset(ds: Dataset) -> Dataset:
    def replace_empty(row):
        if row['targetCategory'] == '':
            row['targetCategory'] = 'none'
        return row
    return ds.select_columns(['post', 'targetCategory']).map(replace_empty).class_encode_column('targetCategory')


class Trainer:
    def __init__(self, experiment: comet_ml.Experiment, max_epochs, log_every):
        self.experiment = experiment
        self.max_epochs = max_epochs
        self.log_every = log_every

    def fit(self, model, /, train_loader, val_loader):
        best_accuracy = 0.0
        with tqdm() as t:
            model = model.to(device)
            optimizer = model.configure_optimizer()
            for epoch in range(self.max_epochs):
                t.reset(total=len(train_loader))
                t.set_description(f'Training Epoch {epoch}')
                model.train()
                with self.experiment.train():
                    for idx, batch in enumerate(train_loader):
                        for tid in range(len(batch)):
                            batch[tid] = batch[tid].to(device)
                        optimizer.zero_grad()
                        loss = model.train_step(idx, batch)
                        loss.backward()
                        optimizer.step()
                        if idx % self.log_every == self.log_every - 1:
                            self.experiment.log_metrics({'loss': loss.item()},
                                                        prefix=model.base_name,
                                                        step=idx+1,
                                                        epoch=epoch)
                        t.set_postfix(loss=loss.item())
                        t.update()
                t.reset(total=len(val_loader))
                t.set_description(f'Validation Epoch {epoch}')
                model.eval()
                with torch.no_grad(), self.experiment.validate():
                    running_accuracy = 0.0
                    for idx, batch in enumerate(val_loader):
                        for tid in range(len(batch)):
                            batch[tid] = batch[tid].to(device)
                        accuracy = model.val_step(idx, batch)
                        running_accuracy += accuracy
                        t.set_postfix(acc=running_accuracy /
                                      (idx + 1.0), best=best_accuracy)
                        t.update()
                    epoch_acc = running_accuracy / (len(val_loader) * 1.0)
                    self.experiment.log_metrics({'accuracy': epoch_acc},
                                                prefix=model.base_name,
                                                epoch=epoch)
                    if epoch_acc > best_accuracy:
                        best_accuracy = epoch_acc
                        log_model(self.experiment,
                                  model,
                                  model_name=f'{model.base_name}-epoch{epoch}-acc{epoch_acc:0.4f}')

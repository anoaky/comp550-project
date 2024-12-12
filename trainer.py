import comet_ml
from comet_ml.integration.pytorch import log_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities.model_summary import summarize

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prep_dataset(ds: Dataset) -> Dataset:
    def replace_empty(row):
        if row['targetCategory'] == '':
            row['targetCategory'] = 'none'
        return row
    return ds.select_columns(['post', 'targetCategory']).map(replace_empty).class_encode_column('targetCategory')


class FabricSummary:
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def print_summary(self, *, module: L.LightningModule):
        model_summary = summarize(module, max_depth=self.max_depth)
        ModelSummary.summarize(model_summary._get_summary_data(),
                               model_summary.total_parameters,
                               model_summary.trainable_parameters,
                               model_summary.model_size,
                               model_summary.total_training_modes)


class CometCallback:
    def __init__(self, *, prefix, **experiment_kwargs):
        self.experiment = comet_ml.start(**experiment_kwargs)
        self.experiment.disable_mp()
        self.prefix = prefix

    def log_metrics(self, *, epoch, step=None, **kwargs):
        self.experiment.log_metrics(
            kwargs, prefix=self.prefix, step=step, epoch=epoch)

    def on_fit_end(self):
        self.experiment.end()


class FabricTrainer:
    def __init__(self, fabric: L.Fabric, *, max_epochs: int, log_every: int):
        self.fabric = fabric
        self.max_epochs = max_epochs
        self.log_every = log_every

    def fit(self, model: L.LightningModule, *, train_loader: DataLoader, val_loader: DataLoader):
        self.fabric.call("print_summary", module=model)
        self.fabric.launch()
        optimizer = model.configure_optimizers()
        model, optimizer = self.fabric.setup(model, optimizer)
        [train_loader, val_loader] = self.fabric.setup_dataloaders(
            train_loader, val_loader)
        best_accuracy = 0.0
        t = tqdm()
        for epoch in range(self.max_epochs):
            t.reset(total=len(train_loader))
            t.set_description(f'Training epoch {epoch}')
            model.train()
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = model.training_step(*batch)
                self.fabric.backward(loss)
                optimizer.step()

                if idx % self.log_every == self.log_every - 1:
                    all_loss = self.fabric.all_reduce(loss)
                    if self.fabric.is_global_zero:
                        self.fabric.call(
                            "log_metrics", epoch=epoch, step=idx, train_loss=all_loss.item())
                    t.set_postfix(loss=all_loss.item())
                t.update()
            # validation
            t.reset(total=len(val_loader))
            t.set_description(f'Validation epoch {epoch}')
            model.eval()
            proc_accuracy = torch.tensor(0, device=self.fabric.device)
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    loss, accuracy = model.validation_step(*batch)
                    proc_accuracy = proc_accuracy + accuracy
                    t.update()
            proc_accuracy = proc_accuracy / len(val_loader)
            accuracy = self.fabric.all_reduce(proc_accuracy).item()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if self.fabric.is_global_zero:
                    self.fabric.call(
                        "log_metrics", epoch=epoch, val_acc=accuracy)
        t.close()
        self.fabric.call("on_fit_end")


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

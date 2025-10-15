import torch
import torch.nn as nn
import torch.nn.functional as F

from random import randint
import torch.optim.lr_scheduler
from torchmetrics import Accuracy
from lightning import pytorch as pl

from boda.model import BassetBranched


def get_default_params_A2G():
    params = dict(
            lr = 1e-3,
            wd = 1e-2,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 200*512,
            epochs = 200
            )
    return params
def get_default_params_G2A():
    params = dict(
            lr = 1e-4,
            wd = 1e-3,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 100*512,
            epochs = 100
            )
    return params


class Malinois(nn.Module):
    def __init__(self, n_outputs=1) -> None:
        super().__init__()

        self.model = BassetBranched(
            input_len=312,
            n_outputs=n_outputs,
            n_linear_layers=1,
            linear_channels=1000,
            linear_dropout_p = 0.11625456877954289,
            n_branched_layers = 3,
            branched_channels = 140,
            branched_activation = "ReLU",
            branched_dropout_p = 0.5757068086404574,
            # loss_criterion = "L1KLmixed",
            # loss_args={'beta': 5.0},
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # x is of shape (batch_size, seqlen, 4). pad with zeros to (batch_size, self.model.input_len, 4)
        assert x.shape[1] <= self.model.input_len, "sequence length must be less than or equal to self.model.input_len for Malinois"
        pad_size = self.model.input_len - x.shape[1]
        left_pad = pad_size // 2
        right_pad = pad_size - left_pad
        x = F.pad(x, (0, 0, left_pad, right_pad), mode='constant', value=0)

        x = x.permute(0, 2, 1) # (batch_size, 4, self.model.input_len)

        encoded = self.model.encode(x)
        decoded = self.model.decode(encoded)
        output = self.model.output(decoded).squeeze()
        return output
    


class MalinoisWrapper(pl.LightningModule):
    def __init__(self,
        hparams: dict = dict(
            lr = 1e-4,
            wd = 1e-4,
            div_fct = 1e1,
            final_div_fct = 1e4,
            num_steps = 1000,
        ),
        seed = None,
        criterion = nn.BCEWithLogitsLoss):
        
        super().__init__()
        if seed is not None:
            self.seed = seed
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            seed = randint(1, 100000)
            self.seed = seed
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        for key in hparams.keys():
            self.hparams[key] = hparams[key]
        self.save_hyperparameters()
        
        self.model = Malinois() #default hyperparams

        self.criterion = criterion()
        self.auroc = []

        
    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, (2 / n) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def compute_loss(self, batch):
        seqs, targets = batch
        pred = self.model(seqs)
        loss = self.criterion(pred, targets)
        with torch.no_grad():
            score = F.sigmoid(pred)
        return loss, score

    def training_step(self, batch, batch_idx):
        loss, preds = self.compute_loss(batch)
        actual_labels = (batch[1] > 0).float()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss, 'actual_labels' : actual_labels.numpy(force=True), 'pred_labels':preds.numpy(force=True)}

    def predict_step(self, batch, batch_idx):
        seqs = batch
        pred = self.model(seqs)
        return pred

    def validation_step(self, batch, batch_idx):
        loss,preds = self.compute_loss(batch)
        actual_labels = (batch[1] > 0).float()
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss, 'actual_labels' : actual_labels.numpy(force=True), 'pred_labels':preds.numpy(force=True)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'],
                                       weight_decay= self.hparams['wd'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= self.hparams['lr'],
                                                        total_steps= self.hparams['num_steps'], 
                                                        div_factor=self.hparams['div_fct'],
                                                        final_div_factor=self.hparams['final_div_fct'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": scheduler.__class__.__name__,
            },
        }

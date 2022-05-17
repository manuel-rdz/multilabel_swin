import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as op
from utils.metrics import get_scores, get_riadd_scores

from models.ml_swin import SwinTransformer


class MLSwinTransformer(pl.LightningModule):
    def __init__(self, img_size, lr, n_classes, use_lmt, backbone, depths, num_heads, window_size):
        super(MLSwinTransformer, self).__init__()

        self.model = SwinTransformer(img_size=img_size, use_lmt=use_lmt, num_classes=n_classes, backbone=backbone,
                                     depths=depths, num_heads=num_heads, window_size=window_size)

        self.n_classes = n_classes

        self.predictions = np.empty((0, n_classes), dtype=np.float32)
        self.target = np.empty((0, n_classes), dtype=np.int16)

        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()

        self.use_lmt = use_lmt

        self.best_score = 0.0

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_index):
        x, y, mask = batch

        b = x.size()
        x = x.view(b, -1)

        if self.use_lmt:
            logits = self(x, mask)
        else:
            logits = self(x)

        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        #print('y', y)
        #print('logits', logits)

        J = self.loss(logits, y)
        #print('loss', J)

        return {
            'loss': J}
        #    'train_acc': acc}
        #    'progress_bar': pbar}

    def validation_step(self, batch, batch_index):
        x, y, mask = batch

        b = x.size()
        x = x.view(b, -1)

        if self.use_lmt:
            preds = self(x, mask)
        else:
            preds = self(x)

        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        J = self.loss(preds, y)

        self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)
        self.target = np.concatenate((self.target, y.detach().cpu().numpy()), 0)

        # torch.sigmoid_(preds)

        # self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)

        # results = self.validation_step(batch, batch_idx)
        # results['test_acc'] = results['val_acc']
        # del results['val_acc']

        return {
            'val_loss': J,
        }

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()

        self.log("avg_val_loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # traditional metrics
        # model_score = get_scores(self.target, self.predictions)

        # RIADD scores
        model_score, _, _, _ = get_riadd_scores(self.target, self.predictions)

        if model_score[-1] > self.best_score:
            # f = open("best_swin.txt", "w")
            # f.write(np.array_str(model_score))
            # f.close()

            print('best_score:', model_score)

            self.best_score = model_score[-1]

        # clear preds and target
        self.clean_metrics_arrays()

        return {'avg_val_loss': avg_val_loss}
        # 'avg_val_acc': avg_val_acc} #, 'progress_bar': pbar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        lr_scheduler = op.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, threshold=0.001)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'avg_val_loss'}
    
    def clean_metrics_arrays(self):
        self.predictions = np.empty((0, self.n_classes), dtype=np.float32)
        self.target = np.empty((0, self.n_classes), dtype=np.int16)
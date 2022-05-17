import argparse

import numpy as np
import pandas as pd
import timm
import torch.nn
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from data.modules import RetinaDataModule
from pytorch_lightning import Trainer, seed_everything

from models import ml_swin_wrapper


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Training args')

parser.add_argument('--data_path', default='', help='path to data')
parser.add_argument('--img_path', nargs='+', default='', help='path to images')
parser.add_argument('--output_path', default='', help='path to output the model checkpoints')
parser.add_argument('--img_size', default=224, help='image size')
parser.add_argument('--use_lmt', default=False, help='use label mask training')
parser.add_argument('--num_classes', default=20, help='num classes to predict')
parser.add_argument('--backbone', default=False, help='use ResNet101 as feature extractor')
parser.add_argument('--depths', nargs='+', default=[2, 2, 6, 2], help='depths to use in the swin transformer')
parser.add_argument('--heads', nargs='+', default=[3, 6, 12, 24], help='num heads to use in the swin transformer')
parser.add_argument('--window_size', default=7, help='swin transformer window size')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--num_workers', default=4, help='num workers')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_model_name():
    name = ''
    if args.use_lmt:
        name += 'lmt_'

    if args.backbone:
        name += 'resnet101_'

    name += 'swin_' + str(args.img_size) + '_d' + ('-'.join(map(str, args.depths))) + '_h' + \
            '-'.join(map(str, args.heads)) + '_ws' + str(args.window_size)

    print(name)

    return name



def train_model(train_x, train_y, val_x, val_y):
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
    )

    early_stopping = EarlyStopping(
        monitor='avg_val_loss',
        patience=17,
        verbose=True,
        min_delta=0.001,
        mode='min')

    checkpoint = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=args.output_path,
        filename= get_model_name() + '-{epoch:02d}-{avg_val_loss:.3f}',
        save_top_k=1,
        mode="min",
    )

    if args.backbone:
        backbone = timm.create_model('resnet101', pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = True

        backbone.global_pool = torch.nn.Identity()
        backbone.fc = torch.nn.Identity()
    else:
        backbone=None

    # img_path = ['C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop',
    #             'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop',
    #             'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training']

    model = ml_swin_wrapper.MLSwinTransformer(
        img_size=args.img_size,
        use_lmt=args.use_lmt,
        n_classes=args.num_classes,
        lr=1e-5,
        backbone=backbone,
        depths=args.depths,
        num_heads=args.heads,
        window_size=args.window_size
    )

    trainer = Trainer(
        gpus=1,
        auto_lr_find=False,
        deterministic=True,
        precision=16,
        gradient_clip_val=0.5,
        callbacks=[checkpoint, lr_monitor, early_stopping],
        default_root_dir=args.output_path,
    )

    data_module = RetinaDataModule(
        df_train=train_x.join(train_y),
        df_val=val_x.join(val_y),
        train_img_path=args.img_path,
        val_img_path=args.img_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        start_col_labels=4,
        transforms=None,
    )

    #if args.auto_lr:
    #trainer.tune(model, datamodule=data_module)
    #    args.lr = model.lr
    #    args.auto_lr = False

    print('Using batch size: ', data_module.batch_size)
    print('Using learning rate: ', model.lr)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args, args_text = _parse_args()
    seed_everything(42, workers=True)

    data = pd.read_csv(args.data_path)

    folds = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_data = pd.DataFrame(np.empty(0))
    val_data = pd.DataFrame(np.empty(0))

    for (train_idx, val_idx) in folds.split(data, data.iloc[:, 4:]):
        train_data = data.iloc[train_idx, :]
        val_data = data.iloc[val_idx, :]
        break

    train_x = train_data.iloc[:, :4]
    train_y = train_data.iloc[:, 4:]

    val_x = val_data.iloc[:, :4]
    val_y = val_data.iloc[:, 4:]

    train_model(train_x, train_y, val_x, val_y)
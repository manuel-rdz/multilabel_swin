from data.datasets import MergedDataset
from torch.utils.data import DataLoader
from data.transforms import res_norm_transform

import pytorch_lightning as pl


class RetinaDataModule(pl.LightningDataModule):
    def __init__(self, df_train=None, df_val=None, df_test=None, train_img_path='', val_img_path='',
                 test_img_path='', img_size=224, batch_size=32, num_workers=4, pin_memory=False,
                 start_col_labels=1, stage='fit', use_tta=False, transforms='riadd'):
        super().__init__()

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.train_img_path = train_img_path
        self.val_img_path = val_img_path
        self.test_img_path = test_img_path

        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.start_col_labels = start_col_labels
        self.stage = stage
        self.use_tta = use_tta

        self.transforms = transforms

    # For distributed training, ran only once on single gpu
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.stage == 'fit' or self.stage is None:
            train_transform, val_transform = res_norm_transform(self.img_size), res_norm_transform(self.img_size)

            self.train_dataset = MergedDataset(image_ids=self.df_train, img_path=self.train_img_path,
                                               transform=train_transform, testing=False, start_col_labels=self.start_col_labels)
            self.val_dataset = MergedDataset(image_ids=self.df_val, img_path=self.val_img_path,
                                             transform=train_transform, testing=True, start_col_labels=self.start_col_labels)

        '''
        if self.stage == 'fit' or self.stage is None:
            train_transforms, valid_transforms = get_transformations(self.transforms, self.img_size)

            self.train_dataset = get_dataset(df_data=self.df_train, img_path=self.train_img_path,
                                             transforms=train_transforms, start_col=self.start_col_labels)
            self.val_dataset = get_dataset(df_data=self.df_val, img_path=self.val_img_path, transforms=valid_transforms,
                                           start_col=self.start_col_labels)

        if self.stage == 'test' or self.stage is None:
            _, test_transforms = get_transformations(self.transforms, self.img_size)
            self.test_dataset = get_dataset(df_data=self.df_test, img_path=self.test_img_path,
                                            transforms=test_transforms, start_col=self.start_col_labels)
        '''
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from data.utils import get_mask


class MergedDataset(data.Dataset):  # for training/testing
    def __init__(self, image_ids, img_path, transform, testing, start_col_labels=3):

        self.image_ids = image_ids
        self.img_path = img_path
        self.start_col_labels = start_col_labels

        self.transform = transform
        self.testing = testing

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        if self.image_ids.iloc[index, 1] == 1:  # ARIA
            imgId = str(imgId) + '.tif'
            dataset_idx = 0
        elif self.image_ids.iloc[index, 2] == 1:  # STARE
            imgId = str(imgId) + '.png'
            dataset_idx = 1
        elif self.image_ids.iloc[index, 3] == 1:  # RFMiD
            imgId = str(imgId) + '.png'
            dataset_idx = 2
        else:  # SYTHETIC
            imgId = str(imgId) + '.png'
            dataset_idx = 3

        label = self.image_ids.iloc[index, self.start_col_labels:].values.astype(np.int64)
        imgpath = os.path.join(self.img_path[dataset_idx], imgId)
        img = cv2.imread(imgpath)
        try:
            img = img[:, :, ::-1]
        except:
            print(imgpath)
        img = self.transform(image=img)['image']

        mask = torch.from_numpy(label).clone()
        unk_mask_indices = get_mask(self.testing, len(label))

        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        return img, label, mask

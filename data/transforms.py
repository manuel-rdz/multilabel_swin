import albumentations as alb
import numpy as np

from albumentations.pytorch import ToTensorV2


def res_norm_transform(image_size):
    return alb.Compose([
        alb.Resize(image_size, image_size),
        alb.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

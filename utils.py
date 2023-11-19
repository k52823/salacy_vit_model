# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import os.path as op
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import logging
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from timm.optim.optim_factory import create_optimizer
from types import SimpleNamespace
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import numpy as np
from timm.models.vision_transformer import VisionTransformer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

name_dict1 = {'Action': 0,
 'Affective': 1,
 'Art': 2,
 'BlackWhite': 3,
 'Cartoon': 4,
 'Fractal': 5,
 'Indoor': 6,
 'Inverted': 7,
 'Jumbled': 8,
 'LineDrawing': 9,
 'LowResolution': 10,
 'Noisy': 11,
 'Object': 12,
 'OutdoorManMade': 13,
 'OutdoorNatural': 14,
 'Pattern': 15,
 'Random': 16,
 'Satelite': 17,
 'Sketch': 18,
 'Social': 19}

name_dict = {'buildings': 0,
 'forest': 1,
 'glacier': 2,
 'mountain': 3,
 'sea': 4,
 'street': 5}
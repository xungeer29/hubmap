import numba, cv2, gc, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff 
import seaborn as sns
import rasterio
from rasterio.windows import Window
import pathlib, sys, os, random, time

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

import segmentation_models_pytorch as smp

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
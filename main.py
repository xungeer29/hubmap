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
from catalyst import dl, contrib
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T

import segmentation_models_pytorch as smp
import albumentations as A

from utils import seed_everything
from dataset import HubDataset, gen_transforms
from losses import DiceLoss, dice_coef


seed_everything()

DATA_PATH = 'data'
EPOCHES = 15
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 



identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

WINDOW = 256
MIN_OVERLAP = 32
#NEW_SIZE = 512


trans = gen_transforms()
ds = HubDataset(DATA_PATH, window=WINDOW, overlap=MIN_OVERLAP, transform=trans['train_transforms'])
d_train = D.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


model= smp.Unet(encoder_name="efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes = 1 )

model.to(DEVICE)
model.train()

base_optimizer = contrib.nn.RAdam([
    {'params': model.decoder.parameters(), 'lr':  5e-3,'weight_decay': 0.00003 }, 
    {'params': model.encoder.parameters(), 'lr':  5e-3/10, 'weight_decay': 0.00003},])

optimizer = contrib.nn.Lookahead(base_optimizer)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5)
diceloss = DiceLoss()
loss_list = [] 
dice_list = []
for epoch in tqdm(range(EPOCHES)):
    ###Train
    train_loss = 0
    for data in d_train:
        optimizer.zero_grad()
        img, mask = data
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        outputs = model(img)
        dice = dice_coef(outputs, mask)
        loss = diceloss(outputs, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        del mask, img, outputs
        gc.collect()
    train_loss /= len(d_train)
    print(f"EPOCH: {epoch + 1}/ : {EPOCHES}, train_loss: {train_loss}, dice:{dice}")
    loss_list.append(train_loss)
    dice_list.append(dice)
    scheduler.step(train_loss)
del d_train, ds
gc.collect()    
torch.save(model.state_dict(), 'model.pth')



plt.figure(figsize=(10,5))
n_e = np.arange(len(loss_list))
plt.plot(n_e, loss_list)
plt.title('Train loss ')
#plt.plot(n_e,history.history['val_dice_coe'],'-o',label='Val dice_coe',color='#1f77b4')

plt.figure(figsize=(10,5))
n_e = np.arange(len(dice_list))
plt.plot(n_e, dice_list)
plt.title('Train diece ')
#plt.plot(n_e,history.history['val_dice_coe'],'-o',label='Val dice_coe',color='#1f77b4')



WINDOW = 2024
MIN_OVERLAP = 32
NEW_SIZE = 512

p = pathlib.Path(DATA_PATH)
subm = {}
model.eval()

for i, filename in enumerate(p.glob('test/*.tiff')):
    dataset = rasterio.open(filename.as_posix(), transform = identity)
    slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
    preds = np.zeros(dataset.shape, dtype=np.uint8)
    for (x1,x2,y1,y2) in slices:
        image = dataset.read([1,2,3], window = Window.from_slices((x1,x2),(y1,y2)))
        image = np.moveaxis(image, 0, -1)
        image = trans['val_transforms'](image)
        with torch.no_grad():
            image = image.to(DEVICE)[None]
            score = model(image)[0][0]
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = cv2.resize(score_sigmoid, (WINDOW, WINDOW))
            
            preds[x1:x2,y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
            
    subm[i] = {'id':filename.stem, 'predicted': rle_numba_encode(preds)}
    del preds
    gc.collect()


submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)



# BASE_PATH = '../input/hubmap-kidney-segmentation/test'
# def plot_image(image_id, BASE_PATH = BASE_PATH, scale=None, verbose=1, df = submission):
#     image = tiff.imread(os.path.join(BASE_PATH, f"{image_id}.tiff"))
#     if len(image.shape) == 5:
#         image = image.squeeze().transpose(1, 2, 0)
    
#     mask = rle_decode(df[df["id"] == image_id]["predicted"].values[0], (image.shape[0],image.shape[1]))
#     if verbose:
#         print(f"[{image_id}] Image shape: {image.shape}")
#         print(f"[{image_id}] Mask shape: {mask.shape}")
    
#     if scale:
#         new_size = (image.shape[1] // scale, image.shape[0] // scale)
#         image = cv2.resize(image, new_size)
#         mask = cv2.resize(mask, new_size)
        
#         if verbose:
#             print(f"[{image_id}] Resized Image shape: {image.shape}")
#             print(f"[{image_id}] Resized Mask shape: {mask.shape}")
#     plt.figure(figsize=(32, 20))
#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title(f"Image {image_id}", fontsize=18)
    
#     plt.subplot(1, 3, 2)
#     plt.imshow(image)
#     plt.imshow(mask, cmap="hot", alpha=0.5)
#     plt.title("Image  + Predicted Mask", fontsize=18)    
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(mask, cmap="hot")
#     plt.title(f"Predicted Mask", fontsize=18)    
    
#     plt.show()
            
#     del image, mask

# for i in range(submission.shape[0]):    
#     img_id_1 = submission.iloc[i,0]
#     plot_image(img_id_1, scale = 20, verbose=1)





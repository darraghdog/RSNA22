import sys
import os
import json
import platform
import glob
import pandas as pd
import torch
import cv2
from PIL import Image
import numpy as np
import seaborn as sns
from bounding_box import bounding_box as bbfn
'''
PATH = '/Users/dhanley/Documents/RSNA22'
os.chdir(f'{PATH}')
'''

BASEDIR = '../rsna2022'
JPGDIR = '../rsna2022/datamount/train_images_jpg/'
grnred = np.array(sns.diverging_palette(10, 133, n = 101, as_cmap=False))
whtred = sns.color_palette("light:r", 101, as_cmap=False)
whtblu = sns.color_palette("light:b", 101, as_cmap=False)

valdf = pd.read_csv(f'{BASEDIR}/datamount/train_image_pseu_v04.csv.gz')
bbdf = pd.read_csv(f'{BASEDIR}/datamount/train_bbox_all_pred_v02.csv.gz')
stls = pd.Series(glob.glob(JPGDIR+'/*')).str.split('/').str[-1]
idx1 = valdf.filter(like='_frac').max(1)>0.95
idx2 = valdf.StudyInstanceUID.isin(stls)
stlsk = valdf[idx1&idx2].StudyInstanceUID.unique()

idx = 46
STNUM = stlsk[idx]
STNUM = '1.2.826.0.1.3680043.12145'
dd = valdf[valdf.StudyInstanceUID ==  STNUM].reset_index(drop = True)
print(idx, dd.filter(like='frac').max(0).values)

bbdf = bbdf[bbdf.StudyInstanceUID ==  STNUM].reset_index(drop = True)

imgls = []
fracclrls  = []
tmpimg = cv2.imread('figs/template.png')
tmpimg = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGB)
tmpimg = cv2.resize(tmpimg, (200,256))
Image.fromarray(tmpimg)

tdimg = cv2.imread('figs/3Dview.png')
tdimg = cv2.cvtColor(tdimg, cv2.COLOR_BGR2RGB)
tdimg = cv2.resize(tdimg, (256,256))

bb  = bbdf.query('has_bbox>0.6')['x0 y0'.split()].min(0).tolist() + \
          bbdf.query('has_bbox>0.6')['x1 y1'.split()].max(0).tolist()
x0,y0,x1,y1 = (np.array(bb)*512).round().astype(int)
for tt, (t,row) in enumerate(dd.iterrows()):
    img = cv2.imread(f'{JPGDIR}/{row.StudyInstanceUID}/{row.slice_number}.dcm.jpg')
    imgcrop = cv2.resize(img[y0:y1,x0:x1], (256,256))
    imgorig = cv2.resize(img, (256,256))
    Image.fromarray(img)
    Image.fromarray(imgcrop)
    frac_vals = row.filter(like = '_frac').values
    vert_vals = row.filter(like = '_pred').values
    # Set up prediction colors
    fracclr = [whtred[i] for i in (frac_vals*100).astype(int)]
    fracclr = (np.array(fracclr)*255).round().astype(np.uint8)
    fracclr = torch.tensor(fracclr).unsqueeze(1).repeat(1, 72, 1)
    fracclr = torch.repeat_interleave(fracclr, 30, 0).numpy()
    fracclrls.append(fracclr)
    
    vertclr = [whtblu[i] for i in (vert_vals*100).astype(int)]
    vertclr = (np.array(vertclr)*255).round().astype(np.uint8)
    vertclr = torch.tensor(vertclr).unsqueeze(1).repeat(1, 72, 1)
    vertclr = torch.repeat_interleave(vertclr, 30, 0).numpy()
    
    tmpl = tmpimg.copy()
    tmpl[-len(vertclr):,-144:-72,:] = vertclr
    tmpl[-len(vertclr):,-72:,:] = np.stack(fracclrls).min(0)
    
    # Mark 3d image
    tdimg1 = tdimg.copy()
    tdpos =  int(len(tdimg1) * (1 - tt/len(dd)))
    tdimg1[tdpos - 1: tdpos] = (255,255,0) 
    
    # Add bounding box
    bbfn.add(imgorig, x0//2,y0//2,x1//2,y1//2, color = 'green')
    
    imgout = np.concatenate((tdimg1, imgorig, imgcrop, tmpl), 1)
    if tt%2==0:
        imgls.append(imgout)
    #Image.fromarray(imgout).save(f'tmp/{str(tt).zfill(5)}.png')

# Save GIF
imgs = map(Image.fromarray,  imgls)
img = next(imgs)  # extract first image from iterator
img.save(fp='figs/study.gif', format='GIF', append_images=imgs,
         save_all=True, duration=40 * 2, loop=0)

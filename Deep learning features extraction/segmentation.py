from cellpose import utils, io, models
from cellpose import plot
from  skimage.measure import label,regionprops
import numpy as np
from tifffile import imread, imwrite
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib as mpl

def segmentation(img,flow_th,cell_th,channel,mode):
  model = models.Cellpose(gpu=True, model_type=mode)


# segment image
  masks, flows, styles, diams = model.eval(img, diameter=None, channels=channel,flow_threshold=flow_th,cellprob_threshold=cell_th)
  return masks

def plotmasks(masks):
  fig = plt.figure(figsize=(12,5))
  ax = fig.add_subplot()
  ax.imshow(masks,cmap="gray",origin="lower")
  ax.set_title("predicted masks")
  ax.axis("off")

def plotoverlays(img,masks):
# display results
 fig = plt.figure(figsize=(12,5))

 overlay = plot.mask_overlay(img, masks)
# plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
 ax = fig.add_subplot()
 ax.imshow(overlay,cmap="gray",origin="lower")
 ax.set_title("overlay")
 ax.axis("off")

"""
Source : 

* `pytorch-SRResNet <https://github.com/twtygqyy/pytorch-SRResNet>`_
"""

import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform

cuda = False
model = "model/model_epoch_415.pth"
image = "images/images1.jpg"
scale = 4


# resize the image
img_in = io.imread(image)
img = transform.resize(img_in, [224, 224])
new_image = image + ".224x224.jpg"
io.imsave(new_image, img)

# load the image
img = Image.open(new_image)
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

mg_gt, im_b, im_l = img_y, img_cb, img_cr

im_gt = im_gt.astype(float).astype(np.uint8)
im_b = im_b.astype(float).astype(np.uint8)
im_l = im_l.astype(float).astype(np.uint8)      


"""
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
"""

import torch
from torch.autograd import Variable
model = torch.load(model)["model"]
if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

im_input = im_l.astype(np.float32).transpose(2,0,1)
im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
im_input = Variable(torch.from_numpy(im_input/255.).float())

    
out = model(im_input)
out = out.cpu()

im_h = out.data[0].numpy().astype(np.float32)
im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.            
im_h = im_h.transpose(1,2,0)

print("Scale=",opt.scale)

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(Bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h.astype(np.uint8))
ax.set_title("Output(SRResNet)")
plt.show()
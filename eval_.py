# This is the evaluation code to output prediction using our saliency model.
#
# Author: Sen Jia
# Date: 09 / Mar / 2020
#
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import resnet

def normalize(img):
    img -= img.min()
    img /= img.max()
def gray(img_gray):
    hight,width = img_gray.shape
    for i in range(hight):
        for j in range(width):
        #img_gray的dtype是uint8,所有灰度值取值范围为0-255，
        #将大于128的变成1，小于128的变成0，就得到二值图的灰度值
            if img_gray[i,j] <= 45:
                img_gray[i,j] = 0
            else:
                img_gray[i,j] = 1
                print('======')
    return img_gray
def eval(model_path,img_path):
    preprocess = transforms.Compose([
        transforms.Resize(((640,480))),
	transforms.ToTensor(),
    ])

    model = resnet.resnet50(model_path)
    model.eval()

    pil_img = Image.open(img_path).convert('RGB')
    pil_img1 = Image.open(img_path)
    processed = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
       
        pred = model(processed)
        pred = pred.squeeze()
        normalize(pred)
        pred = pred.detach().cpu()
        
        
    img_gray = gray(np.array(pred))
    image_np = np.array(pil_img.resize((480,640)))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_gray)
    ax[0].set_title("Input Image")
    ax[1].imshow(image_np)
    ax[1].set_title("Prediction")
    
    plt.show()
    image = np.dstack((img_gray*image_np[:,:,0], img_gray*image_np[:,:,1], img_gray*image_np[:,:,2]))
    return image
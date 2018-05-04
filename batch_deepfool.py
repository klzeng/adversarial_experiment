"""Using deepfool algorithm to generate adversarial samples the code source:https://github.com/LTS4/DeepFool"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import sys

import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import pandas as pd
from  time import time
import os

net = models.resnet34(pretrained=True)

start = time()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A
# Switch to evaluation mode
net.eval()
#fpath = r'D:/onedrive/OneDrive - George Mason University/2018Spring/CS782/final_proj/dataset/dataset/'
fpath = r'/scratch/zfeng3/zhanpeng/images/'
csvpath = os.path.join(fpath, r'dev_dataset.csv')
labels = open(os.path.join(fpath, r'label.txt'), 'r').read().split('\n')
files = pd.read_csv(csvpath)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
image_path = os.path.join(fpath, r'images/')
file_names = files.ImageId.values
deep_ori_img = []
deep_after_img = []
ori_vs_after_label = []


for iter_i, image_name_path in enumerate(file_names):
    each_time = time()
    image_name_path = os.path.join(image_path, image_name_path)
    image_name_path += '.png'

    im_orig = Image.open(image_name_path)
    # Remove the mean
    im = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)])(im_orig)

    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

    deep_ori_img.append(im.data.numpy())
    deep_after_img.append(pert_image.cpu().numpy())
    str_label_orig = int(labels[np.int(label_orig)].split(',')[0])
    str_label_pert = int(labels[np.int(label_pert)].split(',')[0])

    print("Original label = ", labels[str_label_orig])
    print("Perturbed label = ", labels[str_label_pert])

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                             transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])
    ori_vs_after_label.append([str_label_orig, str_label_pert])
    save_img_tf = tf(pert_image.cpu()[0])
    save_fig_path = os.path.join(fpath, r'pertimg/')
    save_fig_path += str(iter_i) + r'.jpg'
    save_img_tf.save(save_fig_path)
    current = time()
    print("this img takes {} seconds".format(current - each_time))
    sys.stdout.flush()

deep_after_img = np.array(deep_after_img)
deep_ori_img = np.array(deep_ori_img)
ori_vs_after_label = np.array(ori_vs_after_label)

stop = time()
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import vgg19

from skimage.io import imread, imsave
from skimage import img_as_float, img_as_uint
from skimage.measure import compare_psnr
from skimage.transform import resize
import cv2

from artificial_bluring import blur_img

from copy import deepcopy
from models.instancenormrs import *

from torch.utils.data import Dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_transform(element):
    if isinstance(element, list):
        for i,each in enumerate(element):
            while len(each.shape) != 4:
                each = each[None,...]
            if not isinstance(each, torch.Tensor):
                each=torch.FloatTensor(each)
            if each.shape[1]!=3:
                each = each.transpose(1,3).transpose(2,3)
            element[i]=each
    elif isinstance(element, np.ndarray) or isinstance(element, torch.Tensor):
        while len(element.shape) != 4:
            element = element[None,...]
        if not isinstance(element, torch.Tensor):
            element = torch.FloatTensor(element)
        if element.shape[1]!=3:
            return element.transpose(1,3).transpose(2,3)
    return element

def input_transform(img, mode=None, level=None):
    assert isinstance(img,list), "input img should be in list (blurred_img_pyramid, blurred_img), (sharp_img_pyramid, sharp_img) or (blurred_img_pyramid, blurred_img)"
    if len(img)==2:
        blurredLP, blurred = img
    else:
        blurredLP, blurred, sharpLP, sharp = img
    if mode is None or mode=='deblurgan':
        if level is None:
            return blurred
        return blurredLP[level]
    elif mode == 'unet':
        blurredGP = get_gaussian_pyramid(np.transpose(blurred[0].numpy(), [1,2,0]))
        blurredGP = torch.FloatTensor(blurredGP[-level-1][None,...]).transpose(1,3).transpose(2,3)
        main_blurred = blurredLP[level]
        blurred_img = blurredGP.float()
        main_blurred = torch.cat([main_blurred, blurred_img],1)
        return main_blurred
    elif mode == 'resnet':
        main_blurred, aux_blurred = get_network_tensors(blurredLP, base_lvl=level)
        return main_blurred, aux_blurred

def readcv2(path, desired_shape=([360,480],[640])):
    imgr = imread(path)[...,:3]
    if imgr.ndim != 3:
        imgr = np.stack([imgr,imgr,imgr],axis=2)
        
    if not desired_shape is None:
        #print('sdsd')
        if imgr.shape[0] not in desired_shape[0] or imgr.shape[0] not in desired_shape[1]:
            d1 = np.argmin(np.abs(np.array(desired_shape[0])-imgr.shape[0]))
            d2 = np.argmin(np.abs(np.array(desired_shape[1])-imgr.shape[1]))
            imgr = resize(image=imgr, output_shape=(desired_shape[0][d1],desired_shape[1][d2]),preserve_range=True, 
                          mode='reflect')
    return np.uint8(imgr)

def get_gaussian_pyramid(img, depth=6):
    G = img.copy()
    gpA = [G]

    for i in range(depth):
        G = cv2.pyrDown(G)
        gpA.append(G)
        
    return gpA

def get_laplacian_pyramid(img, depth=4):
    G = img.copy()
    GP = get_gaussian_pyramid(G, depth)
    
    lpA = [GP[-1]]

    for i in range(len(GP)-1,0,-1):

        size = (GP[i-1].shape[1], GP[i-1].shape[0])
        GE = cv2.pyrUp(GP[i], dstsize = size)
        L = cv2.subtract(GP[i-1],GE)
        lpA.append(L)
    return lpA

def get_laplacian_pyramid_tensor(img, depth=3):
    lpA = get_laplacian_pyramid(img, depth)
    return list(map(lambda x: torch.FloatTensor(np.transpose(x, [2,0,1])), lpA))
    #out = list(map(lambda x: torch.FloatTensor(np.transpose(x, [2,0,1]))[None,...], lpA))
    #return list(map(lambda x: transforms(x), out))

def get_laplacian_pyramid_untensor(pyramid):
    return list(map(lambda x: np.transpose(np.array(x)[0,...], [1,2,0]), deepcopy(pyramid)))
    
def reconstructLP(LapP):
    rec = LapP[0].copy()
    for i in range(1,len(LapP)):
        size = (LapP[i].shape[1], LapP[i].shape[0])
        rec = cv2.pyrUp(rec.copy(), dstsize=size)
        rec = cv2.add(rec.copy(), LapP[i])
    return img_as_float(np.clip(rec,0,1))

class GOPRO_extended(Dataset):
    def __init__(self, basicfolder='/data/datasets/deblur/', train=True, include_sharp=0,
                 include_coco=None, returnLP=3, transform=None, desired_shape=([360,480],[480]), crop=None):
        
        self.basicfolder = basicfolder
        self.transform = transform
        self.train = train
        self.Tpaths = []
        self.Testpaths = []
        self.returnLP = returnLP
        self.include_coco = include_coco
        self.include_sharp = include_sharp
        self.crop = crop
        self.desired_shape = desired_shape
        train_path = self.basicfolder + 'train/'
        
        for each_folder in os.listdir(train_path):
            self.Tpaths += list(map(lambda x: train_path + each_folder + \
                                    '/sharp/'+x, os.listdir(train_path + each_folder + '/sharp')))
        
        test_path = self.basicfolder + 'test/'
        for each_folder in os.listdir(test_path):
            self.Testpaths += list(map(lambda x: test_path + each_folder + \
                                       '/sharp/'+x, os.listdir(test_path + each_folder + '/sharp')))
        
        if not self.include_coco is None:
            self.Tpaths += self.include_coco['train']
            self.Testpaths += self.include_coco['test']
                        
        if self.include_sharp:
            np.random.seed(123)
            self.sharp_paths_train = np.random.choice(self.Tpaths, size=self.include_sharp,replace=False)
            self.sharp_paths_test = np.random.choice(self.Testpaths, size=self.include_sharp,replace=False)
            np.random.seed(None)
            self.sharp_paths_train = list(map(lambda x: 's'+x, self.sharp_paths_train))
            self.sharp_paths_test = list(map(lambda x: 's'+x, self.sharp_paths_test))
            self.Tpaths += self.sharp_paths_train
            self.Testpaths += self.sharp_paths_test
        
    def do_transform(self,x,y):
        x,y = TF.to_pil_image(x), TF.to_pil_image(y)
        
        if np.random.rand() < 0.1:
            x,y = TF.to_grayscale(x,3), TF.to_grayscale(y,3)
        if np.random.rand() < 0.5:
            x,y = TF.hflip(x), TF.hflip(y)
        return x,y
    
    def do_crop(self, x,y):
        if not self.transform:
            x,y = TF.to_pil_image(x), TF.to_pil_image(y)
        j = np.random.randint(low=0, high=x.size[-2]-self.crop[1])
        i = np.random.randint(low=0, high=x.size[-1]-self.crop[0])
        x = TF.crop(x, h=self.crop[0], w=self.crop[1],i=i,j=j)
        y = TF.crop(y, h=self.crop[0], w=self.crop[1],i=i,j=j)
        return x,y
    
    def __len__(self):
        if self.train:
            return len(self.Tpaths)
        return len(self.Testpaths)
    
    def __getitem__(self, idx):
        
        if self.train:
            export_path = self.Tpaths[idx]
        else:
            export_path = self.Testpaths[idx]
        
        if export_path[0] == 's':
            #print('sharp')
            sharp = readcv2(export_path[1:], self.desired_shape)
            blurred = sharp.copy()
        else:
            sharp = readcv2(export_path, self.desired_shape)
            if export_path.find('coco')==-1:
                #print('gopro')
                blurred = readcv2(export_path.replace("/sharp","/blur/"), self.desired_shape)
            else:
                #print('coco')
                blurred = np.uint8(img_as_uint(blur_img(sharp)))
        
        if self.transform:
            blurred,sharp = self.do_transform(blurred, sharp)
        
        if not self.crop is None:
            blurred,sharp = self.do_crop(blurred, sharp)
        
        blurred = img_as_float(np.asarray(blurred))
        sharp = img_as_float(np.asarray(sharp))
        
        if not self.returnLP is None:
            blurred_LP = get_laplacian_pyramid_tensor(blurred, depth=self.returnLP)
            sharp_LP = get_laplacian_pyramid_tensor(sharp, depth=self.returnLP) 
            return (blurred_LP,blurred), (sharp_LP,sharp)
        
        blurred = np.transpose(blurred, [2,0,1])
        sharp = np.transpose(sharp, [2,0,1])
        
        return torch.FloatTensor(blurred), torch.FloatTensor(sharp)
    

def get_network_tensors(LP, base_lvl=-1):
    assert base_lvl, "Base_lvl can't be equal zero"
    base = deepcopy(LP[base_lvl])
    desired_shape = base.shape
    
    stacked = LP[0].repeat(1,1,desired_shape[-2]//LP[0].shape[-2],desired_shape[-1]//LP[0].shape[-1])

    for i, each_lvl in enumerate(LP):
        if not i:
            continue
            
        law_shape = each_lvl.shape
        if law_shape[-1]<desired_shape[-1]:
            repeated = each_lvl.repeat(1,1,desired_shape[-2]//law_shape[-2],desired_shape[-1]//law_shape[-1])
            stacked = torch.cat([stacked, repeated],1)
    return base, stacked

def get_12dim_tensor(LP, base_lvl):
    assert base_lvl, "Base_lvl can't be equal zero"
    lower = list()
    desired_shape = LP[base_lvl].shape
    out = deepcopy(LP)
    
    for i,each_lvl in enumerate(LP):
        if each_lvl.shape[-1]<desired_shape[-1]:
            law_shape = LP[i].shape
            out[i] = LP[i].repeat(1,1,desired_shape[-2]//law_shape[-2],desired_shape[-1]//law_shape[-1])
    return out

def stack_equal_dim(LP):
    (values,counts) = np.unique(np.array([p.shape[-1] for p in LP]),return_counts=True)
    mode = values[np.argmax(counts)]
    stacks, not_stacks = [p for p in LP if p.shape[-1]==mode], [p for p in LP if p.shape[-1]!=mode]
    stacked = stacks[0]
    for i,each_el in enumerate(stacks):
        if not i:
            continue
        stacked = torch.cat([stacked, each_el],1)
    return stacked, not_stacks

class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    
def extract_patches(img, kernel_size=(64,64)):
    list_of_crops = []
    ks0,ks1 = kernel_size[0], kernel_size[1]
    for i in range(img.size()[-2]//ks0): #480
        for j in range(img.size()[-1]//ks1): #640
            list_of_crops.append(img[...,i*ks0:(i+1)*ks0,j*ks1:(j+1)*ks1])
    return list_of_crops

class GOPRO_bp(Dataset):
    def __init__(self, basicfolder='/data/datasets/deblur/', train=True, include_sharp=0,
                 include_coco=None, include_blur_patterns=('/data/datasets/blur_patterns/',0),
                 returnLP=3, transform=False, desired_shape=([360,480],[480]), crop=None):
        """
        basicfolder - path to GOPRO folder
        train - train/test data switcher
        include_sharp - how many sharp photos from COCO include in training data. Will be used for test also
        include_coco - {'train':list, 'test':list}
        include_blur_patterns - 0: blur_patterns included, 1: blur_patterns included and mixed with GOPRO, 2: only blur_patterns included, will be both used in train/test
        returnLP - if not none will return LP of (int) depth
        transform - if True will apply transforms from do_transform
        desired_shape - if specified will transform to a specified shape ([360,480],[480])
        crop - will return image of specified (tuple) shape, if None will return original image
        """
        self.basicfolder = basicfolder
        self.transform = transform
        self.train = train
        self.Tpaths = []
        self.Testpaths = []
        self.returnLP = returnLP
        self.include_coco = include_coco
        self.blur_patterns_path = include_blur_patterns[0]
        self.blur_patterns_mode = include_blur_patterns[1]
        self.include_sharp = include_sharp
        self.crop = crop
        self.desired_shape = desired_shape
        train_path = self.basicfolder + 'train/'
        
        
        for each_folder in os.listdir(train_path):
            self.Tpaths += list(map(lambda x: train_path + each_folder + \
                                    '/sharp/'+x, os.listdir(train_path + each_folder + '/sharp')))
        
        test_path = self.basicfolder + 'test/'
        for each_folder in os.listdir(test_path):
            self.Testpaths += list(map(lambda x: test_path + each_folder + \
                                       '/sharp/'+x, os.listdir(test_path + each_folder + '/sharp')))
        
        if not self.include_coco is None:
            self.Tpaths += self.include_coco['train']
            self.Testpaths += self.include_coco['test']
                        
        if self.include_sharp:
            np.random.seed(123)
            self.sharp_paths_train = np.random.choice(self.Tpaths, size=self.include_sharp,replace=False)
            self.sharp_paths_test = np.random.choice(self.Testpaths, size=self.include_sharp,replace=False)
            np.random.seed(None)
            self.sharp_paths_train = list(map(lambda x: 's'+x, self.sharp_paths_train))
            self.sharp_paths_test = list(map(lambda x: 's'+x, self.sharp_paths_test))
            self.Tpaths += self.sharp_paths_train
            self.Testpaths += self.sharp_paths_test
            
        if self.blur_patterns_mode in (1,2):
            bp_idx = os.listdir(self.blur_patterns_path)
            bp_idx = list(filter(lambda x: 'R' in x, bp_idx))
            bp_idx = list(map(lambda x: self.blur_patterns_path+x, bp_idx))
            
            if self.blur_patterns_mode==1:
                self.Tpaths += bp_idx
                self.Testpaths += bp_idx
            elif self.blur_patterns_mode==2:
                self.Tpaths = bp_idx
                self.Testpaths = bp_idx
                
    def do_transform(self,x,y):
        x,y = TF.to_pil_image(x), TF.to_pil_image(y)
        
        if np.random.rand() < 0.1:
            x,y = TF.to_grayscale(x,3), TF.to_grayscale(y,3)
        if np.random.rand() < 0.5:
            x,y = TF.hflip(x), TF.hflip(y)
        return x,y
    
    def do_crop(self, x,y):
        if not self.transform:
            x,y = TF.to_pil_image(x), TF.to_pil_image(y)
        j = np.random.randint(low=0, high=x.size[-2]-self.crop[1])
        i = np.random.randint(low=0, high=x.size[-1]-self.crop[0])
        x = TF.crop(x, h=self.crop[0], w=self.crop[1],i=i,j=j)
        y = TF.crop(y, h=self.crop[0], w=self.crop[1],i=i,j=j)
        return x,y
    
    def __len__(self):
        if self.train:
            return len(self.Tpaths)
        return len(self.Testpaths)
    
    def __getitem__(self, idx):
        
        if self.train:
            export_path = self.Tpaths[idx]
        else:
            export_path = self.Testpaths[idx]
        
        if export_path[0] == 's':
            sharp = readcv2(export_path[1:], self.desired_shape)
            blurred = sharp.copy()
        else:
            sharp = readcv2(export_path, self.desired_shape)
            
            ###
            which_dataset = ('coco', 'blur_patterns', 'gopro')
            if export_path.find('coco')+1:
                which_dataset=which_dataset[0] #coco
            elif export_path.find('blur_patterns')+1:
                which_dataset=which_dataset[1] #bp
            else:
                which_dataset=which_dataset[2] #gopro
            ###
            
            if which_dataset=='gopro':
                #print('gopro')
                blurred = readcv2(export_path.replace("/sharp","/blur/"), self.desired_shape)
            elif which_dataset=='coco':
                #print('coco')
                blurred = np.uint8(img_as_uint(blur_img(sharp)))
            elif which_dataset=='blur_patterns':
                #print('blur_patterns')
                blurred = readcv2(export_path.replace("R","L"), self.desired_shape)
                
        if self.transform:
            blurred,sharp = self.do_transform(blurred, sharp)
        
        if not self.crop is None:
            blurred,sharp = self.do_crop(blurred, sharp)
        
        blurred = img_as_float(np.asarray(blurred))
        sharp = img_as_float(np.asarray(sharp))
        
        if not self.returnLP is None:
            blurred_LP = get_laplacian_pyramid_tensor(blurred, depth=self.returnLP)
            sharp_LP = get_laplacian_pyramid_tensor(sharp, depth=self.returnLP) 
            return (blurred_LP,blurred), (sharp_LP,sharp)
        
        blurred = np.transpose(blurred, [2,0,1])
        sharp = np.transpose(sharp, [2,0,1])
        
        return torch.FloatTensor(blurred), torch.FloatTensor(sharp)
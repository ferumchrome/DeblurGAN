import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import img_as_float, img_as_uint
from skimage.measure import compare_psnr

from artificial_bluring import blur_img

from copy import deepcopy
from utils import *

def test_deblurring_cur(net, test_dataset, lvl=[1], evaluate_metrics=True, shuffle_test=False, return_sharp_blurred=False):
    test = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=shuffle_test)
    (blurredLP, blurred), (sharpLP, sharp) = next(iter(test))
    blurredLP_orig = deepcopy(blurredLP)
    mark = True
    device = torch.device("cuda:0")
    
    for each_lvl in lvl:
        
        main_blurred, aux_blurred = get_network_tensors(blurredLP, base_lvl=each_lvl)
        
        if each_lvl == 1:
            #net['lvl1_net'].train(False)
            #net['lvl1_net'].eval()
            deblurred1 = net['lvl1_net'](Variable(main_blurred).to(device), Variable(aux_blurred).to(device))
            blurredLP_orig[each_lvl] = deblurred1.cpu().data
            mark=False
            
        if each_lvl == 2:
            #net['lvl2_net'].train(False)
            #net['lvl2_net'].eval()
            deblurred2 = net['lvl2_net'](Variable(main_blurred).to(device), Variable(aux_blurred).to(device))
            blurredLP_orig[each_lvl] = deblurred2.cpu().data
            mark=False
            
        if each_lvl == 3:
            #net['lvl3_net'].train(False)
            #net['lvl3_net'].eval()
            deblurred3 = net['lvl3_net'](Variable(main_blurred).detach().to(device), Variable(aux_blurred).detach().to(device))
            blurredLP_orig[each_lvl] = deblurred3.cpu().data
            mark=False
            
    LP_array = get_laplacian_pyramid_untensor(blurredLP_orig)
    reconstructed = reconstructLP(LP_array)
    
    if mark:
        print('No deblurring done!')
        
    if evaluate_metrics:
        
        sharp = img_as_float(np.array(sharp)[0,...])
        PSNR = compare_psnr(im_true=sharp, im_test=reconstructed)
        
        MSEs = dict.fromkeys(lvl, None)
        LP_array_sharp = get_laplacian_pyramid_untensor(sharpLP)
        
        for each_lvl in lvl:
            lb, ls = LP_array[each_lvl], LP_array_sharp[each_lvl] 
            MSEs[each_lvl] = ((lb - ls)**2).mean()
            
    if return_sharp_blurred:
        blurred = img_as_float(np.array(blurred)[0,...])
        return (reconstructed,sharp,blurred), PSNR, MSEs
    
    return reconstructed,PSNR, MSEs

def test_deblurring_unet(net, test_dataset, lvl=[1], evaluate_metrics=True, shuffle_test=False, return_sharp_blurred=False, full=True):
    test = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=shuffle_test)
    (blurredLP, blurred), (sharpLP, sharp) = next(iter(test))
    blurredLP_orig = deepcopy(blurredLP)
    mark = True
    device = torch.device("cuda:0")
    
    for each_lvl in lvl:
        
        main_blurred = blurredLP[each_lvl]
        
        if each_lvl == 1:
            #net['lvl1_net'].train(False)
            #net['lvl1_net'].eval()
            deblurred1 = net['lvl1_net'](Variable(main_blurred).to(device))
            blurredLP_orig[each_lvl] = deblurred1.cpu().data
            mark=False
            
        if each_lvl == 2:
            #net['lvl2_net'].train(False)
            #net['lvl2_net'].eval()
            deblurred2 = net['lvl2_net'](Variable(main_blurred).to(device))
            blurredLP_orig[each_lvl] = deblurred2.cpu().data
            mark=False
            
        if each_lvl == 3:
            #net['lvl3_net'].train(False)
            #net['lvl3_net'].eval()
            if full:
                blurred_img = blurred.transpose(dim0=2, dim1=3).transpose(dim0=1, dim1=2).float()
                main_blurred = torch.cat([main_blurred, blurred_img],1)
            deblurred3 = net['lvl3_net'](Variable(main_blurred).detach().to(device))
            blurredLP_orig[each_lvl] = deblurred3.cpu().data
            mark=False
            
    LP_array = get_laplacian_pyramid_untensor(blurredLP_orig)
    reconstructed = reconstructLP(LP_array)
    
    if mark:
        print('No deblurring done!')
        
    if evaluate_metrics:
        
        sharp = img_as_float(np.array(sharp)[0,...])
        PSNR = compare_psnr(im_true=sharp, im_test=reconstructed)
        
        MSEs = dict.fromkeys(lvl, None)
        LP_array_sharp = get_laplacian_pyramid_untensor(sharpLP)
        
        for each_lvl in lvl:
            lb, ls = LP_array[each_lvl], LP_array_sharp[each_lvl] 
            MSEs[each_lvl] = ((lb - ls)**2).mean()
            
    if return_sharp_blurred:
        blurred = img_as_float(np.array(blurred)[0,...])
        return (reconstructed,sharp,blurred), PSNR, MSEs
    
    return reconstructed,PSNR, MSEs


def test_deblurring_full(net, test_dataset, lvl=[1], evaluate_metrics=True, shuffle_test=False, return_sharp_blurred=False):
    test = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=shuffle_test)
    (blurredLP, blurred), (sharpLP, sharp) = next(iter(test))
    blurredLP_orig = deepcopy(blurredLP)
    
    device = torch.device("cuda:0")
    
    for each_lvl in lvl:
        
        dim_tensor = get_12dim_tensor(blurredLP, base_lvl=each_lvl)
        var_main, var_aux = stack_equal_dim(dim_tensor)
        
        if each_lvl == 1:
            deblurred1 = net['lvl3_net'](Variable(var_main).to(device), Variable(var_aux[0]).to(device), 
                                         Variable(var_aux[1]).to(device))
            blurredLP_orig[each_lvl] = deblurred1.cpu().data
            
        if each_lvl == 2:
            deblurred2 = net['lvl2_net'](Variable(var_main).to(device), Variable(var_aux[0]).to(device))
            blurredLP_orig[each_lvl] = deblurred2.cpu().data
            
        if each_lvl == 3:
            deblurred3 = net['lvl1_net'](Variable(var_main).to(device))
            blurredLP_orig[each_lvl] = deblurred3.cpu().data
            
    LP_array = get_laplacian_pyramid_untensor(blurredLP_orig)
    reconstructed = reconstructLP(LP_array)
    
    if evaluate_metrics:
        
        sharp = img_as_float(np.array(sharp)[0,...])
        PSNR = compare_psnr(im_true=sharp, im_test=reconstructed)
        
        MSEs = dict.fromkeys(lvl, None)
        LP_array_sharp = get_laplacian_pyramid_untensor(sharpLP)
        
        for each_lvl in lvl:
            lb, ls = LP_array[each_lvl], LP_array_sharp[each_lvl] 
            MSEs[each_lvl] = ((lb - ls)**2).mean()
            
    if return_sharp_blurred:
        blurred = img_as_float(np.array(blurred)[0,...])
        return (reconstructed,sharp,blurred), PSNR, MSEs
    
    return reconstructed,PSNR, MSEs
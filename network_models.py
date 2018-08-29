import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
import functools

from copy import deepcopy
#from models.instancenormrs import *

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.LeakyReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, input_enc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], use_parallel = True, learn_residual = False, padding_type='reflect', partial_downsample=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.partial_downsample = partial_downsample
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_in = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                              bias=use_bias),
                    norm_layer(ngf),
                    nn.LeakyReLU(True)]
        
        encoder = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_enc, ngf, kernel_size=7, padding=0,
                             bias=use_bias),
                   norm_layer(ngf),
                   nn.LeakyReLU(True)]
        
        model = []
        if self.partial_downsample:
            n_downsampling = 1
        else:
            n_downsampling = 2
            
        for i in range(n_downsampling):
            mult = 2**i
            if not i:
                model += [nn.Conv2d(ngf * 2, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.encoder = nn.Sequential(*encoder)
        self.model_in = nn.Sequential(*model_in)
        self.model = nn.Sequential(*model)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def forward(self, input, previous):
        current = self.model_in(input)
        previous = self.encoder(previous)
        combined = torch.cat([current, previous],dim=1)
        
        if self.gpu_ids and isinstance(combined.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, combined, self.gpu_ids)
        else:
            output = self.model(combined)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1, max = 1)
        return output
    
    
class ResnetGeneratorFull(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], use_parallel = True, learn_residual = False, padding_type='reflect', partial_downsample=False):
        assert(n_blocks >= 0)
        super(ResnetGeneratorFull, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.partial_downsample = partial_downsample
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(True)]
        
        if self.partial_downsample:
            n_downsampling = 1
        else:
            n_downsampling = 2
            
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
        #print(self.model)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output
    
class netG_1(nn.Module):
    def __init__(self, encoder_args, gen_args, init=None):
        assert(gen_args['n_blocks'] >= 0)
        assert(encoder_args['output_nc'] == gen_args['input_nc'])
        super(netG_1, self).__init__()
        
        self.use_parallel = gen_args['use_parallel']
        
        self.netG_base = ResnetGeneratorFull(**gen_args)
        
        if not init is None:
            self.netG_base.load_state_dict(torch.load(init))
        
        norm_layer = encoder_args['norm_layer']
        
        self.encode = nn.Sequential(
            nn.Conv2d(encoder_args['input_nc'], encoder_args['output_nc'], kernel_size=3, padding=1, bias=False),
            norm_layer(encoder_args['output_nc']),
            nn.LeakyReLU()
        )
        
        self.encodeF = nn.Sequential(
            nn.Conv2d(encoder_args['output_nc'], encoder_args['output_nc'], kernel_size=3, padding=1, bias=False),
            norm_layer(encoder_args['output_nc']),
            nn.LeakyReLU()
        )
        
    def forward(self, input):
        x = self.encodeF(self.encode(input))
        return self.netG_base(x)
    
class netG_2(nn.Module):
    def __init__(self, encoder_args, gen_args, init=None):
        assert(gen_args['n_blocks'] >= 0)
        assert(encoder_args['output_nc'] == gen_args['input_nc'])
        super(netG_2, self).__init__()
        
        self.use_parallel = gen_args['use_parallel']
        
        self.netG_base = ResnetGeneratorFull(**gen_args)
        
        if not init is None:
            self.netG_base.load_state_dict(torch.load(init))
        
        norm_layer = encoder_args['norm_layer']
        
        
        self.enconvH = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(3),
            nn.LeakyReLU()
        )
        
        self.enconvF = nn.Sequential(
            nn.Conv2d(encoder_args['output_nc'], encoder_args['output_nc'], kernel_size=3, padding=1, bias=False),
            norm_layer(encoder_args['output_nc']),
            nn.LeakyReLU()
        )
                
    def forward(self, input, inputH):
        xh = self.enconvH(inputH)
        x = self.enconvF(torch.cat([input, xh], dim=1))
        return self.netG_base(x)
    
class netG_3(nn.Module):
    def __init__(self, encoder_args, gen_args, init=None):
        assert(gen_args['n_blocks'] >= 0)
        assert(encoder_args['output_nc'] == gen_args['input_nc'])
        super(netG_3, self).__init__()
        
        self.use_parallel = gen_args['use_parallel']
        
        self.netG_base = ResnetGeneratorFull(**gen_args)
        
        if not init is None:
            self.netG_base.load_state_dict(torch.load(init))
        
        norm_layer = encoder_args['norm_layer']
        
        self.enconvH1_1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(3),
            nn.LeakyReLU()
        )
        
        self.enconvH1_2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(3),
            nn.LeakyReLU()
        )
        
        
        self.enconvH2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(3),
            nn.LeakyReLU()
        )
        
        self.enconvF = nn.Sequential(
            nn.Conv2d(encoder_args['output_nc'], encoder_args['output_nc'], kernel_size=3, padding=1, bias=False),
            norm_layer(encoder_args['output_nc']),
            nn.LeakyReLU()
        )
        
    def forward(self, input, inputH1, inputH2):
        xh1 = self.enconvH1_2(self.enconvH1_1(inputH2))
        xh2 = self.enconvH2(inputH1)
        x = self.enconvF(torch.cat([input, xh1, xh2], 1))
        return self.netG_base(x)
    
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_parallel = True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
        
class NLayerDiscriminatorRF(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_parallel = True):
        super(NLayerDiscriminatorRF, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1# int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(True),
            nn.Dropout()
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(True),
                nn.Dropout()
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(True),
        ]
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
import torch
import torch.nn as nn

class downstream_block(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, norm_layer=None, kernel_dim=3, kernel_padding=1, ceil_mode=False):
        super(downstream_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_dim, padding=kernel_padding),
            norm_layer(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            
            nn.Conv2d(out_dim, out_dim, kernel_dim, padding=kernel_padding),
            norm_layer(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=2, padding=1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def forward(self, x):
        return self.block(x)
    
class upstream_block(nn.Module):
    def __init__(self, dim_x=None, dim_y=None, out_dim=None, norm_layer=nn.BatchNorm2d, 
                 kernel_dim=3, kernel_padding=1, size=None):
        super(upstream_block, self).__init__()
        
        self.up = nn.ConvTranspose2d(dim_x, dim_x, kernel_size=4,stride=2,padding=1)
        
        self.block = nn.Sequential(
            nn.Conv2d(dim_x+dim_y, out_dim, kernel_dim, padding=kernel_padding),
            norm_layer(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            
            nn.Conv2d(out_dim, out_dim, kernel_dim, padding=kernel_padding),
            norm_layer(out_dim),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def forward(self,x,y):
        x=self.up(x)
        return self.block(torch.cat([x,y],1))
    
class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, learn_residual = False):
        super(UNetGenerator, self).__init__()
        
        self.learn_residual = learn_residual
        
        self.inconv = nn.Sequential(
            
            nn.Conv2d(input_nc, ngf,kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.LeakyReLU(),
            nn.Dropout(),
            
            nn.Conv2d(ngf, ngf,kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        
        self.outconv = nn.Sequential(
            
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf, output_nc, kernel_size=1,stride=1),
            nn.Tanh()
        )
        
        self.dw1 = downstream_block(ngf, ngf, norm_layer=norm_layer) #256 -> 128 |64 -> 64
        self.dw2 = downstream_block(ngf, ngf*2,norm_layer=norm_layer) #128 -> 64 |64 -> 128
        self.dw3 = downstream_block(ngf*2, ngf*4,norm_layer=norm_layer) #64 -> 32|128 -> 256
        self.dw4 = downstream_block(ngf*4, ngf*8,norm_layer=norm_layer) #32 -> 16|256 -> 512
        self.dw5 = downstream_block(ngf*8, ngf*16,norm_layer=norm_layer) #16 -> 8|512 -> 1024
        
        self.up1 = upstream_block(ngf*16,ngf*8, ngf*8, norm_layer=norm_layer)
        self.up2 = upstream_block(ngf*8, ngf*4, ngf*4, norm_layer=norm_layer)
        self.up3 = upstream_block(ngf*4, ngf*2, ngf*2, norm_layer=norm_layer)
        self.up4 = upstream_block(ngf*2, ngf,   ngf, norm_layer=norm_layer)
        self.up5 = upstream_block(ngf,   ngf,   ngf, norm_layer=norm_layer)
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def forward(self,input):
        
        x_in = self.inconv(input)
        
        x1 = self.dw1(x_in)
        x2 = self.dw2(x1)
        x3 = self.dw3(x2)
        x4 = self.dw4(x3)
        x5 = self.dw5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x_in)
        
        output = self.outconv(x)
        
        if self.learn_residual:
            output = input[:,:3] + output
            output = torch.clamp(output,min = -1, max = 1)
        return output
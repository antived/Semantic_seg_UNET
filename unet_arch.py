import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as tf

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dconv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = 1,padding = 1,bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size = 3, stride = 1,padding = 1,  bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dconv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_chans= 1, out_chans = 1, depths = [64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.ups = nn.ModuleList()  #we are going to traverse through these lists basically.
        self.downs = nn.ModuleList()
        self.depths = depths
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        ###in_channels = in_chans
        for depth in depths:
            self.downs.append(DoubleConv(self.in_chans, depth))
            self.in_chans = depth

        for depth in reversed(depths):
            self.ups.append(
                nn.ConvTranspose2d(2*depth, depth, kernel_size = 2, stride = 2)  #this is used for the means of upsampling by a factor of 2.
            )
            self.ups.append(DoubleConv(2*depth,depth))

        self.bottom_layer = DoubleConv(512,1024)

        self.final_conv = nn.Conv2d(depths[0], out_chans, kernel_size = 1, stride = 1)

    def forward(self,x):
        skip_links = []
        for down in self.downs:
            x = down(x)
            skip_links.append(x)
            x = self.pool(x)

        x = self.bottom_layer(x) 

        skip_links = skip_links[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            concat_skip = skip_links[i//2]
            #now we have to concat the skip link.
            if x.shape != concat_skip.shape:
                x = tf.resize(x, size = concat_skip.shape[2:])
            concat_skip = torch.cat((concat_skip,x), dim = 1)
            x = self.ups[i+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x)) #the corresponding loss function is going to basically now be BCE.


def test():
    x = torch.randn((3,1,161,161))
    model = Unet(in_chans = 1, out_chans = 1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert x.shape == preds.shape

if __name__ == "__main__" :
    test()

        

        

        

        
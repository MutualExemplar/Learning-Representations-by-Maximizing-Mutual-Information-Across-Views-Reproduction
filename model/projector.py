import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super(Projector, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [batch_size, channels, height, width], but got {x.shape}")

        x = self.conv1(x)

        # Prevent pooling if spatial size is too small
        if x.shape[2] > 2 and x.shape[3] > 2:
            x = self.pool(x) 

        x = self.conv2(x)

        if x.shape[2] > 2 and x.shape[3] > 2:
            x = self.pool(x)

        x = self.conv3(x)

        if x.shape[2] > 2 and x.shape[3] > 2:
            x = self.pool(x)

        return x

  
    
class classifier(nn.Module):
    def __init__(self, inp_dim = 256,ndf=128, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
        
    def forward(self,input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        x_1 = self.pool(x_1)
        # x_out = self.linear(x_1)
        x_out = self.final(x_1)
        return x_out
   
      
            
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torchsummary import summary # type: ignore 

class DoubleConv(nn.Module):
    "2 * [conv2d -> BN -> Relu]"
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__() # no arguments inside () because __init__ in class Module expects no arguments
        if not mid_channels:
            mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace = True),
                nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = False)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    "max_pool -> double conv"
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)
    
class Up(nn.Module):
    "upscale -> double conv"
    "there are two types of upscaling"
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim = 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)
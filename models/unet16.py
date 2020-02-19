"""
16-layer U-Net model
"""
import torch.nn as nn
from models.unet_tools import UNetDown, UNetUp, ConvSig, FCNN

class unet_vgg16(nn.Module):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """
    def __init__(self, inp_ch, kernel_size=3, adversary='no', num_cat=11, num_vid=53, skip=True):
        super(unet_vgg16, self).__init__()
        self.adversary = adversary
        self.skip = skip
        self.enc1 = UNetDown(inp_ch, 64, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(64, 128, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(256, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(512, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.dec4 = UNetUp(512, skip*512, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip*256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip*128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(64)

    def forward(self, inp, alpha=0):
        """
        Args:
            inp (tensor) :              Tensor of input Minibatch
            alpha (float or [float]):   Multiplier for gradient reversal layer(s) only when "*dann" is selected as adversary

        Returns:
            (tensor): Change detection output
            (tensor): Domain output. Will not be returned when self.adversarial="no"
        """
        d1 = self.enc1(inp)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        d5 = self.enc5(d4)
        if self.skip:
            u4 = self.dec4(d5, d4)
            u3 = self.dec3(u4, d3)
            u2 = self.dec2(u3, d2)
            u1 = self.dec1(u2, d1)
        else:
            u4 = self.dec4(d5)
            u3 = self.dec3(u4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)
            
        cd_out = self.out(u1)
        return cd_out

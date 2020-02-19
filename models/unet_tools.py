import torch
import torch.nn as nn

class UNetDown(nn.Module):
    """ Encoder blocks of UNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
        maxpool (booelan): Whether to apply max pool in the beginning
    """
    def __init__(self, in_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False, maxpool=False):
        super(UNetDown, self).__init__()
        self.down_block = nn.Sequential()

        if maxpool:
            self.down_block.add_module("maxpool", nn.MaxPool2d(2))
        in_ch_for_conv = in_ch
        for k in range(num_rep):
            self.down_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.down_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.down_block.add_module("bn%d"%(k+1), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.down_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp):
        return self.down_block(inp)

class UNetUp(nn.Module):
    """ Decoder blocks of UNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        res_ch (int): Number of channels coming from the residual, if equal to 0 and no skip connections
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
    """

    def __init__(self, in_ch, res_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False):

        super(UNetUp, self).__init__()
        self.up = nn.Sequential()
        self.conv_block = nn.Sequential()

        self.up.add_module("conv2d_transpose", nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=2,
                                                                  output_padding=(int((kernel_size-1)/2)),
                                                                  padding=(int((kernel_size-1)/2))))
        if batch_norm:
            self.up.add_module("bn1", nn.BatchNorm2d(in_ch))

        in_ch_for_conv = in_ch + res_ch
        for k in range(num_rep):
            self.conv_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.conv_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.conv_block.add_module("bn%d"%(k+2), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp, res=None):
        """
        Args:
            inp (tensor): Input tensor
            res (tensor): Residual tensor to be merged, if res=None no skip connections
        """
        feat = self.up(inp)
        if res is None:
            merged = feat
        else:
            merged = torch.cat([feat, res], dim=1)
        return self.conv_block(merged)

class ConvSig(nn.Module):
    """ Conv layer + Sigmoid

    Args:
        in_ch (int): Number of input channels
    """

    def __init__(self, in_ch):
        super(ConvSig, self).__init__()
        self.out = nn.Sequential()
        self.out.add_module("conv2d", nn.Conv2d(in_ch, 1, 1))
        self.out.add_module("sigmoid", nn.Sigmoid())

    def forward(self, inp):
        return self.out(inp)

class FCNN(nn.Module):
    """ Fully connected Neural Network with Softmax in the end

        Args:
            sizes ([int]): Sizes of the layers starting from input till the output
    """

    def __init__(self, sizes):
        super(FCNN, self).__init__()
        self.fcnn = nn.Sequential()
        for k, (in_ch, out_ch) in enumerate(zip(sizes[:-2], sizes[1:-1])):
            self.fcnn.add_module("fc%d" %(k+1), nn.Linear(in_ch, out_ch))
            self.fcnn.add_module("bn%d" %(k+1), nn.BatchNorm1d(out_ch))
            self.fcnn.add_module("relu%d" %(k+1), nn.ReLU(True))
        self.fcnn.add_module("fc%d" %(len(sizes)-1), nn.Linear(sizes[-2], sizes[-1]))
        self.fcnn.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, inp):
        return self.fcnn(inp)

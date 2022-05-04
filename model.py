import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from utils import save_net,load_net
from layer import convDU,convLR
from masksembles.torch import Masksembles2D

class SFCN(nn.Module):
    def __init__(self, load_weights=False):
        super(SFCN, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512,'MASK', 512, 512, 256, 128, 64]
        self.backend_feat2  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,batch_norm=False, dilation = True)
        self.backend2 = make_layers(self.backend_feat2,in_channels = 512,batch_norm=False, dilation = True)
        self.adpool = nn.AdaptiveAvgPool2d((96,128))
        self.fc = nn.Linear(64*96*128, 2)
        self.convDU = convDU(in_out_channels=64,kernel_size=(1,9))
        self.convLR = convLR(in_out_channels=64,kernel_size=(9,1))
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),nn.ReLU())
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
	    # address the mismatch in key names
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self,x):
        x_share = self.frontend(x)

        x = self.backend(x_share)
        x = self.convDU(x)
        x = self.convLR(x)
        x = self.output_layer(x)
        x = F.upsample(x,scale_factor=8)

        x_class = self.backend2(x_share)
        x_class = self.adpool(x_class)
        x_class = torch.flatten(x_class,1)
        x_class = self.fc(x_class)
        return x,x_class

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MASK':
            layers +=[Masksembles2D(512,3,2.0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

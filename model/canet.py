import torch.nn as nn

from .drn import drn_d_22, drn_d_38
from .attention import ChannelAttentionModule


class CANet(nn.Module):
    def __init__(self, config, inter_channel=512):
        super().__init__()

        # set a base model
        if config.model == 'drn_d_22':
            print('Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif config.model == 'drn_d_38':
            print('Dilated ResNet D 38 wil be used as a base model')
            self.base = drn_d_38(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        else:
            print('There is no option you choose as a base model.')
            print('Instead, Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])

        # convolution before attention module
        self.conv2cam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        # attention modules
        self.cam = ChannelAttentionModule()

        # convolution after attention modules
        self.cam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())

        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, config.n_classes, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)

        # outputs from attention modules
        cam_out = self.conv2cam(x)
        cam_out = self.cam(cam_out)
        cam_out = self.cam2conv(cam_out)

        output = self.conv_out(cam_out)

        return output

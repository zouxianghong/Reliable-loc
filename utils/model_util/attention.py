import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttentionModule(nn.Module):
    """ this function is used to achieve the channel attention module in CBAM paper"""
    def __init__(self, C, ratio=8):  # getting from the CBAM paper, ratio=16
        super(ChannelAttentionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=C // ratio, out_channels=C, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ x: B x C x N"""
        out1 = torch.mean(x, dim=-1, keepdim=True)  # b, c, 1
        out1 = self.mlp(out1)  # b, c, 1
        out2 = nn.AdaptiveMaxPool1d(1)(x)  # b, c, 1
        out2 = self.mlp(out2)  # b, c, 1
        out = self.sigmoid(out1 + out2)
        return out * x


class SpatialAttentionModule(nn.Module):
    """ this function is used to achieve the spatial attention module in CBAM paper"""
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_attn=False):
        """ x: B x C x N"""
        out1 = torch.mean(x, dim=1, keepdim=True)  # B,1,N
        out2, _ = torch.max(x, dim=1, keepdim=True)  # B,1,N
        out = torch.cat([out2, out1], dim=1)  # B,2,N

        out = self.conv1(out)  # B,1,N
        out = self.bn(out)  # B,1,N
        out = self.relu(out)  # B,1,N

        #att = self.sigmoid(out)  # B, 1, N
        att = self.softmax(out)
        res = att * x
        if return_attn:
            res = res, att
        return res


class CBAMAttentionModule(nn.Module):
    def __init__(self, C, ratio=8):
        super(CBAMAttentionModule, self).__init__()
        self.channel_attn = ChannelAttentionModule(C, ratio)
        self.spatial_attn = SpatialAttentionModule()

    def forward(self, x, return_att=False):
        x = self.channel_attn(x)
        return self.spatial_attn(x, return_att)


if __name__ == '__main__':
    # SE
    input = torch.randn(50, 512, 49)
    input = input.unsqueeze(-1)
    se = SEAttention(channel=512, reduction=8)
    output = se(input)
    output = output.squeeze(-1)
    print(output.shape)
    # Spatial attention
    input = input.squeeze(-1)
    spatial_attn = SpatialAttentionModule()
    output = spatial_attn(input)
    print(output.shape)

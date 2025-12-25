import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeMPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


def get_pool(pool_name):
    if pool_name == 'avg':
        return nn.AdaptiveAvgPool2d((1, 1))
    elif pool_name == 'max':
        return nn.AdaptiveMaxPool2d((1, 1))
    elif pool_name == 'gem':
        return GeMPooling(norm=3)
    else:
        raise AttributeError('not support pooling way')


if __name__ == '__main__':
    feed = torch.randn(3, 10, 8)
    my_pool = get_pool('gem')
    row_out = my_pool(feed[...,None])
    feed = feed.transpose(1,2)[...,None]
    col_out = my_pool(feed)
# 经典卷积神经网络的基本组成部分是下面的这个序列：
# 1.带填充以保持分辨率的卷积层；
# 2.非线性激活函数，如ReLU；
# 3.汇聚层，如最大汇聚层。

import torch
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)




def main():
    net = vgg_block()

    temp = torch.randn(2,3,227,227)

    y = net(temp)

    print(y)



if __name__=='__main__':
    main()



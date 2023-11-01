import torch
from torch import nn

# 卷积块  DenseNet使用了ResNet改良版的“批量规范化、激活和卷积”架构。
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

# 稠密块  一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 将每个卷积块的输入和输出在通道维上连结。
            X = torch.cat((X, Y), dim=1)
        return X
    

# 过渡层  由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。 
# 而过渡层可以用来控制模型复杂度。  
def transition_block(input_channels, num_channels):
    
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))



def main():
    net = DenseBlock(2,6,10)
    X = torch.randn(4, 6, 8, 8)
    Y = net(X)

    blk = transition_block(26, 10)
    

    print(Y.shape)
    print(blk(Y).shape)



if __name__=='__main__':
    main()



























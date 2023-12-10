import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 总体来看，LeNet（LeNet-5）由两个部分组成：
# 卷积编码器：由两个卷积层组成;
# 全连接层密集块：由三个全连接层组成。


in_ch = 3
kernel_ch = 6
class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5,self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),


            nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),


            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # 添加全连接层    
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

      
    

    def forward(self,x):
        x = self.conv_unit(x)

        print('全连接前的结构',x.shape)

        x = x.view(x.size(0),-1)

        print('展开feature',x.shape)

        x = self.classifier(x)  # 使用全连接层
        

        return x





def main():
    net = Lenet5()

    

    tem = torch.randn(2,3,32,32)

    y = net(tem)
    print(net)

    # 计算模型的参数数量
    for p in net:
        print(f"模型的参数数量: {p.numel()}")



if __name__=='__main__':


    main()




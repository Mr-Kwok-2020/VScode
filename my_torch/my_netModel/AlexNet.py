# 推动领域进步的是数据特征，而不是学习算法
# AlexNet和LeNet的设计理念非常相似，但也存在显著差异。
# 1.AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
# 2.AlexNet使用ReLU而不是sigmoid作为其激活函数。

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),  
        )

        self.classifiter = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(9216,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,1000)            
        )


    def forward(self,x:torch.tensor):

        print('全连接前的结构')
        for layer in self.features:
            x = layer(x)
            print(layer.__class__.__name__,'output shape:',x.shape)

            
        print('展开feature')
        for layer in self.classifiter:
            x = layer(x)
            print(layer.__class__.__name__,'output shape:',x.shape)
        

        return x
def main():
    net = AlexNet()

    temp = torch.randn(2,3,227,227)

    y = net(temp)

    # print(y)



if __name__=='__main__':
    main()



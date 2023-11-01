import torch
import torch.nn as nn

#编写一个块，作为后面搭建网络结构中的一个小的嵌套环节
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = torch.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        out2 = self.extra(x)

        print('out1.shape',out1.shape)
        print('out2.shape',out2.shape)

        out = out2 + out1
        print('out.shape',out.shape)

        return out
    
    
# 编写网络，可以调用nn.里面的模块，也可以现写
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        self.blk1 = ResBlk(16, 32)
        self.blk2 = ResBlk(32, 64)
        self.blk3 = ResBlk(128, 256)
        self.blk4 = ResBlk(256, 512)

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # 添加全连接层    
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        # x = self.maxpooling(x)

        x = self.blk1(x)
        x = torch.relu(x)
        x = self.maxpooling(x)

        x = self.blk2(x)
        x = torch.relu(x)
        x = self.maxpooling(x)

        # x = self.blk3(x)
        # x = torch.relu(x)
        # x = self.maxpooling(x)

        # x = self.blk4(x)
        # x = torch.relu(x)
        # x = self.maxpooling(x)

        # print('全连接前的结构',x.shape)

        x = x.view(x.size(0),-1)

        # print('展开feature',x.shape)

        x = self.classifier(x)  # 使用全连接层


        return x




def main():
    blk = ResBlk(64,128)
    tem = torch.randn(2, 64, 32, 32)
    out = blk(tem)
    print(out.shape)

    # model = ResNet18()
    # tem = torch.randn(2, 3, 32, 32)
    # out = model(tem)
    # print(out.shape)




if __name__ == '__main__':
    main()

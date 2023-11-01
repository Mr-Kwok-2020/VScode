import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_self.Lenet5 import Lenet5
import matplotlib.pyplot as plt


device = torch.device('cuda:0')
# device = torch.device('cpu')

batch = 4096
epochs = 2 

def print_epoching(now,all,state): 
    print(state+'ID[{:<4}/{:<4}]'.format(
                now+1, all        
            ), end="\r", flush=True)  # 打印每个epoch的进展

def main():
        
    cifar_train = datasets.CIFAR10(root=r"D:\data\cifar10",train = True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train = DataLoader(cifar_train,batch_size = batch,shuffle=True)
    
    
    cifar_test = datasets.CIFAR10(root=r"D:\data\cifar10",train = False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_test = DataLoader(cifar_test,batch_size = batch,shuffle=True)

    net = Lenet5().to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)


    train_loss_list = [] 
    val_epoch_list = []
    val_loss_list = [] 
    val_acc_list = [] 
    for epoch in range(epochs):
        total_num = 0
        batch_loss = 0
        for id,(x,lable) in enumerate(cifar_train):
            net.train()

            
            x,lable = x.to(device),lable.to(device)
            logits = net(x)
            loss = criterion(logits,lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num  += x.size(0)
            batch_loss += loss  # 累加loss
            print_epoching(id,len(cifar_train),'train')
        
        train_loss_list.append(batch_loss.item()/total_num) 
        
  
        
        with torch.no_grad():
            net.eval()
            batch_acc = 0
            total_num = 0
            batch_loss = 0
            for id,(x,lable) in enumerate(cifar_test):
                
                x,lable = x.to(device),lable.to(device)
                logits = net(x)
                loss = criterion(logits,lable)
                batch_loss += loss


                batch_acc += sum(torch.argmax(logits,dim = 1)==lable)
                total_num += x.size(0)
                print_epoching(id,len(cifar_test),'test ')
        
        val_loss_list.append(batch_loss.item()/total_num) 
        val_acc_list.append(batch_acc.item()/total_num) 
        val_epoch_list.append()
        
        print(epoch,train_loss_list[-1],val_loss_list[-1],val_acc_list[-1])    
        # 保存loss图像
    plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss")
    plt.plot(val_epoch_list, val_loss_list, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.clf()
       
        

































if __name__=='__main__': 
    print('afAWEG')
    main()
   
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_self.Lenet5 import Lenet5
import matplotlib.pyplot as plt


def print_epoching(now,all,state): 
    print(state+'ID[{:<4}/{:<4}]'.format(
                now+1, all        
            ), end="\r", flush=True)  # 打印每个epoch的进展
    
def print_epoching_per(now,all,state): 
    print(state+':{:0.2f}%'.format(
                (now+1)/all*100        
            ), end="\r", flush=True)  # 打印每个epoch的进展
    

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)




import os
import torch
from PIL import Image
from torchvision import transforms
def cal_RGB_mean_std(data_path):

    # 数据集路径
    data_path = 'E:/TEST5train'
    folders = ['cat','dog']

    # RGB通道均值和标准差
    mean, std = torch.zeros(3), torch.zeros(3)

    # 计算 RGB 通道均值和标准差
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        for img_name in os.listdir(folder_path):
            # 打开图像并转换为 Tensor
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms.ToTensor()(img)

            # 累加 RGB 通道均值和标准差
            mean += img_tensor.mean(dim=(1, 2))
            std += img_tensor.std(dim=(1, 2))

    # 计算平均 RGB 通道均值和标准差
    num_samples = len(os.listdir(os.path.join(data_path, folders[0])))
    mean /= (3 * num_samples)
    std /= (3 * num_samples)

    print('RGB 通道均值:', mean)
    print('RGB 通道标准差:', std)
    return mean, std


def jjj():
    # 定义CIFAR-10数据集的变换，转换为PyTorch的Tensor格式
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载CIFAR-10训练集
    cifar_train = datasets.CIFAR10(root=r"/home/action-cv/code/cifar10", train=True, transform=transform)

    # 初始化变量用于计算均值和标准差
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]

    # 遍历数据集并计算均值和标准差
    for img, _ in cifar_train:
        for i in range(3):  # 3表示RGB通道
            mean[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()

    # 计算均值和标准差的平均值
    num_samples = len(cifar_train)
    mean = [m / num_samples for m in mean]
    std = [s / num_samples for s in std]

    print("Mean (RGB):", mean)
    print("Std (RGB):", std)



def music_play():
    from pydub import AudioSegment
    from pydub.playback import play

    # 指定MP3文件路径
    mp3_file = r"/home/action-cv/code/giao.mp3"  # 请替换为您的MP3文件路径

    # 加载MP3文件
    audio = AudioSegment.from_mp3(mp3_file)

    # 播放音频
    play(audio)
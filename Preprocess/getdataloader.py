from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import platform
from Preprocess.augment import Cutout, CIFAR10Policy


os_identifier = platform.system()
if os_identifier == "Windows":
    # 在Windows主机上运行的代码逻辑
    print("Running on Windows")
    DIR = {'CIFAR10': 'C:/datasets/cifar10', 'CIFAR100': 'C:/datasets',
           'ImageNet': 'C:/datasets/ImageNet'}
elif os_identifier == "Linux":
    # 在Linux的远程服务器上运行的代码逻辑
    DIR = {'CIFAR10': '/home/ddx/BATC_DDX/datasets/cifar10', 'CIFAR100': 'F:\datasets', 'ImageNet': '/home/ddx/datasets/imagenet'}
    print("Running on Linux, cifar dir is %s"%DIR)
elif os_identifier == "Darwin":
    # 在MacOS上运行的代码逻辑
    print("Running on MacOS")
    DIR = {'CIFAR10': '/Users/ddx/datasets', 'CIFAR100': '/Users/ddx/datasets/cifar100', 'ImageNet': '/home/ddx/datasets/imagenet'}

# your own data dir

# def GetCifar10(batchsize, attack=False):
#     if attack:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   Cutout(n_holes=1, length=16)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor()])
#     else:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                                   Cutout(n_holes=1, length=8)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
#     test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
#     train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
#     test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
#     return train_dataloader, test_dataloader

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=True)
    return train_dataloader, test_dataloader




def GetCifar100(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    return train_dataloader, test_dataloader

def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=0,
                                 # sampler=train_sampler,
                                 pin_memory=True, drop_last=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=0,
                                 # sampler=test_sampler,
                                 drop_last=True)
    return train_dataloader, test_dataloader


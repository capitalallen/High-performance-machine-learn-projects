"""
Model includes functions for setting hypterparameters, 
training and testing resnet18 model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18
from time import perf_counter
from torchsummary import summary
from util import progress_bar


class Model:
    def __init__(self, device='cuda', data_path='./data', lr=0.1,batch_size=32):
        self.net = ResNet18()
        self.num_workers = 2
        self.device = device
        self.data_path = data_path
        self.lr = lr
        # check if data is downloaded
        if not os.path.exists(self.data_path):
            torchvision.datasets.CIFAR10(
                root=self.data_path, train=True, download=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200)

        # choose gpu or cpu
        self.set_device(self.device)
        self.best_acc = 0
        self.batch_size= batch_size
    """
    print model summary
    """

    def get_model_summary(self):
        summary(self.net, input_size=(3, 32, 32))

    """
    set number of workers  
    """

    def set_num_workers(self, num: int):
        self.num_workers = num

    """
    set criterion
    """

    def set_criterion(self, criterion):
        self.criterion = criterion

    """
    set batch size
    - sgd with nesterov, adagrad, adadelta, adam  
    """

    def set_batchsize(self,size:int):
        self.batch_size=size

 
    """
    set to gpu or cpu (cuda or cpu)
    """

    def set_device(self, device):
        self.device = device
        self.net.to(device)
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

    """
    load and convert to dataholder
    """

    def load_dataset(self):
        start = perf_counter()
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        end = perf_counter()
        print("total number of workers: {}; \n total time used for loading data {}".format(
            self.num_workers, end-start))
        return end-start

    """
    train the model 
    """

    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        data_loading_time = 0
        training_time = 0
        last=len(self.trainloader)
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # start of timer
            start = perf_counter()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            data_loading_end = perf_counter()
            data_loading_time += data_loading_end-start
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            training_time_end = perf_counter()
            training_time += training_time_end-data_loading_end
            # progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),last)
        print('training loss {}'.format(round(train_loss/len(self.trainloader),3)))
        print('accuracy: {}%'.format(round(100.*correct/total),3))
        return (data_loading_time, training_time)

    def execute(self, epochs,batch_size):
        print("-------batch size: {}-------".format(batch_size))
        self.batch_size=batch_size
        self.load_dataset()
        data_loading_time = 0
        trianing_time = 0
        for epoch in range(0, epochs):
            temp1, temp2 = self.train(epoch)
            data_loading_time += temp1
            trianing_time += temp2

            print("epoch number: {}, training time used: {}s".format(epoch, temp2))
            print("epoch number: {}, data-loading time used: {}s".format(epoch, temp1))
            print("epoch number: {}, data-loading + training time used: {}s".format(epoch, round(temp1+temp2,3)))
            self.scheduler.step()



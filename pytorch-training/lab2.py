'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import torch 
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 80 # int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None,last=0):
    if current >= last-1:
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time

        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockCustom, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckCustom(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckCustom, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCustom(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCustom, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18Custom():
    return ResNetCustom(BasicBlockCustom, [2, 2, 2, 2])




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
from time import perf_counter
from torchsummary import summary

class Model:
    def __init__(self, device, data_path='./data', lr=0.1,custom_resnet=False):
        if custom_resnet:
            self.net = ResNet18Custom()
        else:
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
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
        #                     momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200)

        # choose gpu or cpu
        self.set_device(self.device)
        self.best_acc = 0
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
    set optimizer
    - sgd with nesterov, adagrad, adadelta, adam  
    """

    def set_optimizer(self, optimizer: str):
        print("optimizer used: {}".format(optimizer))
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                       momentum=0.9, weight_decay=5e-4)
        elif optimizer == 'nesterov':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, nesterov=True,
                                       momentum=0.9, weight_decay=5e-4)
        elif optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=self.lr, weight_decay=5e-4)
        elif optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=self.lr, weight_decay=5e-4)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                                            weight_decay=5e-4)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                       momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200)

    """
    set scheduler 
    """

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

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
            trainset, batch_size=128, shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=self.num_workers)
        end = perf_counter()
        print("total number of workers: {}; \n total time used for loading data {}".format(
            self.num_workers, end-start))
        return end-start

    """
    train the model 
    """

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
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
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total),last)
            if correct/total > self.best_acc:
                self.best_acc = correct/total
        return (data_loading_time, training_time)
    """
    test the model 
    """

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def execute(self, epochs, display_training_time=False, display_total_time=False, display_average_time=False, display_best_acc=False, display_data_loading_time=False):
        self.load_dataset()
        data_loading_time = 0
        trianing_time = 0
        for epoch in range(0, epochs):
            temp1, temp2 = self.train(epoch)
            data_loading_time += temp1
            trianing_time += temp2

            if display_training_time:
                print("epoch number: {}, training time used: {}s".format(epoch, temp2))
            if display_data_loading_time:
                print(
                    "epoch number: {}, data-loading time used: {}s".format(epoch, temp1))
            self.scheduler.step()

        if display_total_time:
            print("total running time used for {} epochs: {}s".format(
                epochs, data_loading_time+trianing_time))
        if display_average_time:
            print("average time used for {} epochs: {}s".format(
                epochs, (data_loading_time+trianing_time)/epochs))
        if display_best_acc:
            print("best training accuracy: {}".format(self.best_acc))


"""
define command line arguments 
- epochs: int, optinal, default=5
- data_path: str, optional
- use_cuda: str, 
- num_workers: int 
- optimizer: str 
- model_summary: str
- question: c2 - c7
    - c2: 
        -> run c2 exercise 
            1. data-loading time 
            2. training time for each epoch 
            3. total running time
    - c3: 
        -> run only dataloader 
        -> print time used 
    - c4: 
        -> use 1 and number of workers with best performance for data loading and computing
    - c5:
        -> run with gpu and cpu and record the time 
    - c6: 
        -> run with different optimizers 
    - c7: 
        -> without using batch norm layers 
"""
import argparse
from matplotlib import pyplot as plt 


"""
question - c1 
- use arguments to init model class 
- display top-1 training accuracy 
- call model.execute
"""
def c1(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr) 
    model_func.num_workers=num_workers
    model_func.execute(epochs,False,False,False,True,False) 

"""
question - c2 
- display data-loading time for each epoch 
- display training time for each epoch 
- display total running time 
"""
def c2(lr,epochs,num_workers,optimizer,data_path,device): 
    model_func=Model(device,data_path,lr) 
    model_func.num_workers=num_workers
    model_func.execute(epochs,True,True,False,False,True) 
 

def c3(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr)
    x,y=[],[]
    for i in range(2,64,2):
        model_func.set_num_workers(i)
        time = model_func.load_dataset()
        x.append(i)
        y.append(time)
    plt.scatter(x,y)
    plt.xlabel("nubmer of workers")
    plt.ylabel("time used in seconds")
    plt.show() 
    plt.savefig('data_loading_time_used.png')

"""
use one worker vs use 18 workers 
- use same measurements as c2 
"""
def c4(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr) 
    model_func.num_workers=num_workers
    model_func.execute(epochs,True,True,False,False,True) 
 

def c5(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr) 
    model_func.num_workers=num_workers
    model_func.execute(epochs,True,True,False,False,True) 

def c6(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr) 
    model_func.set_optimizer(optimizer)
    model_func.num_workers=num_workers
    model_func.execute(epochs,True,True,True,True,True) 
 
def c7(lr,epochs,num_workers,optimizer,data_path,device):
    model_func=Model(device,data_path,lr,True) 
    model_func.set_optimizer(optimizer)
    model_func.num_workers=num_workers
    model_func.execute(epochs,True,True,True,True,True) 
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lab 2')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--num_workers',default=2,type=int)
    parser.add_argument('--optimizer', default='sgd',type=str, help='what optimier to use')
    parser.add_argument('--model_summary', default=False, type=bool)
    parser.add_argument('--question',default='c0',type=str)
    parser.add_argument('--data_path',default='./data',type=str)
    parser.add_argument('--device',default='cpu',type=str)
    args = parser.parse_args()

    if args.device not in ['cpu','cuda']:
        raise Exception("device should be cpu or cuba")
    if args.model_summary:
        if args.question=='c7':
            model_func=Model(args.device,args.data_path,args.lr,True)
        else:
            model_func=Model(args.device,args.data_path,args.lr)
        model_func.get_model_summary()
    else: 
        if args.question=='c1':
            c1(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device) 
        elif args.question=='c2':
            c2(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device) 
        elif args.question=='c3':
            c3(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device) 
        elif args.question=='c4':
            c4(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device) 
        elif args.question=='c5':
            c5(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device)
        elif args.question=='c6':
            c6(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device)
        elif args.question=='c7':
            c7(args.lr,args.epochs,args.num_workers,args.optimizer,args.data_path,args.device)

        


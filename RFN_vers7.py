# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:27:21 2020

@author: pc
"""


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import torch.distributions as td

#from tqdm.notebook import trange, tqdm


class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=32,digit_size=28, deterministic=True, three_channels = True, step_length=4, normalize = True, make_target = False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = step_length
        self.digit_size = digit_size
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 
        self.three_channels = three_channels
        self.normalize = normalize
        self.make_target = make_target
        self.data = datasets.MNIST(
                path,
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.digit_size, interpolation=1),
                     transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
       
        x = np.zeros((self.seq_len,
                          image_size, 
                          image_size, 
                          self.channels),
                        dtype=np.float32)
        
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            digit=digit.numpy()
            ds=digit.shape[1]
            sx = np.random.randint(image_size-ds)
            sy = np.random.randint(image_size-ds)
            dx = np.random.randint(-self.step_length, self.step_length+1)
            dy = np.random.randint(-self.step_length, self.step_length+1)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, self.step_length+1)
                        dx = np.random.randint(-self.step_length, self.step_length+1)
                elif sy >= image_size-ds:
                    sy = image_size-ds-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-self.step_length, 0)
                        dx = np.random.randint(-self.step_length, self.step_length+1)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, self.step_length+1)
                        dy = np.random.randint(-self.step_length, self.step_length+1)
                elif sx >= image_size-ds:
                    sx = image_size-ds-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-self.step_length, 0)
                        dy = np.random.randint(-self.step_length, self.step_length+1)
                   
                x[t, sy:sy+ds, sx:sx+ds, 0] += digit.squeeze()
                sy += dy
                sx += dx
                

        if self.normalize:
          x = (x - 0.1307) / 0.3081
        
        n_channels = 1
        x=x.reshape(self.seq_len, n_channels, self.image_size, self.image_size)
        x[x>1] = 1. # When the digits are overlapping.
        
        if self.three_channels:
            x=np.repeat(x, 3, axis=1)
        
        if self.make_target == True:
          # splits data into two, a one for training and another one for target, output will be a LIST with 2 elements 
          x = np.split(x, 2, axis=0)
        return x

batch_size=64
n_frames=10
testset = MovingMNIST(False, 'Mnist', seq_len=n_frames, image_size=32, digit_size=20, num_digits=1, 
                                              deterministic=False, three_channels=False, step_length=2, normalize=True)
trainset = MovingMNIST(True, 'Mnist', seq_len=n_frames, image_size=32, digit_size=20, num_digits=1, 
                                             deterministic=False, three_channels=False, step_length=2, normalize=True)

train_loader=DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last = True)
test_loader=DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last = True)


### Flow Part

class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]

        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs), requires_grad=True))


    def forward(self, input, logdet=0, reverse=False):
        dims = input.size(2) * input.size(3)
        if reverse == False:
            input = input + self.bias
            input = input * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * dims
            logdet = logdet + dlogdet

        if reverse == True:
            input = input * torch.exp(-self.logs)
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dims
            logdet = logdet + dlogdet

        return input, logdet


class Conv2dResize(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        stride = [in_size[1]//out_size[1], in_size[2]//out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        
        self.conv = nn.Conv2d(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.conv.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return[k0, k1]

    def forward(self, input):
      output = self.conv(input)
      return output 


class Conv2dZeros(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        super().__init__()
        
        padding = (kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv.weight.data.normal_(mean=0.0, std=0.1)
    
    def forward(self, input):
      output = self.conv(input)
      return output 


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, input):
      output = self.linear(input)
      return output 


class LinearNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.normal_(mean=0.0, std=0.1)
        self.linear.bias.data.normal_(mean=0.0, std=0.1)
  
    def forward(self, input):
      output = self.linear(input)
      return output

class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size,
                      stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)

class Conv2dNormy(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv.weight.data.normal_(mean=0.0, std=0.05)
        self.actnorm = ActNorm(out_channels)

    def forward(self, input):
        output = self.conv(input)
        output, _ = self.actnorm(output)
        return output

class Conv2dZerosy(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1]):
        super().__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.logscale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("newbias", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, input):
        output = self.conv(input)
        output = output + self.newbias
        output = output * torch.exp(self.logs * self.logscale_factor)
        return output

class ActFun(nn.Module):
  def __init__(self, non_lin):
    super(ActFun, self).__init__()
    if non_lin=='relu':
      self.net=nn.ReLU()
    if non_lin=='leakyrelu':
      self.net=nn.LeakyReLU(negative_slope=0.20)

  def forward(self,x):
    return self.net(x)

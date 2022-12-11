import json

import argparse
import time
try: import setGPU
except ImportError: pass

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Function, Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math
import numpy as np
import shutil

import setproctitle

import densenet
# import models
from newlayer import diff
from utils import GraphData


class MIPSolver(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, neq, nineq, Q_,  G_, h_, A_, b_, Qpenalty=0.1, eps=1e-4):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nCls = nCls

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.Q = Q_.to(device)
        # self.G = -torch.eye(nHidden).double().to(device)
        # self.h = torch.zeros(nHidden).double().to(device)
        self.G = G_.to(device)
        self.h = h_.to(device)
        self.A = A_.to(device)
        self.b = b_.to(device)

        self.neq = neq
        self.nineq = nineq

    def forward(self, x):
        # nBatch = x.size(0)
        nBatch = 1
        # FC-ReLU-QP-FC-Softmax
        # x = x.view(nBatch, -1)

        # torch.nn.functional 
        # x = F.relu(self.fc1(x))
        
        ## 

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1)).double()
        #p = -x.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1)).double()
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0)).double()
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1)).double()
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0)).double()

        x = diff(verbose=False)(Q, x.double(), G, h, A, b).float()

        x = self.fc2(x)

        return F.log_softmax(x)



def get_net(args, traindata):
    if args.model == 'MIPSolver':
        Q = torch.eye(len(traindata.var_nodes))
        net = MIPSolver(28 * 28, args.nHidden, 10, args.neq, args.nineq, Q ,1,1,1,1)
    else:
        assert(False)

    return net

def get_loaders(args):
    data = torch.load('data_1.pt')
    return data


def train(args, epoch, net,  traindata, optimizer, trainF):
    seed = 1
    torch.manual_seed(seed)
    net.train()
    nProcessed = 0
    
    begin = time.time()
    
    data = traindata.feature # previously trained initial point of x
    target = traindata.obj # target
    optimizer.zero_grad()
    output = net(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    pred = output.data.max(1)[1] # get the index of the max log-probability
    incorrect = pred.ne(target.data).cpu().sum()
    err = 100.*incorrect/len(data)
    
    print('Train Result: \tLoss: {:.6f}\tError: {:.6f}'.format(
            loss.item(), err))
    end = time.time()
    trainF.write('{},{},{}\n'.format(loss.item(), err, end-begin))
    trainF.flush()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nEpoch', type=int, default=100)
    parser.add_argument('--weightDecay', type=float, default=1e-4)
    # parser.add_argument('--opt', type=str, default='sgd',
    #                     choices=('sgd', 'adam'))
    subparsers = parser.add_subparsers(dest='model')
    oursP = subparsers.add_parser('ours')
    oursP.add_argument('--nHidden', type=int, default=200)
    # number of equality constraints
    oursP.add_argument('--neq', type=int, default=50)
    # number of inequality constraints
    oursP.add_argument('--nineq', type=int, default=50)
    mipP = subparsers.add_parser('MIPSolver')
    mipP.add_argument('--nHidden', type=int, default=200)
    mipP.add_argument('--neq', type=int, default=50)
    mipP.add_argument('--nineq', type=int, default=50)
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() # check available cuda
    if args.save is None:
        t = 'models-{}'.format(args.model)
    setproctitle.setproctitle('bamos.'+t)
    args.save = os.path.join(args.work, t)
    
    # set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # save output
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    
    
    # trainLoader and test Loader might be modified later.
    traindata = get_loaders(args)
    
    net = get_net(args , traindata)
    
    optimizer = optim.Adam(net.parameters()) # Adam or SGD
    args.nparams = sum([p.data.nelement() for p in net.parameters()])
    with open(os.path.join(args.save, 'meta.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=2)
        
    print('  + Number of params: {}'.format(args.nparams))
    if args.cuda:
        net = net.cuda()
        
    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    for epoch in range(1, args.nEpoch + 1):
        # adjust_opt(args, optimizer, epoch)
        train(args, epoch, net, traindata, optimizer, trainF)
        # test(args, epoch, net, testLoader, optimizer, testF)
        try:
            torch.save(net, os.path.join(args.save, 'latest.pth'))
        except:
            pass
        os.system('./plot.py "{}" &'.format(args.save))

    trainF.close()
    testF.close()
    
if __name__=='__main__':
    main()
    # print(data)
    data = torch.load('data_1.pt')
    print(data)

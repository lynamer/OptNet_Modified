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
import copy
import math
import numpy as np
import shutil
import random
import setproctitle

import densenet
# import models
from newlayer import diff
import utils
import torch_geometric
from qpth.qp import QPFunction

class MIPSolver(nn.Module):
    def __init__(self, data, mu = 1e-3, L=2, lam = 50):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = data.var_nodes.size(0)
        print(f"mu: {mu}")
        # print('n: ', n)
        self.no_eq_cons_flag = False
        
        eq_cons_idx = []
        eq_cons_val = []
        eq_cons_list = []
        eq_cons_b = []
        neq_cons_idx = []
        neq_cons_val = []
        neq_cons_list = []
        
        neq_cons_b = [] 
        n_conditions = 0
        n_eq_conditions = 0
        print(f"data.edge_attr.size(): {data.edge_attr.size()}")
        
        # number of all constraints
        num_cons = data.cons_nodes.size()[0]
        cons_index_map = [0 for i in range(num_cons)]
        print(f"num_cons: {num_cons}")
        # Task 1: mapping of constraint in cons_node to equality constraint A and inequality constriant G
        #       0 for equation constraint
        #       1 for inequation constraint
        # Task 2: directly construct b and g from cons_nodes
        for con_idx in range(num_cons):

            if data.cons_nodes[con_idx][0] == 1:
                cons_index_map[con_idx] = len(eq_cons_list)
                eq_cons_list.append(con_idx)
                eq_cons_b.append(data.cons_nodes[con_idx][1])
            else:
                cons_index_map[con_idx] = len(neq_cons_list)
                neq_cons_list.append(con_idx)
                neq_cons_b.append(data.cons_nodes[con_idx][1])
        print(f"equ_cons_list: {eq_cons_list}")
        print(f"neq_cons_list: {neq_cons_list}")
        
        # Construct A from cons_nodes
        for i in range(data.edge_attr.size()[0]):
            var_idx = data.edge_index[0][i]
            con_idx = data.edge_index[1][i]
            is_eq = data.cons_nodes[con_idx][0]     # constriant_type, 1:eq_constraint 0:neq       
            if is_eq:
                eq_cons_idx.append([cons_index_map[con_idx], var_idx]) # constaint index
                eq_cons_val.append(data.edge_attr[i]) # corresponding A
            else:
                neq_cons_idx.append([cons_index_map[con_idx], var_idx]) # constaint index
                neq_cons_val.append(data.edge_attr[i]) # corresponding A
        # there is no equality constaints in the problem. For the sake of 0.
        if not eq_cons_list:
            self.no_eq_cons_flag = True
            eq_cons_idx = torch.tensor([[0], [n]]).to(torch.long)
            eq_cons_val = torch.tensor([1]).to(torch.float32)
            eq_cons_b = torch.tensor([0]).to(torch.float32)
            n_eq_conditions = 1
            n = n + 1
        else:
            eq_cons_idx = torch.tensor(np.array(eq_cons_idx).T).to(torch.long)
            eq_cons_val = torch.tensor(eq_cons_val).to(torch.float32)
            eq_cons_b = torch.tensor(eq_cons_b).to(torch.float32)
            n_eq_conditions = len(eq_cons_list)
        A = torch.sparse_coo_tensor(indices=eq_cons_idx, values=eq_cons_val, size=[n_eq_conditions, n]) # Ax = b
        b = eq_cons_b 

        n_conditions = len(neq_cons_list)
        # Here we suppose all of the variables have upper and lower bounds. so we add 1000 * 2 constraints to the matrix to make G become a 27000 * 1000 matrix
        for i in range(n - self.no_eq_cons_flag): 
            lb = data.var_nodes[i][2]
            ub = data.var_nodes[i][3]
            neq_cons_idx.append([n_conditions, i])
            neq_cons_val.append(1)
            neq_cons_b.append(ub)
            n_conditions = n_conditions + 1
            neq_cons_idx.append([n_conditions, i])    # -x <= -lb
            neq_cons_val.append(-1)
            neq_cons_b.append(-lb)
            n_conditions = n_conditions + 1           
        neq_cons_idx = torch.tensor(np.array(neq_cons_idx).T).to(torch.long)
        neq_cons_val = torch.tensor(neq_cons_val).to(torch.float32)
        neq_cons_b = torch.tensor(neq_cons_b).to(torch.float32)

        G = torch.sparse_coo_tensor(indices=neq_cons_idx, values=neq_cons_val, size=([n_conditions, n]))
        h = neq_cons_b

        c = []
        bin = []
        for i in range(n - self.no_eq_cons_flag):
            bin.append(data.var_nodes[i][0]) # whether binary
            c.append(data.var_nodes[i][1])
        # print([f'{c[i]} * x_{i}' for i in range(n)], sep="+", end='')
        if self.no_eq_cons_flag:
            bin.append(0)
            c.append(0)
        Q = torch.eye(n)
        # Q = torch.zeros([n, n])
        Q = Q * mu
        self.Q = Q.to(device)
        self.G = G.to(device)
        self.h = h.to(device)
        self.A = A.to(device)
        self.b = b.to(device)
        self.L = L                          # hyperparameter L=2
        self.c = c                          # c^T x primal function
        self.bin = bin                      # whether binary list
        self.device = device                # device = cuda
        self.n = n                          # n_variables
        self.lam = lam


    def forward(self, y_0):
        nBatch = 1
        
        # Here batch is not for any use, just to meet with newlayer's code.
        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1)).double()
        G = self.G.to_dense()
        G = G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1)).double()
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0)).double()
        A = self.A.to_dense()
        # print(f"A.shape: {A.shape}")
        A = A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1)).double()
        
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0)).double()
        print(f"lambda: {self.lam}")
        print(f"L: {self.L}")
        p = []
        for i in range(len(y_0)):
            temp = self.c[i] 
            if self.bin[i]:                                         # is binary variable
                temp = temp + self.lam * (1 - self.L * math.pow(y_0[i], self.L-1)) # add DC constraints, p = c + 1 - L (z_i)^L     else p = c
            p.append(temp)
        if self.no_eq_cons_flag:
            p.append(0)
        p = torch.tensor(p).to(torch.float32)
        p = p.to(self.device)
        p = p.unsqueeze(0).expand(nBatch, p.size(0)).double()
        print(f"Q.size: {Q.size()}")
        print(f"p.size: {p.size()}")
        print(f"G.size: {G.size()}")
        print(f"h.size: {h.size()}")
        print(f"A.size: {A.size()}")
        print(f"b.size: {b.size()}")
        x = diff(verbose=False)(Q, p, G, h, A, b).float()
        
        # for i in range(len(x)):
            
        #     if self.bin[i]: 
        #         result[i] = round(result[i])# is binary variable
        # print( G @ x.T - h)
        # print( A @ x.T - b)
        # x = QPFunction(verbose=True)(Q, p, G, h, A, b).float()
        if self.no_eq_cons_flag:
            return x[:, :-1]
        else:
            return x
    


def solve_MIP(data):
    mu = 1e-3
    L = 2
    lam = 5
    opt_prob = MIPSolver(data, mu=mu, lam=lam)
    return opt_prob(data.y)
    
# def test_diff():
#     n = 10
#     Q = torch.tensor([[]])
#     Q = Q.unsqueeze(0).double()
#     p = torch.zeros(2).unsqueeze(0).double()
#     A = torch.zeros(2).unsqueeze(0).double()
#     b = torch.zeros(n).unsqueeze(0).double()
#     G = torch.eye(n).unsqueeze(0).double()
#     h = torch.zeros(n).unsqueeze(0).double()
#     x = diff(verbose=False)(Q, p, G, h, A, b).float()
#     print(f"test_diff: x={x}")
    
if __name__=='__main__':
    # data = torch.load(os.path.join("instances", "10r5c_eq_1.pt"))
    # data = torch.load(os.path.join("instances", "10r5c_eq_2.pt"))
    # data = torch.load(os.path.join("instances", "10r5c_eq_3.pt"))
    # data = torch.load(os.path.join("instances", "10r5c_eq_4.pt"))
    # instance_path = os.path.join("instances","tiny_10r_5c_0.55d", "instance_2.pt")
    instance_path = os.path.join("instances","tiny_1000r_500c_0.05d", "instance_3.pt")
    # instance_path = os.path.join("instances","10r5c_neq.pt")
    # instance_path = os.path.join("instances","instance_1.pt")
    # instance_path = os.path.join("instances", "10r5c_eq_4.pt")
    # instance_path = os.path.join("instances", "3r_3c.pt")
    data = torch.load(instance_path)
    print(f"runing instance :{instance_path}")
    x = solve_MIP(data)
    # print(x)
    # print(x.size())
    # print(data.var_nodes)
    # print(data.var_nodes.size())
    sum = 0
    print(f"x: {x[0]}")
    for i in range(len(data.var_nodes)):
        res = data.var_nodes[i][1] * x[0][i]
        sum = sum + res
    print(f"final objective: {sum}")
    print(f"expected objective: {data.obj}")
    # test_diff()
import random
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader as tloader
import torch.nn.functional as F
import torch_geometric
import pickle
import pprint
import os


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

PKL_DIR = "./dataset/pkl/"
PROCESSED_DIR = "./dataset/pkl/processed"
H_VAR = 5-1
H_CONS = 6-1

def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    print("========== pad_tensor ============")
    
    max_pad_size = pad_sizes.max()
    print("max_pad_size: ",max_pad_size)
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    print("output: ", output.shape)
    output = torch.stack([
        F.pad(slice_,
              (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
        for slice_ in output
    ],
                         dim=0)
    return output

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        print(file_path)
        var_nodes, cons_nodes, Adjacent_matrix, var_labels, obj_label = pickle.load(f)
        var_nodes = dict(var_nodes)
        cons_nodes = dict(cons_nodes)
        var_labels = dict(var_labels)
        # pprint.pprint(var_nodes)
        # data = [var_nodes, cons_nodes, Adjacent_matrix, var_labels, obj_label]
        # pprint.pprint(cons_nodes)
        # print(var_labels.keys())


        # assert len(var_nodes) == len(var_labels), "file: {}, num_nodes {}, num_labels{}".format(file_path, len(var_nodes), len(var_labels))
        N_var = len(var_nodes)
        N_cons = len(cons_nodes)

        var_nodes_tensor = np.zeros((N_var, H_VAR))
        cons_nodes_tensor = np.zeros((N_cons, H_CONS))
        var_labels_tensor = torch.Tensor(N_var)
        label_binary_mask = []
        label_continuous_mask = []

        for v_name in var_nodes.keys():
            var_vector = var_nodes[v_name]
            var_nodes_tensor[int(var_vector[0])] = var_vector[1:]

        for v_name in var_labels.keys():
            if v_name == b'#':
                continue
            # print(type(v_name), v_name)
            label = float(var_labels[v_name])
            v_name = str(v_name, encoding = "utf-8") 
            assert v_name in var_nodes.keys()
            var_vector = var_nodes[v_name]
            # print(var_vector[1:], len(var_vector[1:]), type(var_vector)), print(label), print(var_nodes_tensor.shape, var_nodes_tensor[int(var_vector[0])])
            # var_nodes_tensor[int(var_vector[0])] = var_vector[1:]
            if int(var_vector[1]) == 1:
                label_binary_mask.append(int(var_vector[0]))
            else:
                assert int(var_vector[1]) == 0
                label_continuous_mask.append(int(var_vector[0]))
            var_labels_tensor[int(var_vector[0])] = label
        # var_nodes_tensor = torch.tensor(var_nodes_tensor[:,:-1])

        for c_name in cons_nodes.keys():
            # print(c_name)
            cons_vector = cons_nodes[c_name]
            # print(cons_vector)
            assert len(cons_vector[1:]) == 5
            cons_nodes_tensor[int(cons_vector[0])] = cons_vector[1:]
        cons_nodes_tensor = torch.tensor(cons_nodes_tensor)
        var_nodes_tensor = torch.tensor(var_nodes_tensor)

        edge_index = torch.tensor(Adjacent_matrix[1:,:-1].T) # 2*e, var_idx - cons_idx
        edge_attr = torch.tensor(Adjacent_matrix[1:,-1].reshape(-1,1))

        obj_label = torch.tensor(float(obj_label))

        y_mask = [label_binary_mask, label_continuous_mask]

        return var_nodes_tensor, cons_nodes_tensor, edge_index, edge_attr, var_labels_tensor, obj_label, y_mask

class GraphData(torch_geometric.data.Data):
    """
    var_nodes are n*h1 tensor vector representing variable nodes
    cons_nodes are n*h2 tensor vector representing constraint nodes
    edge_index are 2*e tensor vector, each column represents an undirect edge
    edge_attr is e*m tensor vector, each row represents the edge feature 
    y is the n*1 target label for variable assignments
    obj is the scalar target label for objective value
    """

    # def __init__(self, file_name, var_nodes, cons_nodes, edge_index, edge_attr, y, obj, y_mask):
    #     super().__init__()
    #     self.file_name = file_name
    #     # tensor
    #     self.var_nodes = var_nodes
    #     self.cons_nodes = cons_nodes
    #     self.edge_attr = edge_attr
    #     self.edge_index = edge_index
    #     self.y = y 
    #     self.obj = obj
    #     if y_mask:
    #         self.y_binary_mask = y_mask[0]
    #         self.y_continuous_mask = y_mask[1] 

    #     # self.y[self.y_binary_mask]        label for binary variables
    #     # self.y[self.y_continuous_mask]    label for continuous variables

    # def info(self):
    #     print("num_var {}, num_cons {}, edge size {}, y size {}, num_binary {}, obj {}".format(
    #         self.var_nodes.size(0), self.cons_nodes.size(0), list(self.edge_index), list(self.y.size(), len(self.y_binary_mask), self.obj)
    #     ))

    def __init__(self, var_nodes, cons_nodes, edge_index, edge_attr, y, obj):
        super().__init__()
        # tensor
        self.var_nodes = var_nodes
        self.cons_nodes = cons_nodes
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.y = y
        self.obj = obj




class GraphDataset(torch_geometric.data.Dataset):

    def __init__(self, root):
        super().__init__(root, transform = None, pre_transform=None)
        self.root = root
        # self.processed_dir = processed_dir
            

    @property
    def raw_file_names(self):
        
        return [self.root+"{}".format(pkl_file) for pkl_file in findAllFile(self.root)]

    @property
    def processed_file_names(self):
        l = []
        idx = 0
        for pt_file in findAllFile(self.processed_dir):
            if pt_file[:4] == "data":
                l.append(osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        return l
        # return ['data_1.pt', 'data_2.pt', ...]

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        idx = 0
        for pkl_file in findAllFile(self.root):
            # print(self.root)
            # if pkl_file == "ran13x13.pkl":
            if pkl_file.split(".")[-1] != "pkl":
                continue

            if idx % 50 == 0:
                print("======================== {}".format(idx))

            with open("pkl2pt_done.txt", "a+") as ff:
                ff.seek(0,0)
                lines = ff.read().split()
                if pkl_file+"_IDX_{}".format(idx) in lines:
                    print("Existing data_{}.pt file of {}".format(idx, pkl_file))
                    idx += 1
                    continue
            
            with open("error_pkl.txt", "a+") as f:
                f.seek(0,0)
                lines = f.read().split()
                if pkl_file in lines:
                    
                    continue

            try:
                with open("pkl2pt_done.txt", "a+") as ff:
                    ff.seek(0,0)
                    lines = ff.read().split()
                    # print(lines)
                    
                    var_nodes, cons_nodes, edge_index, edge_attr, y, obj, y_mask = load_pkl(osp.join(self.root,pkl_file))
                    graph = GraphData(file_name = pkl_file.split(".")[0], var_nodes = var_nodes, cons_nodes = cons_nodes, edge_index = edge_index, edge_attr = edge_attr, y = y, obj = obj, y_mask = y_mask)
                    # self.data_list.append(graph)
                    torch.save(graph, osp.join(self.processed_dir, f'data_{idx}.pt'))
                    ff.write("{}_IDX_{}\n".format(pkl_file,idx))

                    idx += 1
            except:
                with open("error_pkl.txt", "a+") as f:
                    f.seek(0,0)
                    lines = f.read().split()
                    if pkl_file in lines:
                        print("**********ERROR {}".format(pkl_file))
                    else:
                        f.write("{}\n".format(pkl_file))
                        print("**********ERROR {}".format(pkl_file))


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# var_nodes_tensor, cons_nodes_tensor, edge_index, edge_attr, var_labels, obj_label = load_pkl(PKL_DIR+"10teams.pkl")
# data = GraphData(var_nodes_tensor, cons_nodes_tensor, edge_index, edge_attr, var_labels, obj_label)

class graphdataset(torch_geometric.data.Dataset):

    def __init__(self, files):
        super().__init__(root=None, transform = None, pre_transform=None)
        self.files = files
        # self.processed_dir = processed_dir
    
    def len(self):
        return len(self.files)

    
    def get(self, idx):
        data = torch.load(self.files[idx])
        return data

class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch = +1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


if __name__ == '__main__':
    
    

    print("============================================")
    dataset = GraphDataset(root = PKL_DIR)
    dataset2 = graphdataset(["./dataset/pkl/processed/data_0.pt"])
    print(len(dataset2))
    print(dataset2.len())
    # print(dataset.processed_file_names)
    print("*============================================")
    # dataloader = torch_geometric.loader.DataLoader(dataset, batch_size = 128, shuffle = False, num_workers = 0)
    # dataloader = tloader(dataset, batch_size = 2, shuffle = False)
    print("****============================================")

    # print(len(dataloader))
    # print(type(dataloader))
    
    print(dataset[0])
    # print(dataset[0].y_binary_mask)
    print(dataset[0].var_nodes[:8], type(dataset[0].var_nodes))
    
    for i in range(len(dataset)):
        dataloader = torch_geometric.loader.DataLoader([dataset[i]], 1, shuffle = True)
        print("len dataloader: ",len(dataloader))
        for batch in dataloader:
            print(batch)
            print(batch.file_name)
            print(batch.var_nodes[0].shape)
            print(len(batch.y_binary_mask[0]))
            print(batch.batch)
            break
        break
        

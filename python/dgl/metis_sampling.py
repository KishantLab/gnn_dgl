import numpy as np
import torch
import torch as th
import dgl
import os
_computed_array = None
_part_array = None
_spmm_method = 0  # 0: no reorder, 1: reorderd, 2: no reorder with samplin
_sampling_method = 1

def metis_partition(G, parts=None, method=None, spmm_reorderd=0, sampling=0, dataset=None):
    global _computed_array
    global _part_array
    global _spmm_method
    global _sampling_method
    if _computed_array is None and sampling == 0:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        print(G)
        print(type(G))
        print("partition start")
        Nodes = G.num_nodes() 
        num_parts = int(Nodes/1024)
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)

        # print(g)
        _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
        if method is None:
                if os.path.exists(dataset + "_metis_part.npy"):
                    _computed_array = np.load(dataset + "_metis_part.npy")
                    print(f"Loaded array from {dataset}_metis_part.npy")
                else:   
                    _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
                    np.save(dataset + "_metis_part.npy", _computed_array)
                    print(f"Created new partion and saved to {dataset}_metis_part.npy")
        elif method == "rm":
            _computed_array = np.random.randint(0, parts, size=Nodes)
        elif method == "contig":
            l = Nodes // parts   # calculate the number of repeated values for each number
            _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
            for i in range(parts):
                _computed_array[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
        elif method == "metis":
            _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        # _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        print(_computed_array.shape)
        
        #-------------spmm methos is reorderd then it executed-----------------------------------
        if spmm_reorderd == 1:
            _spmm_method = 1
            # num_parts = int(Nodes/1024)
            # _part_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
            _part_array = _computed_array
            _part_array = list(map(int, _part_array))
            _part_array = [[d, i] for i, d in enumerate(_part_array, 0)]
            _part_array.sort()
            _part_array = [item[1] for item in _part_array]
            _part_array = th.tensor(_part_array)
            _part_array = _part_array.to(device)
            print(_part_array)
        elif spmm_reorderd == 2:
            _spmm_method = 2
        # context = dgl.cuda.get_context(0)
        # context = dgl.cuda.context(0)
        # _computed_array = np.random.rand(10)
        # _computed_array = np.random.randint(10000, 90001, size=10)
        # _computed_array = _computed_array.astype(np.int64)
        # _computed_array = torch.from_numpy(_computed_array)
        # _computed_array = _computed_array.to(device)
        _computed_array = dgl.ndarray.array(_computed_array)
        # _computed_array = _computed_array.to(device)
        # Convert NumPy array to DGL tensor
        # _computed_array = dgl.tensor(_computed_array)
        # device = "cuda" if dgl.cuda.is_available() else "cpu"
        # _computed_array = _computed_array.to(device)
        # _computed_array = _computed_array.tolist
        _sampling_method = sampling
        print("Array computation done and passed to neighbour.py line 631")
    else:
        Nodes = G.num_nodes() 
        _computed_array = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
        _computed_array = dgl.ndarray.array(_computed_array)
        if spmm_reorderd == 2:
            _spmm_method = 2
        # elif spmm_reorderd == 0:
            # _spmm_method = 0
    # print("spmm_methods: ",_spmm_method)
    return _computed_array, _sampling_method

def return_array():
    global _part_array
    global _spmm_method
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
    # if _part_array is None:
        # print("Array not initilized")
    # print("spmm: ",_spmm_method)
    return _part_array, _spmm_method 

def get_part_array(G, parts=None, method=None, spmm_reorderd=0, sampling=0, dataset=None):
    return metis_partition(G, parts, method, spmm_reorderd, sampling, dataset)

def spmm_part_array():
    return return_array()

import numpy as np
import torch
import dgl
_computed_array = None

def metis_partition(G, parts=None, method=None):
    global _computed_array
    if _computed_array is None:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        print(G)
        print(type(G))
        print("partition start")
        Nodes = G.num_nodes() 
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)

        # print(g)
        if method is None:
            _computed_array = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
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
        print("Array computation done and passed to neighbour.py line 631")
    return _computed_array

def get_part_array(G, parts=None, method=None):
    # print("array passed")
    return metis_partition(G, parts, method)

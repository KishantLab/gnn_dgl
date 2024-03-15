from numpy import array
import torch
import numpy as np
import pandas as pd
import sys
import dgl
import csv
import time
import torch as th
from scipy.io import mmread
import random
import os
from tqdm import tqdm
import psutil

totalTime =0
start = time.time()
file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr."
file_name = file_name.split("/")
file_name = file_name[len(file_name)-1]
out_filename1 = str(file_name) + suffix_csr
out_filename2 = str(file_name) + suffix_part + str(sys.argv[2])
print(out_filename2)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} bytes")
if file_extension == '.mtx':
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.DGLGraph()
    G.add_edges(u, v)
elif file_extension == '.tsv':
    columns = ['Source','Dest','Data']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns)
    print("Converting tsv2dgl..")
    print("This might a take while..")
    dest=file['Dest']
    dest=np.array(dest)
    print(file['Dest'])
    source=file['Source']
    source=np.array(source)
    G = dgl.graph((source,dest))
elif file_extension == '.txt':
    columns = ['Source','Dest']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,skiprows=4)
    print("Converting txt2dgl..")
    print("This might a take while..")
    dest=file['Dest']
    dest=np.array(dest)
    print(file['Dest'])
    source=file['Source']
    source=np.array(source)
    G = dgl.graph((source,dest))
elif file_extension == '.mmio':
    print("Converting mmio2dgl..")
    print("This might a take while..")
    a_mtx = mmread(sys.argv[1])
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    G = dgl.DGLGraph()
    G.add_edges(u, v)
elif file_extension == '.tsv_1':
    columns = ['Source','Dest']
    file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,low_memory=False,skiprows=1)
    print("Converting tsv2dgl..")
    print("This might a take while..")
    u=file['Dest']
    u=np.array(u)
    print(file['Dest'])
    v=file['Source']
    v=np.array(v)
    G = dgl.graph((v,u))
else:
    print(f"Unsupported file type: {file_extension}")
    exit("If file is TAB Saprated data then remove all comments in file and save it with extention .tsv \n NOTE: only .tsv (Graph Challange), .txt(snap.stanford), .mtx(suit_sparse), .mmio(all) files are supported")

end = time.time()
totalTime = totalTime + (end-start)
# del u
# del v

print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB")

#----------------------DGL PREPROCESS-----------------------------------#
start = time.time()
print("DGL GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.to_simple(G)
G = dgl.remove_self_loop(G)
print("DGL SIMPLE GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.add_reverse_edges(G)
G = dgl.to_bidirected(G)
print("DGL GRAPH CONSTRUCTION DONE \n",G)

isolated_nodes = ((G.in_degrees() == 0) & (G.out_degrees() == 0)).nonzero().squeeze(1)
G.remove_nodes(isolated_nodes)
print(G)

in_deg = np.array(G.in_degrees())
in_deg_s = len(in_deg)
# d_in_deg = cp.asarray(in_deg)
print(in_deg)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB")
Nodes = G.num_nodes()
Edges = G.num_edges()
end = time.time()
totalTime = totalTime + (end-start)
print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
#-------------------------------------------Graph Construction is done ----------#

#-------------------------------------DGL METIS GRAPH PARTITIONING------------------------#
#nopart = 2
nopart = int(sys.argv[2])
print("Start Partitioning.....")
start = time.time()
#n_cuts, node_parts = pymetis.part_graph(nopart, adjacency=adjacency_list)
#nodes_part = dgl.metis_partition_assignment(G, nopart, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
#parts = dgl.metis_partition(g, k, reshuffle=True, mode='k-way')
#parts = dgl.metis_partition(G, nopart, reshuffle=True)
node_parts = dgl.metis_partition_assignment(G,nopart)
end = time.time()
totalTime = totalTime + (end-start)
print("Partition is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} bytes")

#nodes_part = np.argwhere(np.array(membership) == i).ravel()
print("Partitions Contructions with halo nodes ..")
start = time.time()
parts, orig_nids, orig_eids=dgl.partition_graph_with_halo(G, node_parts, 1, reshuffle=True)
end = time.time()
totalTime = totalTime + (end-start)
print("Halo Node CONSTRUCTION is Done !!!!!\t Time of construction is :",round((end-start),4), "Seconds")

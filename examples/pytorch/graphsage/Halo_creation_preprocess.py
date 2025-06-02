#new created files
#this is the contigeous partition of graph were vertex assigned in partition based on contigeous manner
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
import cupy as cp
import psutil
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset
import argparse

parser = argparse.ArgumentParser()

Find_size = cp.RawKernel(r'''
extern "C" __global__
void Find_size(unsigned long long int *d_in_deg, unsigned long long int *d_org_id, unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr, unsigned long long int in_deg_s, unsigned long long int org_id_s, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    if(id < row_ptr_s)
    {
      unsigned long long int count = 0;
      //unsigned long long int deg_src = row_ptr[i+1] - row_ptr[i];
      unsigned long long int deg_src = d_in_deg[d_org_id[id]];
      //printf(" Deg_src : %llu ",deg_src);
      for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
      {
        //unsigned long long int deg_dst = row_ptr[col_index[j]+1] - row_ptr[col_index[j]];
        unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          count++;
          //col_idx_Dir[pos] = col_idx[j];
          //pos++;
        }
        else if(deg_src == deg_dst)
        {
          if(d_org_id[d_col_idx[j]] < d_org_id[id])
          {
            count++;
            //col_idx_Dir[pos] = col_idx[j];
             //pos++;
          }
        }
        //bar.update();
        temp_arr[id] = count;
        //printf("%llu",count);
        //row_ptr_Dir[i+1] = pos;
      }
    }
}
'''
, 'Find_size')

Convert = cp.RawKernel(r'''
extern "C" __global__
void Convert(unsigned long long int *d_in_deg, unsigned long long int *d_org_id, unsigned long long int *d_row_ptr, unsigned long long int *d_col_idx, unsigned long long int *temp_arr_sum, unsigned long long int *d_row_ptr_Dir, unsigned long long int *d_col_idx_Dir, unsigned long long int in_deg_s, unsigned long long int org_id_s, unsigned long long int row_ptr_s, unsigned long long int col_idx_s)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    unsigned long long int pos;
    if(id < row_ptr_s)
    {
        if(id==0)
        {
        pos = 0;
        d_row_ptr_Dir[0] = 0;
        }
        __syncthreads();
        if(id !=0)
        {
        pos = temp_arr_sum[id-1];
        }
        unsigned long long int deg_src = d_in_deg[d_org_id[id]];
        //printf(" Deg_src : %llu ",deg_src);
        for(unsigned long long int j=d_row_ptr[id]; j<d_row_ptr[id+1]; j++)
        {
        unsigned long long int deg_dst = d_in_deg[d_org_id[d_col_idx[j]]];
        //printf("  Deg_dst : %llu ",deg_dst);
        if(deg_src < deg_dst)
        {
          //count++;
          d_col_idx_Dir[pos] = d_col_idx[j];
          pos++;
        }
        else if(deg_src == deg_dst)
        {
          if(d_org_id[d_col_idx[j]] < d_org_id[id])
          {
            //count++;
            d_col_idx_Dir[pos] = d_col_idx[j];
            pos++;
          }
        }
        //bar.update();
        //temp_arr[id] = count;
        d_row_ptr_Dir[id+1] = pos;
        }
    }
}
''', 'Convert')

Halo_creation = cp.RawKernel(r'''
extern "C" __global__
void Halo_creation(unsigned long long int *d_row_ptr_G, unsigned long long int *d_col_idx_G, unsigned long long int *halo_nodes, unsigned long long int induced_nodes_s, unsigned long long int *d_induced_nodes, unsigned long long int *d_pos)
{
    unsigned long long int id = threadIdx.x + blockIdx.x * blockDim.x ;
    if(id < induced_nodes_s)
    {
        int node = d_induced_nodes[id];
        int start = d_row_ptr_G[node];
        int end = d_row_ptr_G[node+1];
        for(int i = start; i<end; i++){
            //printf("Edges: %llu %llu\n", node, d_col_idx_G[i]);
            int flag = 0;
            for(int j = 0; j<induced_nodes_s; j++){
                if(d_col_idx_G[i]==d_induced_nodes[j]){
                    flag = 1;
                    break;
                }
            }
            //printf("Flag: %d\n", flag);
            if (flag!=1)
            {
                //unsigned long long int j = 0;
                //unsigned long long int k = d_col_idx_G[j];
                unsigned long long int pos = atomicAdd(d_pos, 1);
                halo_nodes[pos] = d_col_idx_G[i];                
            }
        }
    }
}
''', 'Halo_creation')

#-------------------------------------Graph CONSTRUCTION USING data----------------#
totalTime =0
start = time.time()
# Check if GPU is available and set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
file_name, file_extension = os.path.splitext(sys.argv[1])
print(file_extension)
suffix_csr = "_output.csr"
suffix_part = "_part.csr"
# file_name = file_name.split("/")
# file_name = file_name[len(file_name)-1]
# out_filename1 = str(file_name) + suffix_csr
# out_filename2 = str(file_name) + suffix_part
# node_parts_outfilename2_random_2 = str(file_name) + suffix_part + "_node_parts_random_2"
# node_parts_outfilename2_random_4 = str(file_name) + suffix_part + "_node_parts_random_4"
# node_parts_outfilename2_random_8 = str(file_name) + suffix_part + "_node_parts_random_8"
# node_parts_outfilename2_contig_2 = str(file_name) + suffix_part + "_node_parts_contig_2"
# node_parts_outfilename2_contig_4 = str(file_name) + suffix_part + "_node_parts_contig_4"
# node_parts_outfilename2_contig_8 = str(file_name) + suffix_part + "_node_parts_contig_8"
# node_parts_outfilename2_metis_2 = str(file_name) + suffix_part + "_node_parts_metis_2"
# node_parts_outfilename2_metis_4 = str(file_name) + suffix_part + "_node_parts_metis_4"
# node_parts_outfilename2_metis_8 = str(file_name) + suffix_part + "_node_parts_metis_8"


# print(out_filename2)
# mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
# print(f"Current memory usage: { (mem_usage)} bytes")
# if file_extension == '.mtx':
#     print("Converting mtx2dgl..")
#     print("This might a take while..")
#     a_mtx = mmread(sys.argv[1])
#     coo = a_mtx.tocoo()
#     u = th.tensor(coo.row, dtype=th.int64)
#     v = th.tensor(coo.col, dtype=th.int64)
#     G = dgl.DGLGraph()
#     G.add_edges(u, v)
# elif file_extension == '.tsv':
#     columns = ['Source','Dest','Data']
#     file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns)
#     print("Converting tsv2dgl..")
#     print("This might a take while..")
#     dest=file['Dest']
#     dest=np.array(dest)
#     print(file['Dest'])
#     source=file['Source']
#     source=np.array(source)
#     G = dgl.graph((source,dest))
# elif file_extension == '.txt':
#     columns = ['Source','Dest']
#     file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,skiprows=4)
#     print("Converting txt2dgl..")
#     print("This might a take while..")
#     dest=file['Dest']
#     dest=np.array(dest)
#     print(file['Dest'])
#     source=file['Source']
#     source=np.array(source)
#     G = dgl.graph((source,dest))
# elif file_extension == '.mmio':
#     print("Converting mmio2dgl..")
#     print("This might a take while..")
#     a_mtx = mmread(sys.argv[1])
#     coo = a_mtx.tocoo()
#     u = th.tensor(coo.row, dtype=th.int64)
#     v = th.tensor(coo.col, dtype=th.int64)
#     G = dgl.DGLGraph()
#     G.add_edges(u, v)
# elif file_extension == '.tsv_1':
#     columns = ['Source','Dest']
#     file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,low_memory=False,skiprows=1)
#     print("Converting tsv2dgl..")
#     print("This might a take while..")
#     u=file['Dest']
#     u=np.array(u)
#     print(file['Dest'])
#     v=file['Source']
#     v=np.array(v)
#     G = dgl.graph((v,u))
# elif file_extension == '.edgelist':
#     columns = ['Source','Dest']
#     file = pd.read_csv(sys.argv[1],delimiter='\t',names=columns,low_memory=False)
#     print("Converting tsv2dgl..")
#     print("This might a take while..")
#     u=file['Dest']
#     u=np.array(u)
#     print(file['Dest'])
#     v=file['Source']
#     v=np.array(v)
#     G = dgl.graph((v,u))
# else:
#     print(f"Unsupported file type: {file_extension}")
#     exit("If file is TAB Saprated data then remove all comments in file and save it with extention .tsv \n NOTE: only .tsv (Graph Challange), .txt(snap.stanford), .mtx(suit_sparse), .mmio(all) files are supported")

# end = time.time()
# totalTime = totalTime + (end-start)
# del u
# del v
parser.add_argument(
        "--dataset",
        default="ogbn-products",
        # choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M", "reddit"],
        help="pass dataset",
)
args = parser.parse_args()
if args.dataset == "cora":
    dataset = CoraGraphDataset()
elif args.dataset == "citeseer":
    dataset = CiteseerGraphDataset()
elif args.dataset == "pubmed":
    dataset = PubmedGraphDataset()
elif args.dataset == "wisconsin":
    dataset = WisconsinDataset()
elif args.dataset == "flickr":
    dataset = FlickrDataset()
elif args.dataset == "reddit":
    dataset = RedditDataset()
elif args.dataset == "yelp":
    dataset = YelpDataset()
elif args.dataset == "ogbn-products":
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
elif args.dataset == "ogbn-arxiv":
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
else:
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
        # raise ValueError("Unknown dataset: {}".format(args.dataset))


G = dataset[0]

# print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
# mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
# print(f"Current memory usage: { (mem_usage)} GB")

#----------------------DGL PREPROCESS-----------------------------------#
start = time.time()
print("DGL GRAPH CONSTRUCTION DONE \n",G)
G = dgl.remove_self_loop(G)
print("DGL SIMPLE GRAPH CONSTRUCTION DONE \n",G)
#G = dgl.to_bidirected(G)
G = dgl.to_simple(G)
print("DGL GRAPH CONSTRUCTION DONE \n",G)

isolated_nodes = ((G.in_degrees() == 0) & (G.out_degrees() == 0)).nonzero().squeeze(1)
G.remove_nodes(isolated_nodes)
print(G)

in_deg = np.array(G.in_degrees())
in_deg_s = len(in_deg)
d_in_deg = cp.asarray(in_deg)
print(in_deg)
mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
print(f"Current memory usage: { (mem_usage)} GB")
Nodes = G.num_nodes()
Edges = G.num_edges()
end = time.time()
totalTime = totalTime + (end-start)
print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
#-------------------------------------------Graph Construction is done ----------#

#write csr indeg in file..
#--------------------------
out_filename2 = str(args.dataset) + suffix_csr
file = open(out_filename2,'w')
file.write("%i " % Nodes)
file.write("%i\n" % Edges)
row_ptr_G = np.array(G.adj_tensors('csr')[0])
col_idx_G = np.array(G.adj_tensors('csr')[1])
file.write("%i " % (len(row_ptr_G)-1))
file.write("%i " % len(col_idx_G))
file.write("\n")
for data in range(len(row_ptr_G)):
    file.write("%i " % row_ptr_G[data])
file.write("\n")
for data in range(len(col_idx_G)):
    file.write("%i " % col_idx_G[data])
file.write("\n")
for data in range(in_deg_s):
    file.write("%i " % in_deg[data])
file.write("\n")
file.close()

#-------------------------------------DGL METIS GRAPH PARTITIONING------------------------#
#nopart = 2
#number of partition is 2 4 8
# nopart = int(sys.argv[2])
# print("Start Partitioning.....")
# total_part_time = 0
# start_part_time = time.time()
# start = time.time()
# node_parts_random_2 = np.random.randint(0, 2, size=Nodes)
# end = time.time()
# print("2 Partition for random is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# start = time.time()
# l = Nodes // 2   # calculate the number of repeated values for each number

# node_parts_contig_2 = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros

# for i in range(2):
#     node_parts_contig_2[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
# end = time.time()
# print("2 Partition for contig is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")
# # print(node_parts_contig)  # print the resulting array

# start = time.time()
# node_parts_metis_2 = dgl.metis_partition_assignment(G,2)
# end = time.time()
# print("2 Partition for metis is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# file2 = open(node_parts_outfilename2_random_2,'w')
# for data in range(len(row_ptr_G)-1):
#     file2.write("%i " % node_parts_random_2[data])
# file2.write("\n")
# file2.close()
# file3 = open(node_parts_outfilename2_contig_2,'w')
# for data in range(len(row_ptr_G)-1):
#     file3.write("%i " % node_parts_contig_2[data])
# file3.write("\n")
# file3.close()
# file4 = open(node_parts_outfilename2_metis_2,'w')
# for data in range(len(row_ptr_G)-1):
#     file4.write("%i " % node_parts_metis_2[data])
# file4.write("\n")
# file4.close()

# start = time.time()
# node_parts_random_4 = np.random.randint(0, 4, size=Nodes)
# end = time.time()
# print("4 Partition for random is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# start = time.time()
# l = Nodes // 4   # calculate the number of repeated values for each number

# node_parts_contig_4 = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros

# for i in range(4):
#     node_parts_contig_4[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
# # print(node_parts_contig)  # print the resulting array
# end = time.time()
# print("4 Partition for contig is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# start = time.time()
# node_parts_metis_4 = dgl.metis_partition_assignment(G,4)
# end = time.time()
# print("4 Partition for metis is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# file5 = open(node_parts_outfilename2_random_4,'w')
# for data in range(len(row_ptr_G)-1):
#     file5.write("%i " % node_parts_random_4[data])
# file5.write("\n")
# file5.close()
# file6 = open(node_parts_outfilename2_contig_4,'w')
# for data in range(len(row_ptr_G)-1):
#     file6.write("%i " % node_parts_contig_4[data])
# file6.write("\n")
# file6.close()
# file7 = open(node_parts_outfilename2_metis_4,'w')
# for data in range(len(row_ptr_G)-1):
#     file7.write("%i " % node_parts_metis_4[data])
# file7.write("\n")
# file7.close()

# start = time.time()
# node_parts_random_8 = np.random.randint(0, 8, size=Nodes)
# end = time.time()
# print("8 Partition for random is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# start = time.time()
# l = Nodes // 8   # calculate the number of repeated values for each number

# node_parts_contig_8 = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros

# for i in range(8):
#     node_parts_contig_8[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
# # print(node_parts_contig)  # print the resulting array
# end = time.time()
# print("8 Partition for contig is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# start = time.time()
# node_parts_metis_8 = dgl.metis_partition_assignment(G,8)
# end = time.time()
# print("8 Partition for metis is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")

# file8 = open(node_parts_outfilename2_random_8,'w')
# for data in range(len(row_ptr_G)-1):
#     file8.write("%i " % node_parts_random_8[data])
# file8.write("\n")
# file8.close()
# file9 = open(node_parts_outfilename2_contig_8,'w')
# for data in range(len(row_ptr_G)-1):
#     file9.write("%i " % node_parts_contig_8[data])
# file9.write("\n")
# file9.close()
# file10 = open(node_parts_outfilename2_metis_8,'w')
# for data in range(len(row_ptr_G)-1):
#     file10.write("%i " % node_parts_metis_8[data])
# file10.write("\n")
# file10.close()


# totalTime = totalTime + (end-start)
# mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
# print(f"Current memory usage: { (mem_usage)} bytes")

# # file2.write("%i " % Nodes)
# # file2.write("%i\n" % Edges)

# end_part_time = time.time()
# total_part_time = total_part_time + (end_part_time - start_part_time)
# # total_clean_time = 0
# d_row_ptr_G = cp.asarray(row_ptr_G)
# d_col_idx_G = cp.asarray(col_idx_G)

# g = G.to(device)
# d_node_parts = torch.tensor(node_parts, device=device)
# totalHaloTime = 0

# Loop over each unique value
# for value in range(nopart):
#     print(value,"th","Partitions Contructions with halo nodes ..")
#     start_part_time = time.time()
#     #-----------finding the indices of the partition------------
#     start = time.time()
#     print("Finding Indices...")
#     start_halo = time.time()
#     indices = torch.nonzero(d_node_parts == value, as_tuple=False).squeeze()
#     print(indices)
#     induced_nodes = indices.to(device)dir
#     induced_nodes_s = len(induced_nodes)
#     d_induced_nodes = cp.asarray(induced_nodes)
#     d_pos = cp.zeros(1, dtype=cp.uint64)
#     end_halo = time.time()
#     print("index_array created!!!!\t time of construction is :", round((end_halo - start_halo),4), "seconds")
#     d_halo_nodes = cp.full(len(row_ptr_G), -1)
#     #------------------------ finding the predicessor and succesorr for nodes-----
#     start_halo = time.time()
#     # successors = torch.cat([G.successors(induced_nodes[i]).to(device) for i in range(induced_nodes.shape[0])])
#     # predecessors = torch.cat([G.predecessors(induced_nodes[i]).to(device) for i in range(induced_nodes.shape[0])])
#     # all_neighbors = torch.cat([successors, predecessors]).unique()
#     # halo_nodes = all_neighbors[~torch.isin(all_neighbors, induced_nodes)]
#     # print(halo_nodes)
#     block_size = 1024
#     grid_size_G = (len(row_ptr_G) + block_size - 1) // block_size
#     print(induced_nodes_s)
#     Halo_creation((grid_size_G,), (block_size,), (d_row_ptr_G, d_col_idx_G, d_halo_nodes, induced_nodes_s, d_induced_nodes,d_pos))
#     end_halo = time.time()
#     print("Finding Halo nodes done !!!\t time of finding is : ",round((end_halo - start_halo),4),"seconds")
#     # index = cp.argmax(d_halo_nodes==-1)
#     # if d_halo_nodes[index]==-1:
#     #     # Modify the array to keep only the elements before the first -1
#     #     d_halo_nodes = d_halo_nodes[:index]
#     final_halo_nodes = torch.empty(0, dtype=torch.int64)
#     if d_halo_nodes is not None:
#         try:
#             halo_nodes = d_halo_nodes.get()  # Transfer to host (CPU)
#             if halo_nodes.size > 0:  # Ensure it's not empty
#                 filtered_halo_nodes = halo_nodes[halo_nodes != -1]
#                 filtered_halo_nodes = np.unique(filtered_halo_nodes)
#                 # print("Filtered halo nodes:", filtered_halo_nodes)
#                 final_halo_nodes = torch.tensor(filtered_halo_nodes)
#                 # print(halo_nodes)
#             else:
#                 print("Halo nodes array is empty.")
#         except Exception as e:
#             print("Error during d_halo_nodes.get():", e)
#     else:
#         print("d_halo_nodes is None.")
#     # print(filtered_halo_nodes)
#     del d_halo_nodes
#     del d_induced_nodes
#     #-------------------------creting the induced subgraph with halo-----------------------------------
#     start_halo = time.time()
#     # Combine induced subgraph and halo nodes
#     combined_nodes = torch.cat([induced_nodes, final_halo_nodes])
#     induced_with_halo_subgraph = dgl.node_subgraph(G, combined_nodes, relabel_nodes=True, store_ids=True)
#     end_halo = time.time()
#     print("Induced subgraph with halo created!!! \t time of construction is :", round((end_halo - start_halo),4), "seconds")


#     print(induced_with_halo_subgraph)
#     end = time.time()
#     totalHaloTime = totalHaloTime + (end_halo-start_halo)
#     totalTime = totalTime + (end-start)
#     print("Halo Node CONSTRUCTION is Done !!!!!\t Time of construction is :",round((end-start),4), "Seconds")

#     start_halo = time.time()
#     # Label inner nodes (induced nodes) vs halo nodes
#     inner_node = torch.zeros(combined_nodes.shape[0], dtype=torch.bool, device=device)
#     inner_node[:len(induced_nodes)] = True  # Mark induced nodes as True
#     induced_with_halo_subgraph.ndata['inner_node'] = inner_node

#     end_halo = time.time()
#     print("Inner node creation done !!! \t Time of construction is :", round((end_halo - start_halo),4), "seconds")
#     totalTime = totalTime + (end_halo-start_halo)
#     org_id = combined_nodes
#     org_id_s = len(org_id)

#     v_arr = np.array(induced_with_halo_subgraph.ndata['inner_node'].cpu())
#     #len(np.array(parts[i].ndata['inner_node']))
#     t_ver = np.sum(v_arr)
#     v_arr_s = len(v_arr)
#     row_ptr_s = len(np.array(induced_with_halo_subgraph.adj_tensors('csr')[0].cpu()))
#     col_idx_s = len(np.array(induced_with_halo_subgraph.adj_tensors('csr')[1].cpu()))
#     row_ptr = np.array(induced_with_halo_subgraph.adj_tensors('csr')[0].cpu())
#     col_idx = np.array(induced_with_halo_subgraph.adj_tensors('csr')[1].cpu())
#     # Sort the column indices within each row range specified by row_ptr
    # for x in range(len(row_ptr) - 1):
    #     col_idx[row_ptr[x]:row_ptr[x + 1]] = np.sort(col_idx[row_ptr[x]:row_ptr[x + 1]])
#     mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
#     print(f"Current memory usage: { (mem_usage)} GB")

#     print("Converting UNDIR ---> DIR")
#     start1 = time.time()

#     d_org_id = cp.asarray(org_id)
#     d_row_ptr = cp.asarray(row_ptr)
#     d_col_idx = cp.asarray(col_idx)
#     temp_arr = cp.empty_like(row_ptr)
#     N =0
#     # Call the add kernel function on the GPU
#     block_size = 1024
#     grid_size = (row_ptr_s + block_size - 1) // block_size
#     print(row_ptr_s)
#     Find_size((grid_size,), (block_size,), (d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr, in_deg_s, org_id_s, row_ptr_s, col_idx_s))

#     # Print the result

#     #print(temp_arr)
#     temp_arr_sum = cp.cumsum(temp_arr)
#     #print(temp_arr_sum)
#     temp_arr_sum_s = len(temp_arr_sum)

#     d_row_ptr_Dir = cp.empty_like(d_row_ptr)
#     #d_col_idx_Dir = cp.empty_like(d_col_idx)
#     dtype = type(col_idx)
#     N = int(temp_arr_sum[temp_arr_sum_s-1])
#     d_col_idx_Dir = cp.empty(N, dtype=col_idx.dtype)

#     #Convert<<<nblocks,BLOCKSIZE>>>(d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, in_deg_s, org_id_s, row_ptr_s, col_idx_s);
#     Convert((grid_size,), (block_size,), (d_in_deg, d_org_id, d_row_ptr, d_col_idx, temp_arr_sum, d_row_ptr_Dir, d_col_idx_Dir, in_deg_s, org_id_s, row_ptr_s, col_idx_s))

#     # create host array to hold selected elements
#     num_elements = N
#     col_idx_dir = np.empty(num_elements, dtype=d_col_idx_Dir.dtype)


#     # copy selected elements from device array to host array
#     start_index = 0
#     end_index = start_index + num_elements
#     col_idx_dir[start_index:end_index] = d_col_idx_Dir[:N].get()

#     row_ptr_dir = d_row_ptr_Dir.get()

#     print(len(col_idx))
#     print(len(col_idx_dir))

#     end1 = time.time()
#     totalTime = totalTime + (end1-start1)
#     print("Converting is done !!!!! Time taken: ",round((end1-start1),4))
#     mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
#     print(f"Current memory usage: { (mem_usage)} GB")

#     #--------------------writting CSR to file---------------------------------------#
#     start = time.time()
#     G_dir = dgl.graph(('csr', (row_ptr_dir, col_idx_dir, [])))
#     print(G_dir)
#     G_dir = G_dir.to(device)
#     while True:
#         start_degree_time = time.time()
#         G_indeg = G_dir.in_degrees()
#         end_degree_time = time.time()
#         print("Indegree calculation time: ", round((end_degree_time-start_degree_time), 4), "Seconds")
#         total_nodes = (G_dir.num_nodes())
#         in_deg = []
#         print("total nodes",total_nodes)
#         start_for_loop_time = time.time()
#         # for i in range(t_ver,total_nodes):
#         #     if G_indeg[i]==0:
#         #         in_deg.append(i)
#         G_indeg = cp.asarray(G_indeg)
#         indices = cp.arange(t_ver, total_nodes)
#         zero_indeg_indices = indices[G_indeg[indices] == 0]
#         in_deg = zero_indeg_indices.tolist()
#         end_for_loop_time = time.time()
#         print("For loop execution time: ", round((end_for_loop_time-start_for_loop_time), 4), "Seconds")
#         if len(in_deg)==0:
#             break
#         start_remove_time = time.time()
#         G_dir.remove_nodes(in_deg)
#         end_remove_time = time.time()
#         print("Node Removal time: ", round((end_remove_time-start_remove_time), 4), "Seconds")
#     print(G_dir)
#     G_dir = G_dir.to('cpu')
#     clean_row_ptr = G_dir.adj_tensors('csr')[0]
#     clean_col_ptr = G_dir.adj_tensors('csr')[1]
#     end = time.time()
#     end_part_time = time.time()
#     total_part_time = total_part_time + (end_part_time - start_part_time)
#     total_clean_time = total_clean_time + (end-start)
#     totalTime = totalTime + (end-start)
#     print("Duplicate edge deletion is done !!!!! Time taken: ",round((end-start),4),"Seconds")
#     # Wrriting cleaned data into file
#     file2.write("%i " % G_dir.num_nodes())
#     file2.write("%i " % len(clean_row_ptr))
#     file2.write("%i " % len(clean_col_ptr))
#     file2.write("%i " % t_ver)
#     file2.write("\n")
#     for data in range(len(clean_row_ptr)):
#         file2.write("%i " % clean_row_ptr[data])
#     file2.write("\n")
#     for data in range(len(clean_col_ptr)):
#         file2.write("%i " % clean_col_ptr[data])
#     file2.write("\n")

#     #-----Wrriting Original data into file------
#     start = time.time()
#     file.write("%i " % v_arr_s)
#     file.write("%i " % len(row_ptr_dir))
#     file.write("%i " % len(col_idx_dir))
#     file.write("%i " % t_ver)
#     file.write("\n")
#     #for data in range(v_arr_s):
#         #file.write("%i " % v_arr[data])
#     #file.write("\n")
#     for data in range(len(row_ptr_dir)):
#         file.write("%i " % row_ptr_dir[data])
#     file.write("\n")
#     for data in range(len(col_idx_dir)):
#         file.write("%i " % col_idx_dir[data])
#     file.write("\n")
#     end = time.time()
#     print("Writing is done !!!!! Time taken: ",round((end-start),4),"Seconds")
#     totalTime = totalTime + (end-start)
#     mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
#     print(f"Current memory usage: { (mem_usage)} GB")

#     del clean_col_ptr
#     del clean_row_ptr
#     #temp_arr.free()
#     del d_org_id
#     del d_row_ptr
#     del d_col_idx
#     del temp_arr
#     del temp_arr_sum
#     del d_row_ptr_Dir
#     del d_col_idx_Dir
#     del org_id
#     del row_ptr
#     del col_idx
#     del row_ptr_dir
#     del col_idx_dir
#     #cp.cuda.runtime.free(intptr_t temp_arr)
#     cp._default_memory_pool.free_all_blocks()
# del d_in_deg
# del d_row_ptr_G
# del d_col_idx_G
file.close()
# print("Total Clean Time: ", round(total_clean_time,4), "Seconds")
# print("Total Partition Time: ", round(total_part_time,4), "Seconds")
# print("Total Halo Nodes Generation Time: ", round(totalHaloTime, 4), "Seconds")
print("Data Preprocessing is Successfull Total Time taken: ",round(totalTime,4),"Seconds")

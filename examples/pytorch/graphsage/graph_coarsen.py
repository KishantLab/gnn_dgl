#!/data/kishan/anaconda3/envs/dgl-dev-gpu-117/bin/python3
import argparse
import numpy as np
import dgl
import time
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import cupy as cp
from tqdm import tqdm
# import tqdm
from dgl.metis_sampling import *
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset


csr_to_coo = cp.RawKernel(r'''
extern "C" __global__
void csr_to_coo(const int* indptr, int* row_ids, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        int start = indptr[row];
        int end   = indptr[row+1];
        for (int eid = start; eid < end; ++eid) {
            row_ids[eid] = row;
        }
    }
}
''', "csr_to_coo")

def csr_to_directed_csr_gpu(indptr, indices):
    # ensure GPU arrays
    indptr = cp.asarray(indptr, dtype=cp.int32)
    indices = cp.asarray(indices, dtype=cp.int32)

    n = indptr.shape[0] - 1
    deg = indptr[1:] - indptr[:-1]
    nnz = indices.shape[0]


    # expand CSR -> COO
    # total_edges = indices.shape[0]
    # row_ids = cp.empty(total_edges, dtype=cp.int32)
    # for i in tqdm(range(n), desc="Expanding CSR to COO"):
    #     row_ids[indptr[i]:indptr[i+1]] = i
    # row_counts = degree of each node
    row_ids = cp.empty(nnz, dtype=cp.int32)
    # col_ids = cp.empty_like(indices)

    threads = 256
    blocks = (n + threads - 1) // threads
    csr_to_coo((blocks,), (threads,),
           (indptr, row_ids, n))

    col_ids = indices

    # only keep u < v (avoid duplicates)
    mask = row_ids < col_ids
    u = row_ids[mask]
    v = col_ids[mask]

    deg_u = deg[u]
    deg_v = deg[v]

    # orientation
    cond1 = deg_u > deg_v
    cond2 = deg_u < deg_v
    cond3 = ~(cond1 | cond2)

    src = cp.empty_like(u)
    dst = cp.empty_like(v)

    src[cond1], dst[cond1] = v[cond1], u[cond1]
    src[cond2], dst[cond2] = u[cond2], v[cond2]
    src[cond3] = cp.minimum(u[cond3], v[cond3])
    dst[cond3] = cp.maximum(u[cond3], v[cond3])

    # sort by src
    order = cp.argsort(src)
    src, dst = src[order], dst[order]

    # build new CSR
    indptr_new = cp.zeros(n+1, dtype=cp.int32)
    cp.add.at(indptr_new, src+1, 1)   # count edges per row
    indptr_new = cp.cumsum(indptr_new)

    return indptr_new, dst.astype(cp.int32)

if __name__ == "__main__":

    print("inside the main")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--dataset",
        default="yelp",
        # choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M", "reddit"],
        help="pass dataset",
    )
    parser.add_argument(
        "--batch_size",
        default="1024",
        # choices=["1024", "2048", "4096", "8192"],
        help="batch_size for train",
    )
    parser.add_argument(
        "--epoch",
        default="1",
        help="batch_size for train",
    )
    parser.add_argument(
        "--method",
        type=str,
        default = None,
        choices=["metis", "rm", "contig"],
        help="Partition method for sampling"
    )
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--spmm", default="cusparse")
    parser.add_argument("--sampling", default="default")
    parser.add_argument("--sampler", default="default")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"\nTraining in {args.mode} mode.")

    # load and preprocess dataset
    # print("\nLoading data")
    # dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    # load and preprocess dataset
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
    elif args.dataset == "igb-tiny":
        dataset, meta = dgl.load_graphs("dataset/igb_datasets/igb_tiny.dgl")
    elif args.dataset == "igb-small":
        dataset, meta = dgl.load_graphs("dataset/igb_datasets/igb_small.dgl")
    elif args.dataset == "igb-medium":
        dataset, meta = dgl.load_graphs("dataset/igb_datasets/igb_medium.dgl")
    elif args.dataset == "igb-large":
        dataset, meta = dgl.load_graphs("dataset/igb_datasets/igb_large.dgl")
    elif args.dataset == "amazon-products":
        dataset, meta = dgl.load_graphs("/data/Dataset/gnn_dataset/amazon_products.dgl")
    elif args.dataset == "wiki5M":
        dataset, meta = dgl.load_graphs("/data/Dataset/gnn_dataset/wikidata5M/wikidata5m_dgl_graph.bin")
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
        # raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.spmm == "cusparse":
        spmm_method = 0
    elif args.spmm == "respmm":
        spmm_method = 1
    elif args.spmm == "gespmm":
        spmm_method = 2
    else:
        print("please provide valid spmm mathod like respmm or gespmm. default value is cusparse")
        
    if args.sampling == "metis":
        sampling_method = 0
    elif args.sampling == "default":
        sampling_method = 1
    else:
        print("please provide valid sampling mathod like metis (0) or default (1). default value is cusparse")

    g = dataset[0]
    print(g)

    out_degrees = np.array(g.out_degrees())
    in_degrees = np.array(g.in_degrees())
    if np.sum(out_degrees) == np.sum(in_degrees):
        print("graph is undirected")
    else:
        print("graph is direct")
        start = time.time()
        sym_g = dgl.to_bidirected(g)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
        print("Convert a graph into a bidirected graph: {:.3f} seconds".format(
            time.time() - start
        ))

    # 🔹 Example
    indptr = np.array(g.adj_tensors('csr')[0])
    indices = np.array(g.adj_tensors('csr')[1])
    
    start_converting_time = time.time()
    indptr_new, indices_new = csr_to_directed_csr_gpu(indptr, indices)
    print("Time taken to convert undirected csr to directed csr: {:.3f} seconds".format(
        time.time() - start_converting_time
    ))
     # verify the result

    print("Directed CSR indptr:", indptr_new.get())
    print("Directed CSR indices:", indices_new.get())
    print("len of indices:", len(indices_new.get()))
    print("len of indptr:", len(indptr_new.get()))

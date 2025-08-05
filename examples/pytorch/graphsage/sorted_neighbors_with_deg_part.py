
import argparse
import numpy as np
import dgl
import time
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
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

import cupy as cp
# def sort_neighbors_by_deg_then_part_csr(indptr, indices, deg, part):

import cupy as cp

csr_sort_kernel = cp.RawKernel(r'''
extern "C" __global__
void csr_sort_by_deg(const int* indptr, int* indices,
                     const int* deg, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    int len = end - start;

    // Simple selection sort on GPU per row
    for (int i = 0; i < len - 1; ++i) {
        int max_idx = i;
        int max_deg = deg[indices[start + i]];
        for (int j = i + 1; j < len; ++j) {
            int cur_deg = deg[indices[start + j]];
            if (cur_deg > max_deg) {
                max_deg = cur_deg;
                max_idx = j;
            }
        }
        // Swap
        if (max_idx != i) {
            int temp = indices[start + i];
            indices[start + i] = indices[start + max_idx];
            indices[start + max_idx] = temp;
        }
    }
}
''', 'csr_sort_by_deg')

def gpu_sort_csr_by_deg(indptr, indices, deg):
    indptr = cp.asarray(indptr, dtype=cp.int32)
    indices = cp.asarray(indices, dtype=cp.int32)
    deg = cp.asarray(deg, dtype=cp.int32)

    num_rows = indptr.size - 1

    threads_per_block = 128
    blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block

    csr_sort_kernel((blocks_per_grid,), (threads_per_block,),
                    (indptr, indices, deg, num_rows))

    return indices


def sort_neighbors_segmented_csr(indptr, indices, deg, part):
    indptr = cp.asarray(indptr)
    indices = cp.asarray(indices)
    deg = cp.asarray(deg)
    part = cp.asarray(part)

    sorted_indices = cp.empty_like(indices)

    for i in tqdm(range(indptr.size - 1), desc="Sorting neighbors"):
        start = indptr[i].item()
        end = indptr[i + 1].item()

        if start == end:
            continue

        nbrs = indices[start:end]
        degs = deg[nbrs]
        parts = part[nbrs]

        # Step 1: stable sort by degree DESC
        sort_by_deg = cp.argsort(-degs, kind='stable')
        nbrs_deg_sorted = nbrs[sort_by_deg]
        parts_deg_sorted = parts[sort_by_deg]

        # Step 2: stable sort by part ASC
        sort_by_part = cp.argsort(parts_deg_sorted, kind='stable')
        final_sorted = nbrs_deg_sorted[sort_by_part]

        sorted_indices[start:end] = final_sorted

    return sorted_indices


def sort_neighbors_segmented_csr_deg(indptr, indices, deg):
    # Ensure everything is on GPU
    indptr = cp.asarray(indptr)
    indices = cp.asarray(indices)
    deg = cp.asarray(deg)
    # part = cp.asarray(part)

    sorted_indices = cp.empty_like(indices)

    for i in tqdm(range(indptr.size - 1), desc="Sorting neighbors"):
        start = indptr[i].item()
        end = indptr[i + 1].item()

        if start == end:
            continue  # skip empty rows

        nbrs = indices[start:end]
        # parts = part[nbrs]
        degs = deg[nbrs]

        # Stack keys for lexsort: shape must be (2, N)
        # keys = cp.stack([-parts, degs], axis=0)
        # keys = cp.stack([-deg, parts], axis=1)
        # sort_order = cp.lexsort(keys)
        sort_order = cp.argsort(-degs)  # descending degree
        # sort_order = cp.argsort(-sort_order)  # descending degree

        sorted_indices[start:end] = nbrs[sort_order]

    return sorted_indices

def sort_neighbors_by_degree_desc(indptr, indices, deg):
    # Ensure everything is on GPU
    indptr = cp.asarray(indptr)
    indices = cp.asarray(indices)
    deg = cp.asarray(deg)

    # Prepare
    N = indptr.size - 1  # number of rows
    total_nbrs = indices.size

    # Generate segment ids for each neighbor
    seg_ids = cp.empty_like(indices)
    for i in tqdm(range(N), desc="Generating segment IDs"):
        start = indptr[i]
        end = indptr[i + 1]
        seg_ids[start:end] = i

    # Get degrees of neighbors
    nbr_deg = deg[indices]

    # Sort key = degree, segment = seg_ids
    # Since CuPy doesn't support segmented sort directly, simulate via lexsort
    # Combine segment id and degree to create a composite key

    # Convert degrees to float (if not already)
    nbr_deg = nbr_deg.astype(cp.float32)

    # To simulate segment-wise descending sort: reverse deg and use seg_ids
    # We sort by (seg_id, -deg)
    # sort_keys = cp.lexsort(( -nbr_deg, seg_ids ))
    sort_keys = cp.lexsort(cp.stack((-nbr_deg, seg_ids)))


    # Apply the sort
    sorted_indices = indices[sort_keys]

    return sorted_indices



def sort_neighbors_segmented_csr2(indptr, indices, deg, part):
    # Ensure inputs are all on GPU
    indptr = cp.asarray(indptr)
    indices = cp.asarray(indices)
    deg = cp.asarray(deg)
    part = cp.asarray(part)

    sorted_indices = cp.empty_like(indices)

    for i in range(indptr.size - 1):
        start = indptr[i].item()
        end = indptr[i + 1].item()

        nbrs = indices[start:end]
        parts = part[nbrs]
        degs = deg[nbrs]

        # Ensure both are cupy arrays
        parts = cp.asarray(parts)
        degs = cp.asarray(degs)

        sort_order = cp.lexsort((-degs, parts))  # part asc, degree desc
        sorted_indices[start:end] = nbrs[sort_order]

    return sorted_indices



def sort_neighbors_segmented_csr1(indptr, indices, deg, part):
    """
    Sort neighbors in CSR format row-wise by (part[nbr], -deg[nbr]).
    
    Parameters:
    - indptr: (N+1,) CSR row pointer
    - indices: (nnz,) neighbor list
    - deg: (num_nodes,) degrees of each node
    - part: (num_nodes,) partition of each node
    
    Returns:
    - sorted_indices: (nnz,) row-wise sorted by part asc, degree desc
    """
    sorted_indices = cp.empty_like(indices)

    for i in range(indptr.size - 1):
        start = indptr[i].item()
        end = indptr[i + 1].item()

        nbrs = indices[start:end]
        parts = part[nbrs]
        degs = deg[nbrs]

        # Sort by (part ASC, degree DESC)
        keys = cp.stack([parts, -degs], axis=1)
        sort_order = cp.lexsort(( -degs, parts ))  # primary=part, secondary=-deg
        sorted_indices[start:end] = nbrs[sort_order]

    return sorted_indices




row_id_kernel = cp.RawKernel(r'''
extern "C" __global__
void generate_row_ids(const int* indptr, int* row_ids, int num_rows) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    for (int i = start; i < end; ++i) {
        row_ids[i] = row;
    }
}
''', 'generate_row_ids')
# This function computes the row IDs for each neighbor in a CSR graph.

def compute_row_ids(indptr):
    num_rows = indptr.size - 1
    nnz = indptr[-1].item()
    row_ids = cp.empty(nnz, dtype=cp.int32)

    threads_per_block = 128
    blocks = (num_rows + threads_per_block - 1) // threads_per_block

    row_id_kernel((blocks,), (threads_per_block,), (indptr, row_ids, num_rows))
    return row_ids


def compute_row_ids1(indptr):
    nnz = indptr[-1].item()  # total number of edges
    row_ids = cp.empty(nnz, dtype=cp.int32)

    kernel = cp.ElementwiseKernel(
        in_params='raw int32 indptr, int32 num_rows',
        out_params='raw int32 row_ids',
        operation="""
        for (int row = 0; row < num_rows; ++row) {
            int start = indptr[row];
            int end = indptr[row + 1];
            for (int i = start; i < end; ++i) {
                row_ids[i] = row;
            }
        }
        """,
        name='generate_row_ids',
        preamble='#include <stdint.h>'
    )

    # Launch the kernel (1 element == 1 thread for each row)
    num_rows = indptr.size - 1
    kernel(indptr, num_rows, row_ids)
    return row_ids


def gpu_sort_neighbors_by_part_and_degree(indptr, indices, deg, part):
    # Step 1: Compute row IDs from indptr
    row_ids = compute_row_ids(indptr)              # (nnz,)
    
    # Step 2: Get composite key: (row_id, part[nbr], -deg[nbr])
    nbr_part = part[indices]
    nbr_neg_deg = -deg[indices]

    # Step 3: stack keys for lexsort (lexsort sorts rows, so shape must be (k, N))
    # Primary key = row_id, then part, then -deg
    keys = cp.stack([row_ids, nbr_part, nbr_neg_deg], axis=0)

    # Step 4: Perform lexsort
    sorted_order = cp.lexsort(keys)

    # Step 5: Reorder indices using the computed permutation
    sorted_indices = indices[sorted_order]

    return sorted_indices



def gpu_sort_neighbors_by_part_and_degree3(indptr, indices, deg, part):
    row_ids = compute_row_ids(indptr)           # (nnz,)
    nbr_parts = part[indices]                   # (nnz,)
    nbr_neg_deg = -deg[indices]                 # (nnz,)

    # Lexsort with primary = row, then partition, then -degree
        # stack into shape (3, nnz): lexsort uses last axis as rows
    keys = cp.stack([row_ids, nbr_parts, nbr_neg_deg], axis=0)
    sort_order = cp.lexsort(keys)                   # returns indices for sorted order
    # sort_order = cp.lexsort((nbr_neg_deg, nbr_parts, row_ids))
    sorted_indices = indices[sort_order]
    return sorted_indices



def gpu_sort_neighbors_by_part_and_degree2(indptr, indices, deg, part):
    num_nodes = indptr.size - 1

    # Step 1: get row_id for each neighbor (csr row id)
    row_ids = compute_row_ids(indptr)

    # Step 2: generate sort keys (partition ASC, degree DESC)
    nbr_parts = part[indices]
    nbr_neg_deg = -deg[indices]

    # Step 3: lexsort: row_ids ASC ➜ part ASC ➜ -deg DESC
    sort_order = cp.lexsort((nbr_neg_deg, nbr_parts, row_ids))

    # Step 4: apply sort
    sorted_indices = indices[sort_order]
    return sorted_indices


def gpu_sort_neighbors_by_part_and_degree1(indptr, indices, deg, part):
    """
    Parameters
    ----------
    indptr : (N+1,) cupy.ndarray
    indices : (nnz,) cupy.ndarray
    deg : (num_nodes,) cupy.ndarray
    part : (num_nodes,) cupy.ndarray
    
    Returns
    -------
    sorted_indices : (nnz,) cupy.ndarray
    """
    num_nodes = indptr.size - 1
    nnz = indices.size

    # For each edge, determine which row (source node) it belongs to
    row_ids = cp.repeat(cp.arange(num_nodes, dtype=cp.int32), indptr[1:] - indptr[:-1])

    # For each neighbor in indices, get (part, -deg)
    nbr_parts = part[indices]                          # Partition ID of each neighbor
    nbr_neg_deg = -deg[indices]                        # Degree (descending sort => use -deg)

    # Perform lexsort by segment: (row_id, part, -degree)
    sort_keys = cp.lexsort((nbr_neg_deg, nbr_parts, row_ids))  # lexsort uses last key as primary

    sorted_indices = indices[sort_keys]
    return sorted_indices



def sort_neighbors_by_part_and_degree(indptr, indices, deg, part):
    sorted_indices = np.empty_like(indices)
    
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        nbrs = indices[start:end]
        
        # Composite key: (partition ascending, degree descending)
        sort_keys = [(part[n], -deg[n]) for n in nbrs]
        sorted_nbrs = [x for _, x in sorted(zip(sort_keys, nbrs))]
        
        sorted_indices[start:end] = sorted_nbrs
    
    return sorted_indices
# Sample CSR graph
indptr = np.array([0, 2, 6, 9, 12, 14])
indices = np.array([1, 2, 0, 2, 3, 4, 0, 1, 3, 1, 2, 4, 1, 3])
deg = np.array([2, 4, 3, 3, 2])
part = np.array([1, 0, 0, 1, 0])

# Apply sorting
sorted_indices = sort_neighbors_by_part_and_degree(indptr, indices, deg, part)

print("Sorted Indices:", sorted_indices)


# Convert all inputs to cupy
indptr_cp = cp.asarray(indptr)
indices_cp = cp.asarray(indices)
deg_cp = cp.asarray(deg)
part_cp = cp.asarray(part)


sorted_indices_cp = sort_neighbors_segmented_csr(indptr, indices, deg, part)


# sorted_indices_cp = gpu_sort_neighbors_by_part_and_degree(
#     indptr_cp, indices_cp, deg_cp, part_cp
# )

# If you need it on CPU
sorted_indices_gpu = cp.asnumpy(sorted_indices_cp)
print("sorted Indices GPU: ", sorted_indices_gpu)

sorted_indices_cp = sort_neighbors_segmented_csr_deg(indptr, indices, deg)
sorted_indices_gpu = cp.asnumpy(sorted_indices_cp)
print("sorted Indices GPU: ", sorted_indices_gpu)

sorted_indices_cp = sort_neighbors_by_degree_desc(indptr, indices, deg)
sorted_indices_gpu = cp.asnumpy(sorted_indices_cp)
print("sorted Indices GPU: ", sorted_indices_gpu)


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
        default="ogbn-products",
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

    G = dataset[0]
    print(G)

    # print("metis partition called")
    # No_parts = int(args.fan_out.split(",")[0])
    # if No_parts > :
    # part_array = get_part_array(g, args.parts, args.method, spmm_method, sampling_method)
    # part_array = get_part_array(g, No_parts, args.method, spmm_method, sampling_method, args.dataset)


    start_prep_time = time.time()
    deg = G.in_degrees().numpy()  # Get in-degrees of nodes
    # part_id = dgl.metis_partition_assignment(G, parts, balance_ntypes=None, balance_edges=True, mode='k-way', objtype='cut')
    indptr = np.array(G.adj_tensors('csr')[0]) # Get the indptr of the adjacency matrix 
    indices = np.array(G.adj_tensors('csr')[1])  # Get the indices of the adjacency matrix
    # Apply sorting

    # start_sort_time = time.time()
    # # sorted_indices_cp = sort_neighbors_segmented_csr(indptr, indices, deg, part_id)
    # sorted_indices_cp = sort_neighbors_segmented_csr_deg(indptr, indices, deg)
    # end_sort_time = time.time()
    # print(f"Sorting took {end_sort_time - start_sort_time:.2f} seconds")

    # If you need it on CPU
    # sorted_indices = cp.asnumpy(sorted_indices_cp)
    # print("sorted Indices GPU: ", sorted_indices)

    start_sort_time = time.time()
    # sorted_indices_cp = sort_neighbors_segmented_csr(indptr, indices, deg, part_id)
    sorted_indices_cp = sort_neighbors_by_degree_desc(indptr, indices, deg)
    end_sort_time = time.time()
    print(f"Sorting took {end_sort_time - start_sort_time:.2f} seconds")

    # If you need it on CPU
    sorted_indices = cp.asnumpy(sorted_indices_cp)
    print("sorted Indices GPU: ", sorted_indices)
    



    start_sort_time = time.time()
    sorted_indices_cp = gpu_sort_csr_by_deg(indptr, indices, deg)
    end_sort_time = time.time()
    print(f"Sorting took {end_sort_time - start_sort_time:.2f} seconds")

    # If you need it on CPU
    sorted_indices = cp.asnumpy(sorted_indices_cp)
    print("sorted Indices GPU: ", sorted_indices)


import argparse
import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from tqdm import tqdm
#
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset
import time

# --- Load METIS shared library ---
libmetis = ctypes.cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libmetis.so.5")

# --- Types ---
idx_t = ctypes.c_int32     # change to c_int64 if METIS compiled with 64-bit
real_t = ctypes.c_float    # METIS default real type

# --- METIS function prototypes ---
METIS_SetDefaultOptions = libmetis.METIS_SetDefaultOptions
METIS_SetDefaultOptions.restype = None
METIS_SetDefaultOptions.argtypes = [ctypes.POINTER(idx_t)]

METIS_PartGraphKway = libmetis.METIS_PartGraphKway
METIS_PartGraphKway.restype = ctypes.c_int
METIS_PartGraphKway.argtypes = [
    ctypes.POINTER(idx_t),   # nvtxs
    ctypes.POINTER(idx_t),   # ncon
    ctypes.POINTER(idx_t),   # xadj
    ctypes.POINTER(idx_t),   # adjncy
    ctypes.POINTER(idx_t),   # vwgt
    ctypes.POINTER(idx_t),   # vsize
    ctypes.POINTER(idx_t),   # adjwgt
    ctypes.POINTER(idx_t),   # nparts
    ctypes.POINTER(real_t),  # tpwgts
    ctypes.POINTER(real_t),  # ubvec
    ctypes.POINTER(idx_t),   # options
    ctypes.POINTER(idx_t),   # objval
    ctypes.POINTER(idx_t)    # part
]

# --- METIS Option indices ---
METIS_OPTION_PTYPE   = 1
METIS_OPTION_OBJTYPE = 2
METIS_OPTION_CTYPE   = 3
METIS_OPTION_NCUTS   = 10
METIS_OPTION_SEED    = 11

# --- METIS option values ---
# PTYPE
METIS_PTYPE_RB   = 0
METIS_PTYPE_KWAY = 1

# OBJTYPE
METIS_OBJTYPE_CUT = 0
METIS_OBJTYPE_VOL = 1

# CTYPE
METIS_CTYPE_RM   = 0
METIS_CTYPE_SHEM = 1


# --- Wrapper function ---
def metis_partition(args, xadj, adjncy, nparts, adjwgt=None):
    nvtxs = idx_t(len(xadj) - 1)
    ncon = idx_t(1)

    # Convert numpy arrays to ctypes
    xadj_c = xadj.ctypes.data_as(ctypes.POINTER(idx_t))
    adjncy_c = adjncy.ctypes.data_as(ctypes.POINTER(idx_t))

    vwgt = None
    vsize = None

    # Edge weights
    if adjwgt is not None:
        adjwgt_c = adjwgt.ctypes.data_as(ctypes.POINTER(idx_t))
    else:
        adjwgt_c = None

    nparts_c = idx_t(nparts)

    # Target partition weights (uniform)
    # tpwgts = (real_t * (nparts * 1))()
    # for i in range(nparts):
    #     tpwgts[i] = 1.0 / nparts

    # Imbalance tolerance
    # ubvec = (real_t * 1)(1.05)

    tpwgts = None
    ubvec = None
    obj_cut = True  # True for edge-cut, False for total communication volume

    options = (idx_t * 40)()
    METIS_SetDefaultOptions(options)

    # Map CLI args to METIS constants
    ptype_map = {"rb": METIS_PTYPE_RB, "kway": METIS_PTYPE_KWAY}
    objtype_map = {"cut": METIS_OBJTYPE_CUT, "vol": METIS_OBJTYPE_VOL}
    ctype_map = {"rm": METIS_CTYPE_RM, "shem": METIS_CTYPE_SHEM}

    # options[METIS_OPTION_PTYPE]   = ptype_map[args.ptype]
    options[METIS_OPTION_OBJTYPE] = objtype_map[args.objtype]
    options[METIS_OPTION_CTYPE]   = ctype_map[args.ctype]
    # options[METIS_OPTION_NCUTS]   = 5
    # options[METIS_OPTION_SEED]    = 42
    objval = idx_t()
    part = (idx_t * (len(xadj) - 1))()

    # # --- Call METIS ---
    status = METIS_PartGraphKway(
        ctypes.byref(nvtxs),
        ctypes.byref(ncon),
        xadj_c,
        adjncy_c,
        vwgt,
        vsize,
        adjwgt_c,    # ✅ edge weights passed here
        ctypes.byref(nparts_c),
        tpwgts,
        ubvec,
        options,
        ctypes.byref(objval),
        part
    )

    if status != 1:  # METIS_OK = 1
        raise RuntimeError(f"METIS failed with status {status}")

    return np.frombuffer(part, dtype=np.int32, count=len(xadj)-1), objval.value


# --- Example usage ---
if __name__ == "__main__":
    # Graph: 4-cycle (0-1-2-3-0)
    # xadj = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    # adjncy = np.array([1, 3, 0, 2, 1, 3, 0, 2], dtype=np.int32)

    # # Edge weights: must match adjncy length
    # # Example: make edge (0,1) heavier (10 instead of 2)
    # adjwgt = np.array([
    #     10, 2,   # neighbors of node 0 (to 1,3)
    #     10, 2,   # neighbors of node 1 (to 0,2)
    #     2,  2,   # neighbors of node 2 (to 1,3)
    #     2,  2    # neighbors of node 3 (to 0,2)
    # ], dtype=np.int32)

    # partitions, edgecut = metis_partition(xadj, adjncy, nparts=2, adjwgt=adjwgt)

    # print("Partitions:", partitions)
    # print("Edge cut (weighted):", edgecut)
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
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--spmm", default="cusparse")
    parser.add_argument("--sampling", default="default")
    parser.add_argument("--sampler", default="default")
    parser.add_argument("--nparts", type=int, default=4)
    parser.add_argument("--ptype", choices=["rb", "kway"], default="kway")
    parser.add_argument("--objtype", choices=["cut", "vol"], default="cut")
    parser.add_argument("--ctype", choices=["rm", "shem"], default="shem")
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

    g = dataset[0]
    print(g)
    if args.undirected:
        start = time.time()
        sym_g = dgl.to_bidirected(g)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g 
        print("Convert a graph into a bidirected graph: {:.3f} seconds".format(
            time.time() - start
        ))

    xadj = np.array(g.adj_tensors('csr')[0])
    adjncy = np.array(g.adj_tensors('csr')[1])
    # adjwgt = np.ones(len(adjncy), dtype=np.int32)  # all edges weight 1
    adjwgt = np.full(len(adjncy), 100, dtype=np.int32)  # array of size 1000, all values = 100


    xadj = xadj.astype(np.int32, copy=False)
    adjncy = adjncy.astype(np.int32, copy=False)
    adjwgt = adjwgt.astype(np.int32, copy=False)
    print("xadj:", xadj)
    print("adjncy:", adjncy)
    print("adjwgt:", adjwgt)

     # Example: make some edges heavier
    start = time.time()
    partitions, edgecut = metis_partition(args, xadj, adjncy, nparts=args.nparts, adjwgt=adjwgt)
    print("Partitions:", partitions)
    # print("Edge cut (weighted):", edgecut, "in", args.nparts, "parts")
    print("Number of parts:", args.nparts)
    print(
        "Metis partitioning: {:.3f} seconds".format(
            time.time() - start
        ))


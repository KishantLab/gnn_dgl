import argparse
import time

import dgl
import dgl.function as fn
import dgl.nn as dglnn

import numpy as np
import ogb
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default=".",
        help="Directory to download the OGB dataset.",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="./graph.dgl",
        help="Path to the graph.",
    )
    parser.add_argument(
        "--full-feature-path",
        type=str,
        default="./full.npy",
        help="Path to the features of all nodes.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model.pt",
        help="Path to store the best model.",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="./results",
        help="Submission directory.",
    )
    args = parser.parse_args()

    dataset = MAG240MDataset(root=args.rootdir)

    print("Loading graph")
    (g,), _ = dgl.load_graphs(args.graph_path)
    g = g.formats(["csc"])
    print(g)

    print("Loading features")
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    feats = np.memmap(
        args.full_feature_path,
        mode="r",
        dtype="float16",
        shape=(num_nodes, num_features),
    )
    print("Features shape:", feats.shape)
    print(feats)
    print("feats dtype:", feats.dtype)

    features = feats.copy()  # make writable if it's read-only
    print("Converting features to torch tensor")
    torch_feats = torch.from_numpy(features)
    torch.save(torch_feats, 'full_feats_tensor.pt') 
    print("Features saved to full_feats_tensor.pt")
    g.ndata["feat"] = torch_feats
    print(g)
    # del feats
    print("feats deleted from memory")
    dgl.save_graphs("graph_with_features.dgl", [g])
    print("Features loaded and saved to graph_with_features.dgl")
    # g.ndata["feat"] = torch.from_numpy(feats).to(torch.float32)

    print("Loading labels")
    labels = dataset.paper_label
    print("Labels shape:", labels.shape)


    g.ndata["label"] = torch.from_numpy(labels).to(torch.int64)
    dgl.save_graphs("graph_with_labels.dgl", [g])
    print("Labels loaded and saved to graph_with_labels.dgl")

    print(g)

    # print("Loading masks and labels")
    # train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    # valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
 


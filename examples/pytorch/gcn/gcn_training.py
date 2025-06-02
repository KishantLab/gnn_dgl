import argparse

import time
import dgl
import dgl.nn as dglnn

from dgl.metis_sampling import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask].to(torch.bool)
        labels = labels[mask].to(torch.bool)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1].to(torch.bool)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    total_training_time = 0.0
    total_model_time = 0.0
    total_loss_opt_time = 0.0
    epoch_lines = []
    # training loop
    for epoch in range(int(args.epoch)):
        start_time1 = time.time()
        model_exe_time = 0.0
        start_model_time = time.time()
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_model_time = time.time()
        end_time1 = time.time()
        acc = evaluate(g, features, labels, val_mask, model)
        model_exe_time += end_model_time - start_model_time
        total_training_time += model_exe_time
        # print(
        #         "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Model Time {:  .4f} ".format(
        #         epoch, loss.item(), acc, model_exe_time
        #     )
        # )
        
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Model Time {:.4f} ".format(
                epoch, loss.item(), acc, model_exe_time
            )

        epoch_lines.append(epoch_line)
    total_tt = "Total_Training_Time {:.4f}".format(total_training_time)
    epoch_lines.append(total_tt)
    return epoch_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
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

    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--spmm", default="cusparse")
    parser.add_argument("--sampling", default="default")
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    # if args.dataset == "cora":
    #     data = CoraGraphDataset(transform=transform)
    # elif args.dataset == "citeseer":
    #     data = CiteseerGraphDataset(transform=transform)
    # elif args.dataset == "pubmed":
    #     data = PubmedGraphDataset(transform=transform)
    # else:
    #     raise ValueError("Unknown dataset: {}".format(args.dataset))
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
    g = dgl.add_self_loop(g)
    data = dataset
    print("metis partition called")
    print(spmm_method)
    # part_array = get_part_array(g, args.parts, spmm_method)
    part_array = get_part_array(g, args.parts, args.method, spmm_method, sampling_method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = dataset.num_classes
    model = GCN(in_size, 16, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    execution_time1 = 0.0
    start_time1 = time.time()
    epoch_lines = train(g, features, labels, masks, model)
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1

    # train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
    Accuracy = "Test Accuracy {:.4f}".format(acc)
    epoch_lines.append(Accuracy)
    with open('epoch_data.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')



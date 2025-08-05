import argparse
import numpy as np
import dgl
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.metis_sampling import *
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    #train_idx = dataset.train_idx.to(device)
    #val_idx = dataset.val_idx.to(device)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze().to(device)
    print("train_idxi: ", train_idx.len(), "val_idx: ", val_idx.len(), "test_idx: ", test_idx.len())
    execution_time = 0.0
    start_time = time.time()
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],
        #[30, 30, 30],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    end_time = time.time()
    execution_time = end_time - start_time
    # G = dgl.to_homogeneous(g)
    # print(G, type(G))
    # print("total sampling time:", execution_time, "seconds")

    # _computed_array = dgl.metis_partition_assignment(G, 4, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
    # dgl.distributed.partition_graph(G, 'test', 2, out_path='output/')

    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size= int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        # part_array,
        device=device,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    total_training_time = 0.0
    epoch_lines = []
    for epoch in range(int(args.epoch)):
        model.train()    
        total_loss = 0
        execution_time = 0.0
        start_time1 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        # print("training time:", execution_time, "seconds")
        acc = evaluate(model, g, val_dataloader, num_classes)
        # print(
        #     "\nEpoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(
        #          epoch, total_loss / (it + 1), acc.item(), execution_time
        #      )
        #  )
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(
                    epoch, total_loss / (it + 1), acc.item(), execution_time
        )
        epoch_lines.append(epoch_line)
    tt_time = "Total Training time {:.4f}".format( total_training_time)
    epoch_lines.append(tt_time)
    return epoch_lines

    # print("\nTotal Training Time {:.4f} seconds".format(total_training_time))
        

if __name__ == "__main__":
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
        choices=["1024", "2048", "4096", "8192"],
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

    parser.add_argument("--fan_out", type=str, default="20,20,20")
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--spmm", default="respmm")
    parser.add_argument("--sampling", default="metis")
    parser.add_argument("--sampler", default="default")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"\nTraining in {args.mode} mode.")

    # load and preprocess dataset
    # print("\nLoading data")
    # dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    # load and preprocess dataset
    # if args.dataset == "cora":
    #     dataset = CoraGraphDataset()
    # elif args.dataset == "citeseer":
    #     dataset = CiteseerGraphDataset()
    # elif args.dataset == "pubmed":
    #     dataset = PubmedGraphDataset()
    # elif args.dataset == "wisconsin":
    #     dataset = WisconsinDataset()
    # elif args.dataset == "flickr":
    #     dataset = FlickrDataset()
    # elif args.dataset == "reddit":
    #     dataset = RedditDataset()
    # elif args.dataset == "ogbn-products":
    #     dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    # elif args.dataset == "ogbn-arxiv":
    #     dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    # elif args.dataset == "yelp":
    #     dataset = YelpDataset()
    # else:
    #     dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
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
 

    g = dataset[0]
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    # g = g.to("cuda")
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

    print(type(g))
    print(g)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze()
    val_idx = torch.nonzero(val_mask).squeeze()
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze().to(device)
    print("train_idxi: ", train_idx.shape, "val_idx: ", val_idx.shape, "test_idx: ", test_idx.shape)

    #---------find cosin similarity of graph--------------
    # Extract node features for all edges
    # src, dst = g.edges()
    # src_features = g.ndata['feat'][src]
    # dst_features = g.ndata['feat'][dst]

    # Compute cosine similarity for all edges
    # cosine_similarities = F.cosine_similarity(src_features, dst_features)
    # print(type(cosine_similarities))
    # print(cosine_similarities)
    # Print the results
    # for (s, d), sim in zip(zip(src.tolist(), dst.tolist()), cosine_similarities.tolist()):
    #     print(f"Edge ({s}, {d}) - Cosine Similarity: {sim}")
    #
    out_degrees = np.array(g.out_degrees())
    in_degrees = np.array(g.in_degrees())
    if np.sum(out_degrees) == np.sum(in_degrees):
        print("graph is undirected")
    else:
        print("graph is direct")
    deg = out_degrees
    zero_deg_mask = (deg == 0)
    num_removed = zero_deg_mask.sum().item()
    print(f"Removed {num_removed} zero-degree nodes.")

    # out_degrees = np.array(g.out_degrees())
    max_value = np.max(out_degrees)
    avg_value = np.mean(out_degrees)
    print("maximum degree : ",max_value)
    print("Average degree : ",avg_value)
    # Count the number of nodes with in-degree less than 100
    num_nodes_less_than_100 = len(out_degrees[out_degrees < 100])
    num_nodes_less_than_128 = len(out_degrees[out_degrees < 128])
    num_nodes_less_than_1024 = len(out_degrees[out_degrees < 1024])
    num_nodes_less_than_1024_1 = len(out_degrees[out_degrees >= 1024])
    print("Total number of nodes with in-degree less than 100:", num_nodes_less_than_100)
    print("Total number of nodes with in-degree less than 1024:", num_nodes_less_than_1024)
    print("Total number of nodes with in-degree greter than 1024:", num_nodes_less_than_1024_1)
    print("Total number of nodes with in-degree less than 128:", num_nodes_less_than_128)
    unique_values, frequencies = np.unique(out_degrees, return_counts=True)
    #
    # # Create a TSV file
    # output_file = str(args.dataset) + ".tsv"
    # # Write unique values and frequencies to the TSV file
    # np.savetxt(output_file, np.column_stack((unique_values, frequencies)), delimiter='\t', fmt='%d')
    #
    plt.bar(unique_values, frequencies)
    plt.xlabel('Degree of Vertex', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.ylim(0, 200000)
    max_y = max(frequencies)
    highest_y = np.max(frequencies)
    highest_x = unique_values[np.argmax(frequencies)]
    plt.annotate(str(highest_y), xy=(highest_x, highest_y), ha='center', va='bottom', fontsize=18)
    #
    # # Add text annotation for the highest value
    # plt.text(unique_values[frequencies.index(max_y)], max_y, str(max_y), ha='center', va='bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.ylim(1, 10**7)
    #plt.title('Degree Distribution')
    plot_name = str(args.dataset) + ".pdf"
    plt.savefig(plot_name, format='pdf')

    fanout_part = int(args.fan_out.split(",")[0])
    No_parts = int(avg_value)
    if No_parts < fanout_part:
        No_parts = fanout_part

    No_parts = int(args.fan_out.split(",")[0])
    part_array = get_part_array(g, No_parts, args.method, spmm_method, sampling_method, args.dataset)
    part_id = get_part_id()

    num_parts = int(part_id.max().item()) + 1  # total partitions
    counts = torch.bincount(part_id, minlength=num_parts)

    plt.figure(figsize=(10, 5))
    plt.bar(torch.arange(num_parts).tolist(), counts.tolist())
    plt.xlabel("Partition ID")
    plt.ylabel("Number of Nodes")
    plt.title("Node Count per Partition (METIS)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    plot_name = str(args.dataset) + "_partition_distribution" + str(num_parts) + ".pdf"
    plt.savefig(plot_name, format='pdf')


    # g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()

    # num_classes = dataset.num_classes

    # create GraphSAGE model)
    # in_size = g.ndata["feat"].shape[1]
    # out_size = dataset.num_classes
    # model = SAGE(in_size, 256, out_size).to(device)
    #
    # # convert model and graph to bfloat16 if needed
    # if args.dt == "bfloat16":
    #     g = dgl.to_bfloat16(g)
    #     model = model.to(dtype=torch.bfloat16)

    # out partion create array 
    #part_array = np.ones(5)
    # part_array = get_part_array(g, args.parts)

    # part_array = torch.from_numpy(part_array)
    # model training
    # print("\nTraining...")
    execution_time1 = 0.0
    start_time1 = time.time()
    #train(args, device, g, dataset, model, num_classes)
    # epoch_lines=train(args, device, g, dataset, model, num_classes)
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    # print("total training time:", execution_time1, "seconds")


    # test the model
    # print("\nTesting...")
    # acc = layerwise_infer(
    #     device, g, test_idx, model, num_classes, batch_size=4096
    # )
    # #print("\nTest Accuracy {:.4f}".format(acc.item()))
    # Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    # epoch_lines.append(Accuracy)
    # with open('epoch_data.txt', 'w') as file:
    #     for value in epoch_lines:
    #         file.write(str(value) + '\n')
    #



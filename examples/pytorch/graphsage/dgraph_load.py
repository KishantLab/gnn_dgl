import dgl
import numpy as np
import torch

# Load the data from the npz file
data = np.load('DGraphFin/dgraphfin.npz')

# Extract arrays from the file
x = torch.tensor(data['x'], dtype=torch.float32)        # Node features
y = torch.tensor(data['y'], dtype=torch.long)            # Node labels
edge_index = torch.tensor(data['edge_index'], dtype=torch.long).t()  # Edge list (transpose to make it (2, num_edges))
edge_type = torch.tensor(data['edge_type'], dtype=torch.long)        # Edge types
edge_timestamp = torch.tensor(data['edge_timestamp'], dtype=torch.float32)  # Edge timestamps

# Get the number of nodes
num_nodes = x.shape[0]

# Ensure masks have the same number of nodes by padding with False
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:data['train_mask'].shape[0]] = torch.tensor(data['train_mask'], dtype=torch.bool)

valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
valid_mask[:data['valid_mask'].shape[0]] = torch.tensor(data['valid_mask'], dtype=torch.bool)

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[:data['test_mask'].shape[0]] = torch.tensor(data['test_mask'], dtype=torch.bool)

# Create a DGL graph
graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

# Add node features and labels to the graph
graph.ndata['feat'] = x        # Node features
graph.ndata['label'] = y       # Node labels

# Add train/val/test masks
graph.ndata['train_mask'] = train_mask
graph.ndata['valid_mask'] = valid_mask
graph.ndata['test_mask'] = test_mask

# Add edge data (optional)
graph.edata['edge_type'] = edge_type         # Edge types
graph.edata['timestamp'] = edge_timestamp    # Edge timestamps

# Now you have the graph in DGL format with all features, labels, and masks.
print(graph)
print(graph.ndata['feat'])
print(graph.ndata['label'])


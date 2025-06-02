import argparse, dgl
from dataloader import IGB260MDGLDataset

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M/', 
    help='path containing the datasets')
parser.add_argument('--dataset_size', type=str, default='tiny',
    choices=['tiny', 'small', 'medium', 'large', 'full'], 
    help='size of the datasets')
parser.add_argument('--num_classes', type=int, default=19, 
    choices=[19, 2983], help='number of classes')
parser.add_argument('--in_memory', type=int, default=0, 
    choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
parser.add_argument('--synthetic', type=int, default=0,
    choices=[0, 1], help='0:nlp-node embeddings, 1:random')
args = parser.parse_args()

dataset = IGB260MDGLDataset(args)
graph = dataset[0]
print(graph)
# Graph(num_nodes=100000, num_edges=547416,
#      ndata_schemes={'feat': Scheme(shape=(1024,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64), 'train_mask': Scheme(shape=(), #dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool)}
#      edata_schemes={})

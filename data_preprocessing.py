import os
from typing import List, Tuple
import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data, Dataset
import random

# read the adjacency list
def read_edge_list(path: str) -> List[Tuple[int,int]]:
    """Read adjacency file with lines 'row,col' or 'row col' (1-based indices)."""
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # handle comma or space separated
            if ',' in line:
                a,b = line.split(',')
            else:
                a,b = line.split()
            edges.append((int(a), int(b)))
    return edges

# read a vector of integers from file
def read_vector_int(path: str) -> List[int]:
    with open(path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

# read node attributes (float values) from file
def read_node_attributes(path: str) -> List[List[float]]:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            # comma separated floats
            parts = [p.strip() for p in line.split(',') if p.strip()]
            rows.append([float(x) for x in parts])
    return rows

# build list of Data objects from raw files
def build_graphs_from_raw(data_dir: str):
    """
    Builds list of torch_geometric.data.Data objects from ENZYMES raw files in data_dir.
    Expects files:
     - ENZYMES_A.txt         (edges)
     - ENZYMES_graph_indicator.txt
     - ENZYMES_graph_labels.txt
     - ENZYMES_node_labels.txt   (optional)
     - ENZYMES_node_attributes.txt (optional)
    Node/graph indexing in the files is 1-based.
    """
    # file paths
    edges_file = os.path.join(data_dir, 'ENZYMES_A.txt')
    graph_indicator_file = os.path.join(data_dir, 'ENZYMES_graph_indicator.txt')
    graph_labels_file = os.path.join(data_dir, 'ENZYMES_graph_labels.txt')
    node_attr_file = os.path.join(data_dir, 'ENZYMES_node_attributes.txt')
    node_label_file = os.path.join(data_dir, 'ENZYMES_node_labels.txt')

    edges = read_edge_list(edges_file)
    graph_indicator = read_vector_int(graph_indicator_file)  # length = n nodes
    graph_labels = read_vector_int(graph_labels_file)  # length = N graphs
    node_attrs = None
    node_labels = None
    if os.path.exists(node_attr_file):
        node_attrs = read_node_attributes(node_attr_file)
    if os.path.exists(node_label_file):
        node_labels = read_vector_int(node_label_file)

    # total nodes
    n = len(graph_indicator)
    # Build adjacency per graph: ENZYMES_A is block-diagonal across graphs.
    # edges use 1-based node ids
    # Create mapping graph_id -> list of node ids (1-based)
    graph_nodes = {}
    for node_idx, g in enumerate(graph_indicator, start=1):
        graph_nodes.setdefault(g, []).append(node_idx)

    # Build Data objects
    data_list = []
    N = len(graph_labels)
    for g_id in range(1, N+1):
        nodes = graph_nodes.get(g_id, [])
        if len(nodes) == 0:
            continue
        # map global node id -> local idx (0-based)
        local_idx = {global_id: i for i, global_id in enumerate(nodes)}
        # collect edges for this graph
        edge_index = [[], []]
        for (u, v) in edges:
            # if both endpoints in this graph
            if u in local_idx and v in local_idx:
                edge_index[0].append(local_idx[u])
                edge_index[1].append(local_idx[v])
                # assume undirected; add reverse if not present
                edge_index[0].append(local_idx[v])
                edge_index[1].append(local_idx[u])

        if len(edge_index[0]) == 0:
            # empty graph: create no edges
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        # node features
        if node_attrs is not None:
            x = np.array([node_attrs[idx-1] for idx in nodes], dtype=np.float32)  # idx-1 for 0-based list
            x = torch.tensor(x, dtype=torch.float)
        elif node_labels is not None:
            # use one-hot encoding of node labels
            labels = [node_labels[idx-1] for idx in nodes]
            max_label = max(node_labels)
            x = label_binarize(labels, classes=list(range(1, max_label+1))).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float)
        else:
            # fallback: identity or degree feature
            x = torch.ones((len(nodes), 1), dtype=torch.float)

        y = torch.tensor([graph_labels[g_id-1]-1], dtype=torch.long)  # make 0-based class labels

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

# Define the Dataset class (Pytorch Geometric Style)
class EnzymesDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = build_graphs_from_raw(root)  # directly use raw folder

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    #for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    #create dataset
    dataset = EnzymesDataset(root='data')
    print(f"Total graphs: {len(dataset)}")

    # infer num classes and in_channels
    sample = dataset[0]
    in_channels = sample.x.size(1)
    labels = [int(d.y.item()) for d in dataset]
    num_classes = int(max(labels)+1)
    print(f"in_channels: {in_channels}, num_classes: {num_classes}")
    

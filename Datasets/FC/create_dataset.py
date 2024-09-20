import os
import scipy.io as scio
from scipy.sparse import coo_matrix

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def read_dataset():
    """
    read data from Dataset/ASD.mat' and reconstruct data as 'torch_geometric.data.Data'
    """
    
    data = scio.loadmat('Datasets/FC/ASD.mat')   # Data is available at google drive (https://drive.google.com/drive/folders/1EkvBOoXF0MB2Kva9l4GQbuWX25Yp81a8?usp=sharing).
    dataset = []
    for graph_index in range(len(data['label'])):
        label = data['label']

        graph_struct = data['graph_struct'][0]

        edge = torch.Tensor(graph_struct[graph_index]['edge'])

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])

        node_tags = torch.Tensor(graph_struct[graph_index]['node_tags'])
        adj = torch.Tensor(graph_struct[graph_index]['adj'])
        neighbor = graph_struct[graph_index]['neighbor']
        y = torch.Tensor(label[graph_index])
        A = torch.sparse_coo_tensor(
            indices = edge[:, :2].t().long(),
            values = edge[:, -1].reshape(-1,).float(),
            size = (116, 116)
            )
        G = (A.t() + A).coalesce()

        graph = Data(x=ROI.reshape(-1,116).float(),
                     edge_index=G.indices().reshape(2,-1).long(),
                     edge_attr=G.values().reshape(-1,1).float(),
                     y=y.long())
        dataset.append(graph)
    return dataset

def read_dataset_MDD():
    """
    read data from 'Datasets/FC/MDD.mat' and reconstruct data as 'torch_geometric.data.Data'
    """
    
    data = scio.loadmat('Datasets/FC/MDD.mat')
    dataset = []
    # Adjust the loop to iterate over the second dimension of the `label` array
    for graph_index in range(data['label'].shape[1]):
        label = data['label'][0]  # Access the first row containing all labels

        graph_struct = data['graph_struct'][0]

        edge = torch.Tensor(graph_struct[graph_index]['edge'])

        ROI = torch.Tensor(graph_struct[graph_index]['ROI'])

        node_tags = torch.Tensor(graph_struct[graph_index]['node_tags'])
        adj = torch.Tensor(graph_struct[graph_index]['adj'])
        neighbor = graph_struct[graph_index]['neighbor']
        y = torch.Tensor([label[graph_index]])  # Ensure `y` is correctly shaped as a tensor
        A = torch.sparse_coo_tensor(
            indices=edge[:, :2].t().long(),
            values=edge[:, -1].reshape(-1,).float(),
            size=(116, 116)
        )
        G = (A.t() + A).coalesce()

        graph = Data(x=ROI.reshape(-1, 116).float(),
                     edge_index=G.indices().reshape(2, -1).long(),
                     edge_attr=G.values().reshape(-1, 1).float(),
                     y=y.long())
        dataset.append(graph)
    return dataset

if __name__ == '__main__':
    dataset = read_dataset()
    print(len(dataset))
    loader = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print(loader.y)
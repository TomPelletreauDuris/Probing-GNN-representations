from typing import Optional, Callable

import torch
from networkx.generators import random_graphs, lattice, small, classic
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import barabasi_albert_graph
import networkx as nx
import pickle as pkl
import random
import numpy as np
import torch_geometric.transforms as T
from networkx.algorithms.operators.binary import compose, union
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from networkx.algorithms import smallworld, community


class BA_houses_color(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-houses_color.pkl','rb') as fin:
            (adjs, feas, labels,_) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index
            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea,dtype=torch.float), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)



class ER_nb_stars(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/ER-nb_stars.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)


class ER_nb_stars2(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/ER-nb_stars2.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)





class BA_2grid_house(InMemoryDataset):

    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid-house.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            
            
            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)


class BA_2grid_house_with_node_degree_as_features(InMemoryDataset):
    
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid-house.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            node_degree = torch.tensor([g.degree(node) for node in g.nodes()])
            fea = torch.cat((fea.unsqueeze(1),node_degree.unsqueeze(1)),dim=1)
            
            data = Data(x=fea, edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

class BA_2grid_house_with_node_degree_as_features_and_expand_10_dimensions(InMemoryDataset):
    def __init__(self, diffpool=False, max_node=None, transform=None, pre_filter=None):
        super().__init__('.', transform, pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid-house.pkl', 'rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            # Use node degrees as features and expand to 10 dimensions
            fea = np.array(list(dict(g.degree()).values())).reshape(-1, 1)  # Node degrees
            fea = np.repeat(fea, 10, axis=1)  # Repeat to create 10-dimensional features

            label = labels[i]

            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1], dtype=torch.float)

            data = Data(x=torch.tensor(fea, dtype=torch.float), edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), expl_mask=expl_mask, edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)



class BA_2grid(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

class BA_2grid_to_test(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2grid.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)
        data_list = []
        for i in range(len(adjs)):
            num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()
            density = density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            # average_shortest_path_length = nx.average_shortest_path_length(g)
            # transitivity = nx.transitivity(g)
            # average_clustering = nx.average_clustering(g)
            # clustering_coefficient = nx.clustering(g)
            # small_world = nx.smallworld.sigma(g)
            # modularity = nx.community.modularity(g)
            # assortativity_index = nx.degree_assortativity_coefficient(g)

            y = torch.tensor([num_nodes, num_edges, density])
            # y = torch.tensor([num_nodes, num_edges, density, average_shortest_path_length, transitivity, average_clustering,clustering_coefficient,small_world,modularity,assortativity_index])

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index,y=y,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

class BA_2motfs(InMemoryDataset):

    def __init__(self,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-2motif.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        data_list = []
        for i in range(len(adjs)):
            num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            
            
            data = Data(x=torch.tensor(fea), edge_index=edge_index, y=label,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)





class BA_multipleShapes2(InMemoryDataset):

    def __init__(self,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)

        with open('Datasets/BA-multipleShapes2.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        num_nodes = len(adjs[0][0])
        self.num_nodes = num_nodes
        data_list = []
        for i in range(len(adjs)):
            adj = adjs[i]

            if labels[i]  == 0.0:
                label = 0
            else:
                label = 1
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)
            data = from_networkx(nx.from_numpy_matrix(adj)) #to make edge_index undirected
            data = Data(x=torch.tensor(feas[i]), edge_index=data.edge_index, y=label,expl_mask=expl_mask)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

class ProbingDataset(InMemoryDataset):
    def __init__(self,diffpool=False,max_node=None,transform: Optional[Callable] = None,pre_filter: Optional[Callable] = None):
        super().__init__('.',transform,pre_filter)
        random.seed(10)
        
        data_list = []
        
        with open('Datasets/BA-2grid-house.pkl','rb') as fin:
            (adjs, feas, labels) = pkl.load(fin)

        for i in range(len(adjs)):
            if diffpool:
                num_nodes = max_node
            else:
                num_nodes = len(adjs[i][0])
            adj = adjs[i]
            g = nx.from_numpy_array(adj)
            tmp_data = from_networkx(g)

            fea = feas[i]
            label = labels[i]
            
            if random.random() <= 0.5:
                expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
            else:
                expl_mask = torch.ones(num_nodes, dtype=torch.bool)

            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()
            density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            average_shortest_path_length = nx.average_shortest_path_length(g)
            cliques = list(nx.find_cliques(g))
            num_cliques = len(cliques)
            # transitivity = nx.transitivity(g)
            # average_clustering = nx.average_clustering(g)
            # small_world = nx.smallworld.sigma(g)
            # assortativity_index = nx.degree_assortativity_coefficient(g)

            # Community detection for modularity calculation
            # communities = list(community.greedy_modularity_communities(g))
            # modularity = community.modularity(g, communities)

            # # Summary statistic for clustering coefficients
            # clustering_coefficients = nx.clustering(g)
            # avg_clustering_coefficient = sum(clustering_coefficients.values()) / len(clustering_coefficients)


            y = torch.tensor([label, num_nodes, num_edges, density, average_shortest_path_length, num_cliques])
            # y = torch.tensor([
            #     num_nodes, num_edges, density, average_shortest_path_length, 
            #     transitivity, average_clustering, small_world, assortativity_index
            # ], dtype=torch.float)

            # y = torch.tensor([
            #     num_nodes, num_edges, density, average_shortest_path_length, 
            #     transitivity, average_clustering, avg_clustering_coefficient, 
            #     small_world, modularity, assortativity_index
            # ], dtype=torch.float)

            edge_index = tmp_data.edge_index

            edge_weights = torch.zeros(edge_index.shape[-1],dtype=torch.float)

            data = Data(x=torch.tensor(fea), edge_index=edge_index,y=y,expl_mask=expl_mask,edge_attr=edge_weights)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
    
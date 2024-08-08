from Datasets.FC.create_dataset import read_dataset
import torch
import pickle as pkl
import networkx as nx
from torch_geometric.data import Data

dataset = read_dataset()

print(len(dataset))

torch.manual_seed(37)

DATASET = "FC"

MODEL = "GIN"
from models.models_FC import GIN_framework as framework # import the model
gnn = framework(dataset)

MODEL3 = "GIN3"
from models.models_FC import GIN_framework3 as framework3 # import the model
gnn3 = framework3(dataset)

gnn.load_model(path="models/"+DATASET+"_"+MODEL+"server.pt", map_location=torch.device('cpu'))

gnn3.load_model(path="models/"+DATASET+"_"+MODEL3+"server.pt", map_location=torch.device('cpu'))

MODEL = MODEL3

print('we are using model: ', MODEL)

def calculate_avg_path_length(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        largest_component = max(components, key=len)
        return nx.average_shortest_path_length(largest_component)

def compute_graph_properties(data):
    properties = []
    for graph_data in data:
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges
        density = float(num_edges) / (num_nodes * (num_nodes - 1) / 2)
        avg_path_len = calculate_avg_path_length(graph_data.edge_index)
        num_cliques = len(list(nx.find_cliques(nx.Graph(graph_data.edge_index.t().tolist()))))
        num_triangles = sum(nx.triangles(nx.Graph(graph_data.edge_index.t().tolist())).values()) / 3
        num_squares = sum(nx.square_clustering(nx.Graph(graph_data.edge_index.t().tolist())).values()) / 4
        num_components = len(list(nx.connected_components(nx.Graph(graph_data.edge_index.t().tolist()))))
        largest_component_size = len(max(nx.connected_components(nx.Graph(graph_data.edge_index.t().tolist())), key=len))
        if nx.is_connected(nx.Graph(graph_data.edge_index.t().tolist())):
            small_world = nx.algorithms.smallworld.sigma(nx.Graph(graph_data.edge_index.t().tolist()))
        else:
            components = [nx.Graph(graph_data.edge_index.t().tolist()).subgraph(c).copy() for c in nx.connected_components(nx.Graph(graph_data.edge_index.t().tolist()))]
            largest_component = max(components, key=len)
            small_world = nx.algorithms.smallworld.sigma(largest_component)

        properties.append((num_nodes, num_edges, density, avg_path_len, num_cliques, num_triangles, num_squares, num_components, largest_component_size, small_world))
    return properties

train_idx_list = gnn.train_idx.tolist()
selected_dataset = [gnn.dataset[i] for i in train_idx_list]
train_properties = compute_graph_properties(selected_dataset[0:1])

test_idx_list = gnn.test_idx.tolist()
selected_dataset = [gnn.dataset[i] for i in test_idx_list]
test_properties = compute_graph_properties(selected_dataset[0:1])

# Move the data to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_properties = torch.tensor(train_properties).to(device)
test_properties = torch.tensor(test_properties).to(device)

# Save the properties to files
with open("results/"+DATASET+"_"+MODEL+"_train_properties_with_sm.pkl", "wb") as f:
    pkl.dump(train_properties.cpu().tolist(), f)

with open("results/"+DATASET+"_"+MODEL+"_test_properties_with_sm.pkl", "wb") as f:
    pkl.dump(test_properties.cpu().tolist(), f)

print(train_properties[0:5])
print('finished')
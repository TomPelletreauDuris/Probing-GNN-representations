# %% [markdown]
# # Node probing for AIFB dataset and GCN

# %% [markdown]
# Here we'll first be loading the dataset and explore its structure
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("rdflib")

# %%
#dataset MUTAG
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Entities
from torch_geometric.transforms import TargetIndegree

# Load the AIFB dataset
dataset = Entities(root='/tmp/AIFB', name='AIFB')
dataset[0]

# %%
# import rdflib
# from collections import defaultdict

# # Load the AIFB dataset
# graph = rdflib.Graph()
# graph.parse("aifb_dataset.ttl", format="ttl")

# # Dictionary to store node labels
# node_labels = defaultdict(list)

# # Iterate through the triples and extract rdf:type triples
# for subj, pred, obj in graph:
#     if pred == rdflib.RDF.type:
#         node_labels[subj].append(obj)

# # Remove rdf:type triples from the graph
# graph.remove((None, rdflib.RDF.type, None))

# # Print the labels for each node
# for node, labels in node_labels.items():
#     print(f"Node: {node}, Labels: {labels}")


# %%
from torch_geometric.datasets import TUDataset
#utag
mutag = TUDataset(root='/tmp/MUTAG', name='MUTAG')
mutag
mutag[0]

# %%
data = dataset[0]  # AIFB is a single graph

# Create one-hot encoded node features for the entire dataset
num_nodes = data.num_nodes
data.x = torch.eye(num_nodes)  # One-hot encoding for all nodes


# %%
dataset.data = data  # This line ensures the dataset now references the modified data


# %%
dataset.data

# %% [markdown]
# Model

# %%
#set the seed
torch.manual_seed(37)

MODEL = "GCN"
DATASET = "AIFB"

from models.models_AIFB import GCN_framework as framework # import the model

gnn = framework(dataset)


# %%
# gnn.iterate()

# %%
# gnn.train()

# %%
#save the model 
# gnn.save_model(path="models/"+DATASET+"_"+MODEL+".pt")

# %%
#load the model
gnn.load_model(path="models/"+DATASET+"_"+MODEL+".pt")

# %%
gnn.evaluate()

# %% [markdown]
# ### Features / embeddings

# %% [markdown]
# ```plaintext
# [
#     (
#         Feature shape: torch.Size([8285, 128]), x1,
#         Feature shape: torch.Size([8285, 128]), x2,
#         Feature shape: torch.Size([8285, 128]), x3,
#         Feature shape: torch.Size([8285, 128]), x4,
#         Feature shape: torch.Size([8285, 4]), x5
#     ),
# ]

# %%
train_features, test_features = gnn.evaluate_with_features2()

# %%
print(len(train_features[0]))
print(len(test_features[0]))
len(train_features), len(test_features)

# %%
train_features[0][0].shape

# %%
#print the possibles classes that we're classifying in train_y of dataset 
print(set(dataset.data.train_y.numpy()))

# %% [markdown]
# ### node properties

# %%
#we have the train_idx=[140], test_idx=[36]

#we want to calculate the node properties of the train and test nodes such as node degree, clustering coefficient, betweenness centrality, closeness centrality, eigenvector centrality, and pagerank. And get the train_node_properties and test_node_properties

#we will use the networkx library to calculate the node properties
# Convert PyTorch Geometric Data to NetworkX format
import networkx as nx
from torch_geometric.utils import to_networkx
import torch

G = to_networkx(data, to_undirected=True)

# Calculate various node properties using NetworkX
degree = dict(G.degree())
clustering = nx.clustering(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
pagerank = nx.pagerank(G)


# %%

# Create a function to gather properties for specific nodes
def get_node_properties(indices):
    properties = []
    for idx in indices:
        properties.append({
            'degree': degree[idx],
            'clustering': clustering[idx],
            'betweenness': betweenness[idx],
            'closeness': closeness[idx],
            'eigenvector': eigenvector[idx],
            'pagerank': pagerank[idx]
        })
    return properties

# Extract node properties for train and test indices
train_node_properties = get_node_properties(data.train_idx.tolist())
test_node_properties = get_node_properties(data.test_idx.tolist())

# Convert to tensor format if needed
train_node_properties_tensor = {key: torch.tensor([prop[key] for prop in train_node_properties]) for key in train_node_properties[0]}
test_node_properties_tensor = {key: torch.tensor([prop[key] for prop in test_node_properties]) for key in test_node_properties[0]}



# %% [markdown]
# ## Diagnostic classifier 


# %% [markdown]
# ## Probe for different properties with a one classifier per property across all nodes
# 
# The code aggregates all node embeddings within a layer into a single feature matrix. It then trains a single linear model (probe) to predict the corresponding property values (e.g., "degree") across all nodes in that layer. This means the probe is treating the layer as a whole and learning to map the entire layer's embedding space to the property values collectively. This approach is more about understanding how the entire layer's representation relates to the properties collectively.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

def train_probe_for_layer(features, property_values, test_features, test_property_values, num_epochs=1000000, learning_rate=0.01):
    # Convert features and property values to PyTorch tensors if they are NumPy arrays
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    if isinstance(property_values, np.ndarray):
        property_values = torch.tensor(property_values, dtype=torch.float32)

    if isinstance(test_features, np.ndarray):
        test_features = torch.tensor(test_features, dtype=torch.float32)
    if isinstance(test_property_values, np.ndarray):
        test_property_values = torch.tensor(test_property_values, dtype=torch.float32)

    print(f"Training on features with shape: {features.shape} for property values shape: {property_values.shape}")

    model = LinearModel(features.shape[1])  # Features should be 2D: (num_nodes, feature_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features).squeeze()  # Remove single-dimensional entries
        loss = criterion(output, property_values)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        pred = model(test_features).squeeze()
        mse = criterion(pred, test_property_values).item()
        # Flatten the tensors for proper use of r2_score
        r2 = r2_score(test_property_values.cpu().numpy(), pred.cpu().numpy())

    return mse, r2

def evaluate_layer_probes(train_features, test_features, train_properties, test_properties):
    num_layers = len(train_features)
    results = []

    for layer_idx in range(num_layers):
        layer_train_features = train_features[layer_idx]
        layer_test_features = test_features[layer_idx]

        # Stack node features into a single matrix for each layer
        train_feature_matrix = np.vstack(layer_train_features)  # Shape: (num_nodes, feature_size)
        test_feature_matrix = np.vstack(layer_test_features)

        # Train and evaluate probe for each property
        for property_name in train_properties[0].keys():  # Assuming all nodes have the same properties
            train_property_values = np.array([node[property_name] for node in train_properties])
            test_property_values = np.array([node[property_name] for node in test_properties])

            mse, r2 = train_probe_for_layer(train_feature_matrix, train_property_values, test_feature_matrix, test_property_values)
            
            results.append({
                'layer': layer_idx,
                'property': property_name,
                'mse': mse,
                'r2': r2
            })

    return results

def plot_results(results):
    layers = sorted(set(result['layer'] for result in results))
    properties = sorted(set(result['property'] for result in results))
    
    plt.figure(figsize=(12, 6))  # Increase the figure size for better readability
    
    for property_name in properties:
        r2_scores = []
        for layer in layers:
            layer_results = [r for r in results if r['layer'] == layer and r['property'] == property_name]
            # Calculate mean R² score for the layer
            mean_r2 = np.mean([r['r2'] for r in layer_results])
            # Set any R² value below -0.05 to -0.05
            if mean_r2 < -0.05:
                mean_r2 = -0.05
            r2_scores.append(mean_r2)
        
        # Plot the R² scores with crosses and lines
        plt.plot(layers, r2_scores, marker='x', linestyle='-', label=property_name)

    # Set the x-ticks to be the layer names
    plt.xticks(ticks=layers, labels=[f'Layer {layer}' for layer in layers])

    plt.title('R² Scores Across Layers for Different Properties')
    plt.xlabel('Layer')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)  # Add grid for better visibility of points and lines
    plt.show()

# Assuming train_features, test_features, train_properties, test_properties are already defined
results = evaluate_layer_probes(train_features, test_features, train_node_properties, test_node_properties)

import pickle

with open('results/'+DATASET+'_'+MODEL+'_node_probing.pkl', 'wb') as f:
    pickle.dump(results, f)

# %%
#save the results
# import pickle

# with open('results/'+DATASET+'_'+MODEL+'_node_probing.pkl', 'wb') as f:
#     pickle.dump(results, f)

#load the results
import pickle

with open('results/'+DATASET+'_'+MODEL+'_node_probing.pkl', 'rb') as f:
    results = pickle.load(f)

# %%
plot_results(results)

# %%
# Identify unique layers
layers = set(result['layer'] for result in results)

for layer in layers:
    # Filter results for the current layer
    layer_results = [result for result in results if result['layer'] == layer]
    
    # Extract and sort r2 values along with property names in descending order
    sorted_r2_values = sorted([(result['r2'], result['property']) for result in layer_results], reverse=True, key=lambda x: x[0])
    
    # Print the sorted r2 values with property names for the current layer
    print(f"Layer {layer}:")
    for r2, property in sorted_r2_values:
        print(f"  Property: {property}, R2: {r2}")
    print()  # Add a blank line for better readability



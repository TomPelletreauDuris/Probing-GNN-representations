import logging
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import GCNConv, BatchNorm
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)
# Define the GCN Model
class GCN_framework:
    def __init__(self, data, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.data = data[0].to(self.device)
        
        self.num_features = self.data.x.size(1)
        self.num_classes = torch.max(self.data.train_y).item() + 1  # Infer num classes from labels

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv_layers = torch.nn.ModuleList([
                    GCNConv(num_features if i == 0 else 128, 128)
                    for i in range(3)
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128) for _ in range(3)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
                self.bn2 = BatchNorm(num_classes)

            def forward(self, x, edge_index, return_intermediate=False):
                intermediates = []
                for i in range(3):  # 3 GCN layers
                    x = F.relu(self.conv_layers[i](x, edge_index))
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)

                x = F.relu(self.bn1(self.lin1(x)))
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn2(self.lin2(x))
                if return_intermediate:
                    intermediates.append(x)

                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)

        self.model = Net(self.num_features, self.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_idx], self.data.train_y)  # Use train_idx and train_y
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        accs = []
        for idx, y in [(self.data.train_idx, self.data.train_y), (self.data.test_idx, self.data.test_y)]:  # Use test_idx and test_y
            pred = logits[idx].argmax(dim=1)
            acc = (pred == y).sum().item() / idx.size(0)
            accs.append(acc)
        return accs

    def iterate(self):
        best_test_acc = 0
        for epoch in range(1, 141):
            loss = self.train()
            train_acc, test_acc = self.test()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model('best_model.pth')
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def evaluate(self):
        train_acc, test_acc = self.test()
        print(f'Final Evaluation - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')


    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        out, features = self.model(self.data.x, self.data.edge_index, return_intermediate=True)
        print(f"Features generated from model, shape of output: {[f.shape for f in features]}")

        train_indices = self.data.train_idx
        test_indices = self.data.test_idx

        # Check if train_indices and features sizes match before indexing
        for f in features:
            print(f"Feature shape: {f.shape}, train_indices max: {train_indices.max().item()}")

        # Ensure correct indexing
        if train_indices.max().item() >= features[0].size(0):
            raise IndexError("Train indices exceed the size of the feature tensors.")
        if test_indices.max().item() >= features[0].size(0):
            raise IndexError("Test indices exceed the size of the feature tensors.")

        # Gather features for train and test indices
        for feature_set in features:
            # Extract and append the features for the training indices
            train_features.append(feature_set[train_indices].cpu().numpy())
            # Extract and append the features for the testing indices
            test_features.append(feature_set[test_indices].cpu().numpy())

        return train_features, test_features




class GCN_framework_2:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()
        
        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 60)
                self.conv2 = GCNConv(60, 60)
                self.conv3 = GCNConv(60, 60)
                self.conv4 = GCNConv(60, 60)
                self.lin1 = Linear(60, 60)
                self.lin2 = Linear(60, 10)
                self.lin3 = Linear(10, num_classes)

            def forward(self, x, edge_index, edge_attr, batch=None, return_intermediate=False):
                """
                Embeddings : 4 layers of GCN, a global max pooling, 3 linear layers. Total 8 layers.
                """
                x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr.squeeze()))
                x2 = F.relu(self.conv2(x1, edge_index, edge_weight=edge_attr.squeeze()))
                x3 = F.relu(self.conv3(x2, edge_index, edge_weight=edge_attr.squeeze()))
                x4 = F.relu(self.conv4(x3, edge_index, edge_weight=edge_attr.squeeze()))
                x_global = global_max_pool(x4, batch)
                
                x5 = F.relu(self.lin1(x_global))
                x6 = F.relu(self.lin2(x5))
                x7 = self.lin3(x6)

                if return_intermediate:
                    return F.log_softmax(x7, dim=-1), (x1, x2, x3, x4, x_global, x5, x6, x7)
                else:
                    return F.log_softmax(x7, dim=-1)

        self.model = Net(116, num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256, shuffle=False)

    def _infer_num_classes(self):
        # Find maximum label value in the dataset and assume classes are 0-indexed
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1
    

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 201):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_attr, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])

        # Extract features for test data
        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_attr, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features






from torch_geometric.nn import SAGEConv

import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, Linear, BatchNorm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

class GraphSAGE_framework:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()
        
        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv_layers = torch.nn.ModuleList([
                    SAGEConv(num_features if i == 0 else 128, 128)
                    for i in range(5)  # Using 5 layers as specified
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128) for _ in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
                self.bn2 = BatchNorm(num_classes)

            def forward(self, x, edge_index, batch, return_intermediate=False):
                """
                Embeddings : 5 layers of GraphSAGE, a global mean pooling, 2 linear layers. Total 8 layers.
                """
                intermediates = []
                for i, conv in enumerate(self.conv_layers):
                    x = F.relu(conv(x, edge_index))
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)
                x = global_mean_pool(x, batch)
                
                x = F.relu(self.bn1(self.lin1(x)))
                x = self.bn2(self.lin2(x))
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)

        self.model = Net(116, num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256, shuffle=False)

    def _infer_num_classes(self):
        # Infer the number of classes from the dataset
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        self.scheduler.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 501):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_attr, data.batch, return_intermediate=True)
            train_features.append(features)  # Collect all desired features

        # Extract features for test data
        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_attr, data.batch, return_intermediate=True)
            test_features.append(features)  # Collect all desired features

        return train_features, test_features



class GraphSAGE_framework_2:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = SAGEConv(num_features, 60)
                self.conv2 = SAGEConv(60, 60)
                self.conv3 = SAGEConv(60, 60)
                self.conv4 = SAGEConv(60, 60)
                self.lin1 = Linear(60, 60)
                self.lin2 = Linear(60, 10)
                self.lin3 = Linear(10, num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None):
                """
                Embeddings : 4 layers of GraphSAGE, a global mean pooling, 3 linear layers. Total 8 layers.
                """
                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                x = F.relu(self.conv4(x, edge_index,edge_mask))

                
                x = global_mean_pool(x,batch)
                

                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                x = self.lin3(x)

                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.1) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 501):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
     

#GAT
from torch_geometric.nn import GATConv



from torch_geometric.nn import MLP, GINConv
class GIN_framework:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.gin_layers = torch.nn.ModuleList([
                    GINConv(torch.nn.Sequential(
                        torch.nn.Linear(num_features if i == 0 else 128, 128),  # Ensure this matches 'num_features'
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 128),
                        BatchNorm(128)
                    )) for i in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
            
            def forward(self, x, edge_index, batch=None, return_intermediate=False):
                """
                Embeddings : 5 layers of GIN, a global mean pooling, 2 linear layers. Total 8 layers.
                """
                intermediates = []
                for gin_layer in self.gin_layers:
                    x = gin_layer(x, edge_index)
                    x = F.relu(x)
                    if return_intermediate:
                        intermediates.append(x)
                x = global_mean_pool(x, batch)
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn1(self.lin1(x))
                if return_intermediate:
                    intermediates.append(x)
                x = F.relu(x)
                if return_intermediate:
                    intermediates.append(x)
                x = self.lin2(x)
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)


        self.model = Net(num_features=116, num_classes=num_classes).to(self.device).double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        # self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        # self.scheduler.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 401):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])


        return train_features, test_features
    
class GIN_framework_bis:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.gin_layers = torch.nn.ModuleList([
                    GINConv(torch.nn.Sequential(
                        torch.nn.Linear(num_features if i == 0 else 128, 128),  # Ensure this matches 'num_features'
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 128),
                        BatchNorm(128)
                    )) for i in range(5)
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128) for _ in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
            
            def forward(self, x, edge_index, batch=None, return_intermediate=False):
                """
                Embeddings : 5 layers of GIN, a global mean pooling, 2 linear layers. Total 8 layers."""
                intermediates = []
                for i, gin_layer in enumerate(self.gin_layers):
                    x = gin_layer(x, edge_index, )
                    x = F.relu(x)
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)
                x = global_mean_pool(x, batch)
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn1(self.lin1(x))
                if return_intermediate:
                    intermediates.append(x)
                x = F.relu(x)
                if return_intermediate:
                    intermediates.append(x)
                x = self.lin2(x)
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)


        self.model = Net(num_features=116, num_classes=num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index,data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        # self.scheduler.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 1301):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])


        return train_features, test_features
    
class GIN_framework_tri:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.gin_layers = torch.nn.ModuleList([
                    GINConv(torch.nn.Sequential(
                        torch.nn.Linear(num_features if i == 0 else 128, 128),  # Ensure this matches 'num_features'
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 128),
                        BatchNorm(128)
                    )) for i in range(5)
                ])
                self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(0.5) for _ in range(5)])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
            
            def forward(self, x, edge_index, batch=None, return_intermediate=False):
                """
                Embeddings : 5 layers of GIN, a global mean pooling, 2 linear layers. Total 8 layers.
                """
                intermediates = []
                for i, gin_layer in enumerate(self.gin_layers):
                    x = gin_layer(x, edge_index)
                    x = F.relu(x)
                    x = self.dropouts[i](x)
                    if return_intermediate:
                        intermediates.append(x)
                x = global_mean_pool(x, batch)
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn1(self.lin1(x))
                if return_intermediate:
                    intermediates.append(x)
                x = F.relu(x)
                if return_intermediate:
                    intermediates.append(x)
                x = self.lin2(x)
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)


        self.model = Net(num_features=116, num_classes=num_classes).to(self.device).double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        # self.scheduler.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 206):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])


        return train_features, test_features

class GIN_framework2:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.gin_layers = torch.nn.ModuleList([
                    GINConv(torch.nn.Sequential(
                        torch.nn.Linear(num_features if i == 0 else 256, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 256),
                        BatchNorm(256),
                        torch.nn.Dropout(0.5)  # Dropout layer added
                    )) for i in range(5)
                ])
                self.lin1 = Linear(256, 256)
                self.bn1 = BatchNorm(256)
                self.lin2 = Linear(256, num_classes)
                self.dropout = torch.nn.Dropout(0.5)  # Dropout layer added
            
            def forward(self, x, edge_index, batch=None, return_intermediate=False):
                """
                Embeddings : 5 layers of GIN, a global mean pooling, 2 linear layers. Total 8 layers."""
                intermediates = []
                for gin_layer in self.gin_layers:
                    x = gin_layer(x, edge_index)
                    x = F.relu(x)
                    if return_intermediate:
                        intermediates.append(x)
                x = global_mean_pool(x, batch)  # Experimenting with max pooling
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn1(self.lin1(x))
                if return_intermediate:
                    intermediates.append(x)
                x = F.relu(x)
                x = self.dropout(x)
                if return_intermediate:
                    intermediates.append(x)
                x = self.lin2(x)
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)

        num_features = dataset[0].x.shape[1]
        self.model = Net(num_features=num_features, num_classes=num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        self.scheduler.step()  # Scheduler step
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 521):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy(), f[9].cpu().numpy()) for f in zip(*features)])

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy(), f[9].cpu().numpy()) for f in zip(*features)])


        return train_features, test_features
class GIN_framework3:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.gin_layers = torch.nn.ModuleList([
                    GINConv(torch.nn.Sequential(
                        torch.nn.Linear(num_features if i == 0 else 128, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 128),
                        BatchNorm(128),
                        torch.nn.Dropout(0.5)
                    )) for i in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.bn1 = BatchNorm(128)
                self.lin2 = Linear(128, num_classes)
                self.dropout = torch.nn.Dropout(0.5)
            
            def forward(self, x, edge_index, batch=None, return_intermediate=False, return_node_embeddings=False):
                """
                Embeddings : 5 layers of GIN, a global mean pooling, 2 linear layers. Total 8 layers."""
                intermediates = []
                for gin_layer in self.gin_layers:
                    x = gin_layer(x, edge_index)
                    x = F.relu(x)
                    if return_intermediate:
                        intermediates.append(x)

                if return_node_embeddings:
                    return intermediates
                
                x = global_max_pool(x, batch)
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn1(self.lin1(x))
                if return_intermediate:
                    intermediates.append(x)
                x = F.relu(x)
                x = self.dropout(x)
                if return_intermediate:
                    intermediates.append(x)
                x = self.lin2(x)
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)

        num_features = dataset[0].x.shape[1]
        self.model = Net(num_features=num_features, num_classes=num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        self.kf = KFold(n_splits=10, shuffle=True, random_state=42)
        self.num_epochs = 900
        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)
        
        train_data = [self.dataset[i] for i in self.train_idx]
        test_data = [self.dataset[i] for i in self.test_idx]
        self.train_loader = DataLoader(train_data, batch_size=32)
        self.test_loader = DataLoader(test_data, batch_size=32)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        self.scheduler.step()
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def cross_validate(self):
        results = []
        for fold, (train_idx, test_idx) in enumerate(self.kf.split(self.dataset)):
            train_data = [self.dataset[i] for i in train_idx]
            test_data = [self.dataset[i] for i in test_idx]
            self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            self.test_loader = DataLoader(test_data, batch_size=32)

            self.model.apply(self._reset_parameters)  # Reset model parameters for each fold

            for epoch in range(1, self.num_epochs + 1):
                train_loss = self.train(self.train_loader)
                train_acc, train_loss = self.test(self.train_loader)
                test_acc, test_loss = self.test(self.test_loader)
                if epoch % 5 == 0:
                    print(f'Fold: {fold + 1}, Epoch: {epoch}, Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
            
            test_acc, test_loss = self.test(self.test_loader)
            results.append(test_acc)
            print(f'Fold {fold + 1}, Test Accuracy: {test_acc:.3f}')

        avg_accuracy = np.mean(results)
        print(f'Average Test Accuracy: {avg_accuracy:.3f}')

    def _reset_parameters(self, m):
        if isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm1d)):
            m.reset_parameters()

    def iterate(self):
        idx = torch.arange(len(self.dataset))

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train(self.train_loader)
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self, loader=None):
        if loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided and no default test loader available.")
            loader = self.test_loader
        acc, loss = self.test(loader)
        print(f'Accuracy: {acc:.3f}, Loss: {loss:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self, loader=None, return_node_embeddings=False):
        if loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided and no default test loader available.")
            loader = self.test_loader

        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)
                print("features[2].shape: ",features[2].shape)
                print("features[3].shape: ",features[3].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:

            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features

# Usage example
# dataset = load_your_dataset_here()
# gnn3 = GIN_framework3(dataset)
# gnn3.cross_validate()
# gnn3.iterate()

# Evaluate using the default test_loader from the last iteration or cross-validation
# gnn3.evaluate()

# Or evaluate with a specific loader
# test_loader = DataLoader(test_dataset, batch_size=32)  # Define your test dataset loader
# gnn3.evaluate(test_loader)


import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_max_pool, Linear
from torch.nn import Dropout
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import logging

class RGCN_framework:
    def __init__(self, dataset, device=None, num_relations=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.dataset = dataset
        self.data = dataset[0].to(self.device)
        self.num_relations = self._infer_num_relations()
        self.num_features = self.data.x.size(1)
        self.num_classes = torch.max(self.data.train_y).item() + 1  # Infer num classes from labels
        print(f"Number of classes: {self.num_classes}")

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels, num_relations):
                super(Net, self).__init__()
                self.conv1 = RGCNConv(in_channels, 128, num_relations)
                # self.drop1 = Dropout(0.2)
                self.conv2 = RGCNConv(128, 128, num_relations)
                # self.drop2 = Dropout(0.2)
                self.conv3 = RGCNConv(128, 128, num_relations)
                # self.drop3 = Dropout(0.2)
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, out_channels)

            def forward(self, x, edge_index, edge_type, return_intermediate=False):
                x1 = F.relu(self.conv1(x, edge_index, edge_type))
                # x1 = self.drop1(x1)
                x2 = F.relu(self.conv2(x1, edge_index, edge_type))
                # x2 = self.drop2(x2)
                x3 = F.relu(self.conv3(x2, edge_index, edge_type))
                # x_global = global_max_pool(x2, batch)
                x = F.relu(self.lin1(x3))
                out = self.lin2(x)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x3, x, out)
                else:
                    return F.log_softmax(out, dim=-1)

        self.model = Net(self.num_features, self.num_classes, self.num_relations).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        
    def _infer_num_relations(self):
        edge_types = torch.cat([data.edge_type for data in self.dataset])
        unique_edge_types = torch.unique(edge_types)
        return len(unique_edge_types)

    def train(self):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_type)
            loss = F.nll_loss(out[self.data.train_idx], self.data.train_y)  # Use train_idx and train_y
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            return loss.item()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index, self.data.edge_type)
        accs = []
        for idx, y in [(self.data.train_idx, self.data.train_y), (self.data.test_idx, self.data.test_y)]:  # Use test_idx and test_y
            pred = logits[idx].argmax(dim=1)
            acc = (pred == y).sum().item() / idx.size(0)
            accs.append(acc)
        return accs

    def iterate(self):
        best_test_acc = 0
        for epoch in range(1, 23):
            loss = self.train()
            train_acc, test_acc = self.test()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model('best_model.pth')
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info("Model saved in: " + path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info("Model loaded from: " + path)

    def evaluate(self):
            train_acc, test_acc = self.test()
            print(f'Final Evaluation - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        out, features = self.model(self.data.x, self.data.edge_index, self.data.edge_type, return_intermediate=True)
        print(f"Features generated from model, shape of output: {[f.shape for f in features]}")

        train_indices = self.data.train_idx
        test_indices = self.data.test_idx

        # Check if train_indices and features sizes match before indexing
        for f in features:
            print(f"Feature shape: {f.shape}, train_indices max: {train_indices.max().item()}")

        # Ensure correct indexing
        if train_indices.max().item() >= features[0].size(0):
            raise IndexError("Train indices exceed the size of the feature tensors.")
        if test_indices.max().item() >= features[0].size(0):
            raise IndexError("Test indices exceed the size of the feature tensors.")

        # Gather features for train and test indices
        for feature_set in features:
            # Extract and append the features for the training indices
            train_features.append(feature_set[train_indices].cpu().numpy())
            # Extract and append the features for the testing indices
            test_features.append(feature_set[test_indices].cpu().numpy())

        return train_features, test_features

import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_max_pool, Linear, BatchNorm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

class GAT_framework:
    def __init__(self, dataset, device=None, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        if num_classes is None:
            num_classes = self._infer_num_classes()

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv_layers = torch.nn.ModuleList([
                    GATConv(num_features if i == 0 else 128*4, 128, heads=4, concat=True)  # Add heads parameter
                    for i in range(5)
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128*4) for _ in range(5)
                ])
                self.lin1 = Linear(128*4, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
                self.bn2 = BatchNorm(num_classes)

            def forward(self, x, edge_index, batch=None, return_intermediate=False, return_node_embeddings=False):
                intermediates = []
                for i, conv in enumerate(self.conv_layers):
                    x = F.relu(conv(x, edge_index))
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)

                if return_node_embeddings:
                    return intermediates

                x_global = global_max_pool(x, batch)
                if return_intermediate:
                    intermediates.append(x_global)                
                x = F.relu(self.bn1(self.lin1(x_global)))
                if return_intermediate:
                    intermediates.append(x)
                x = self.bn2(self.lin2(x))
                if return_intermediate:
                    intermediates.append(x)
                if return_intermediate:
                    return F.log_softmax(x, dim=-1), intermediates
                else:
                    return F.log_softmax(x, dim=-1)

        self.model = Net(116, num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        # self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=False)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256, shuffle=False)

    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        # self.scheduler.step()
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

    def iterate(self):
        for epoch in range(1, 151):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                      f'Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)

    def load_model(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval()

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)
                print("features[2].shape: ",features[2].shape)
                print("features[3].shape: ",features[3].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:

            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features

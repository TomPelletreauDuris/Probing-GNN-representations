
import logging
logging.basicConfig(level=logging.INFO)
from torch.nn import Linear
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, Linear, BatchNorm, GATv2Conv,global_max_pool,global_add_pool,global_mean_pool, GATConv, RGCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

class GCN_framework:
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
                    GCNConv(num_features if i == 0 else 128, 128)
                    for i in range(5)
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128) for _ in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
                self.bn2 = BatchNorm(num_classes)

            def forward(self, x, edge_index, edge_attr, batch=None, return_intermediate=False):
                intermediates = []
                for i in range(5):
                    x = F.relu(self.conv_layers[i](x, edge_index, edge_weight=edge_attr.squeeze()))
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)
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
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256, shuffle=False)

    def _infer_num_classes(self):
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
        for epoch in range(1, 141):
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
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

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
        self.scheduler.step()
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

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy(), f[8].cpu().numpy()) for f in zip(*features)])


        return train_features, test_features


import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_max_pool, Linear
from torch.nn import Dropout
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import logging

class RGCN_framework:
    def __init__(self, dataset, device=None, num_relations=2, num_classes=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        self.num_relations = num_relations
        if num_classes is None:
            num_classes = self._infer_num_classes()

        num_features = dataset[0].x.shape[1]  # Assuming 'x' is present and correctly shaped

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.conv1 = RGCNConv(in_channels, 64, num_relations)
                self.drop1 = Dropout(0.2)
                self.conv2 = RGCNConv(64, 64, num_relations)
                self.drop2 = Dropout(0.2)
                self.conv3 = RGCNConv(64, 64, num_relations)
                self.drop3 = Dropout(0.2)
                self.lin1 = Linear(64, 64)
                self.lin2 = Linear(64, out_channels)

            def forward(self, x, edge_index, edge_type, batch, return_intermediate=False):
                x1 = F.relu(self.conv1(x, edge_index, edge_type))
                x1 = self.drop1(x1)
                x2 = F.relu(self.conv2(x1, edge_index, edge_type))
                x2 = self.drop2(x2)
                x_global = global_max_pool(x2, batch)
                x = F.relu(self.lin1(x_global))
                out = self.lin2(x)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x_global, x, out)
                else:
                    return F.log_softmax(out, dim=-1)

        self.model = Net(num_features, num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader([self.dataset[i] for i in self.test_idx], batch_size=256, shuffle=False)

        # Add edge_type to the data with edge_type always 1
        # Add edge_type to the data with edge_type always 1
        train_data_list = []
        for data in self.train_loader.dataset:
            data.edge_type = torch.ones(data.edge_index.shape[1], dtype=torch.long)
            train_data_list.append(data)
        self.train_loader = DataLoader(train_data_list, batch_size=self.train_loader.batch_size)

        test_data_list = []
        for data in self.test_loader.dataset:
            data.edge_type = torch.ones(data.edge_index.shape[1], dtype=torch.long)
            test_data_list.append(data)
        self.test_loader = DataLoader(test_data_list, batch_size=self.test_loader.batch_size)



    def _infer_num_classes(self):
        max_label = max(data.y.max().item() for data in self.dataset)
        return max_label + 1

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            output = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = F.nll_loss(output, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_type, data.batch)
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
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info("Model saved in: " + path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info("Model loaded from: " + path)

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            train_features.extend(features)

        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            test_features.extend(features)

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
                    GATConv(num_features if i == 0 else 128, 128)
                    for i in range(5)
                ])
                self.batch_norms = torch.nn.ModuleList([
                    BatchNorm(128) for _ in range(5)
                ])
                self.lin1 = Linear(128, 128)
                self.lin2 = Linear(128, num_classes)
                self.bn1 = BatchNorm(128)
                self.bn2 = BatchNorm(num_classes)

            def forward(self, x, edge_index, batch=None, return_intermediate=False):
                intermediates = []
                for i, conv in enumerate(self.conv_layers):
                    x = F.relu(conv(x, edge_index))
                    x = self.batch_norms[i](x)
                    if return_intermediate:
                        intermediates.append(x)
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
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.95, stratify=[data.y.numpy() for data in self.dataset], random_state=10)

        self.train_loader = DataLoader([self.dataset[i] for i in self.train_idx], batch_size=256, shuffle=True)
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
        self.scheduler.step()
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
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            train_features.extend(features)

        # Extract features for test data
        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            test_features.extend(features)

        return train_features, test_features

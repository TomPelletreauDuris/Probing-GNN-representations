
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv,GATv2Conv,global_max_pool,global_add_pool,global_mean_pool, GATConv, RGCNConv
from torch_geometric.loader import DataLoader
import logging
logging.basicConfig(level=logging.INFO)
from torch.nn import Linear
import networkx as nx
from torch_geometric.utils import to_networkx
#dropout
from torch.nn import Dropout

class GCN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 60)
                self.conv2 = GCNConv(60, 60)
                self.conv3 = GCNConv(60, 60)
                self.conv4 = GCNConv(60, 60)
                self.lin1 = Linear(60,60)
                self.lin2 = Linear(60,10)
                self.lin3 = Linear(10,num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None, return_intermediate=False, return_node_embeddings=False):

                x1 = F.relu(self.conv1(x, edge_index,edge_mask))
                x2 = F.relu(self.conv2(x1, edge_index,edge_mask))
                x3 = F.relu(self.conv3(x2, edge_index,edge_mask))
                x4 = F.relu(self.conv4(x3, edge_index,edge_mask))

                if return_node_embeddings:
                    print("x1 shape:", x1.shape)
                    print("x2 shape:", x2.shape)
                    print("x3 shape:", x3.shape)
                    print("x4 shape:", x4.shape)
                    return (x1, x2, x3, x4)

                x_global = global_max_pool(x4,batch)                
                x5 = F.relu(self.lin1(x_global))
                x6 = F.relu(self.lin2(x5))
                x7 = self.lin3(x6)

                if return_intermediate:
                    return F.log_softmax(x7, dim=-1), (x1, x2, x3, x4, x_global, x5, x6, x7)
                else:
                    return F.log_softmax(x7, dim=-1)


            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=64)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
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

        for epoch in range(1, 1001):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self, path, map_location=None):
        if map_location == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs

        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
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
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])

        return train_features, test_features
    
class GCN_framework_xavier:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 60)
                self.conv2 = GCNConv(60, 60)
                self.conv3 = GCNConv(60, 60)
                self.conv4 = GCNConv(60, 60)
                self.lin1 = Linear(60,60)
                self.lin2 = Linear(60,10)
                self.lin3 = Linear(10,num_classes)

                # Initialize weights
                self.initialize_weights()   

            def initialize_weights(self):
                # Initialize convolutional layers
                for m in self.modules():
                    if isinstance(m, GCNConv):
                        torch.nn.init.xavier_uniform_(m.lin.weight)
                        if m.lin.bias is not None:
                            torch.nn.init.zeros_(m.lin.bias)
                    elif isinstance(m, Linear):
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)

            def forward(self,x,edge_index,batch,edge_mask=None, return_intermediate=False, return_node_embeddings=False):

                x1 = F.relu(self.conv1(x, edge_index,edge_mask))
                x2 = F.relu(self.conv2(x1, edge_index,edge_mask))
                x3 = F.relu(self.conv3(x2, edge_index,edge_mask))
                x4 = F.relu(self.conv4(x3, edge_index,edge_mask))

                if return_node_embeddings:
                    print("x1 shape:", x1.shape)
                    print("x2 shape:", x2.shape)
                    print("x3 shape:", x3.shape)
                    print("x4 shape:", x4.shape)
                    return (x1, x2, x3, x4)

                x_global = global_max_pool(x4,batch)                
                x5 = F.relu(self.lin1(x_global))
                x6 = F.relu(self.lin2(x5))
                x7 = self.lin3(x6)

                if return_intermediate:
                    return F.log_softmax(x7, dim=-1), (x1, x2, x3, x4, x_global, x5, x6, x7)
                else:
                    return F.log_softmax(x7, dim=-1)


        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=64)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
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

        for epoch in range(1, 1001):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self, path, map_location=None):
        if map_location == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs

        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
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
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])
                
        return train_features, test_features


class GCN_framework_L2:
    def __init__(self, dataset, device=None):   
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset

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

                    # Initialize weights
                self.initialize_weights()   

            def initialize_weights(self):
                # Initialize convolutional layers
                for m in self.modules():
                    if isinstance(m, GCNConv):
                        torch.nn.init.xavier_uniform_(m.lin.weight)
                        if m.lin.bias is not None:
                            torch.nn.init.zeros_(m.lin.bias)
                    elif isinstance(m, Linear):
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)

            def forward(self, x, edge_index, batch, edge_mask=None, return_intermediate=False, return_node_embeddings=False):
                x1 = F.relu(self.conv1(x, edge_index, edge_mask))
                x2 = F.relu(self.conv2(x1, edge_index, edge_mask))
                x3 = F.relu(self.conv3(x2, edge_index, edge_mask))
                x4 = F.relu(self.conv4(x3, edge_index, edge_mask))

                if return_node_embeddings:
                    return (x1, x2, x3, x4)

                x_global = global_max_pool(x4, batch)                
                x5 = F.relu(self.lin1(x_global))
                x6 = F.relu(self.lin2(x5))
                x7 = self.lin3(x6)

                if return_intermediate:
                    return F.log_softmax(x7, dim=-1), (x1, x2, x3, x4, x_global, x5, x6, x7)
                else:
                    return F.log_softmax(x7, dim=-1)

        self.model = Net(10, self.dataset.num_classes).to(self.device).double()

        # **Added weight_decay parameter here**
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)  # Adjust weight_decay as needed

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=64)
                
    def train(self):   
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
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
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(self.train_loader.dataset)

    def iterate(self):
        for epoch in range(1, 1001):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    # The rest of the methods remain unchanged
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)
            
    def load_model(self, path, map_location=None):
        if map_location == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
            
    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs

        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
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
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])
                
        return train_features, test_features
    
    
class GCN_framework_Dropout:
    def __init__(self, dataset, device=None):   
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset

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
                # **Added dropout layers**
                self.dropout = Dropout(p=0.2)  # Dropout probability of 0.2

                    # Initialize weights
                self.initialize_weights()   

            def initialize_weights(self):
                # Initialize convolutional layers
                for m in self.modules():
                    if isinstance(m, GCNConv):
                        torch.nn.init.xavier_uniform_(m.lin.weight)
                        if m.lin.bias is not None:
                            torch.nn.init.zeros_(m.lin.bias)
                    elif isinstance(m, Linear):
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)

            def forward(self, x, edge_index, batch, edge_mask=None, return_intermediate=False, return_node_embeddings=False):
                x1 = F.relu(self.conv1(x, edge_index, edge_mask))
                x1 = self.dropout(x1)  # Apply dropout
                x2 = F.relu(self.conv2(x1, edge_index, edge_mask))
                x2 = self.dropout(x2)
                x3 = F.relu(self.conv3(x2, edge_index, edge_mask))
                x3 = self.dropout(x3)
                x4 = F.relu(self.conv4(x3, edge_index, edge_mask))
                x4 = self.dropout(x4)

                if return_node_embeddings:
                    return (x1, x2, x3, x4)

                x_global = global_max_pool(x4, batch)                
                x5 = F.relu(self.lin1(x_global))
                x5 = self.dropout(x5)
                x6 = F.relu(self.lin2(x5))
                x6 = self.dropout(x6)
                x7 = self.lin3(x6)

                if return_intermediate:
                    return F.log_softmax(x7, dim=-1), (x1, x2, x3, x4, x_global, x5, x6, x7)
                else:
                    return F.log_softmax(x7, dim=-1)

        self.model = Net(10, self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=64)
                
    def train(self):   
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y.view(-1))
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
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_correct / len(loader.dataset), total_loss / len(self.train_loader.dataset)

    def iterate(self):
        for epoch in range(1, 1001):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    # The rest of the methods remain unchanged
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in:", path)
            
    def load_model(self, path, map_location=None):
        if map_location == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
            
    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs

        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
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
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])
                
        return train_features, test_features


from torch_geometric.nn import SAGEConv

class GraphSAGE_framework:
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
class GAT_framework_:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GATConv(num_features, 60)
                self.conv2 = GATConv(60, 60)
                self.conv3 = GATConv(60, 60)
                self.conv4 = GATConv(60, 60)
                self.lin1 = Linear(60, 60)
                self.lin2 = Linear(60, 10)
                self.lin3 = Linear(10, num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None):
                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                x = F.relu(self.conv4(x, edge_index,edge_mask))

                
                x = global_max_pool(x,batch)
                

                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                x = self.lin3(x)

                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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


from torch_geometric.nn import GATConv
class GAT_framework_2:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv1 = GATConv(num_features, 60)
                self.conv2 = GATConv(60, 60)
                self.conv3 = GATConv(60, 60)
                self.conv4 = GATConv(60, 60)
                self.lin1 = Linear(60, 60)
                self.lin2 = Linear(60, 10)
                self.lin3 = Linear(10, num_classes)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, x, edge_index, batch, edge_mask=None):
                x = F.relu(self.conv1(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.relu(self.conv3(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.relu(self.conv4(x, edge_index, edge_mask))
                x = self.dropout(x)
                
                x = global_max_pool(x, batch)
                
                x = F.relu(self.lin1(x))
                x = self.dropout(x)
                x = F.relu(self.lin2(x))
                x = self.lin3(x)

                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=256)

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

class GAT_Framework_3:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.conv1 = GATConv(num_features, 60, heads=8, dropout=0.6)
                self.conv2 = GATConv(60 * 8, 60, heads=8, dropout=0.6)
                self.conv3 = GATConv(60 * 8, 60, heads=8, dropout=0.6)
                self.conv4 = GATConv(60 * 8, 60, heads=8, dropout=0.6)
                self.lin1 = Linear(60 * 8, 60)
                self.lin2 = Linear(60, 10)
                self.lin3 = Linear(10, num_classes)
                self.dropout = torch.nn.Dropout(p=0.6)

            def forward(self, x, edge_index, batch, edge_mask=None):
                x = F.elu(self.conv1(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.elu(self.conv2(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.elu(self.conv3(x, edge_index, edge_mask))
                x = self.dropout(x)
                x = F.elu(self.conv4(x, edge_index, edge_mask))
                x = self.dropout(x)
                
                x = global_mean_pool(x, batch)
                
                x = F.relu(self.lin1(x))
                x = self.dropout(x)
                x = F.relu(self.lin2(x))
                x = self.lin3(x)

                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=256)

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

class GAT_Framework_4:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.conv1 = GATConv(num_features, 32 * 4, heads=8)
                self.bn1 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.conv2 = GATConv(32 * 4 * 8, 32 * 4, heads=8)
                self.bn2 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.conv3 = GATConv(32 * 4 * 8, 32 * 4, heads=8)
                self.bn3 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.lin1 = Linear(32 * 4 * 8, 20)
                self.lin2 = Linear(20, num_classes)
                self.dropout = torch.nn.Dropout(p=0.6)

                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            def forward(self, x, edge_index, batch, edge_mask=None):
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = F.elu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                x = self.bn2(x)
                x = F.elu(x)
                x = self.dropout(x)
                x = self.conv3(x, edge_index)
                x = self.bn3(x)
                x = F.elu(x)
                x = self.dropout(x)
                x = global_max_pool(x, batch)
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=256)

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

class GAT_Framework_5:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super(Net, self).__init__()
                self.conv1 = GATConv(num_features, 32 * 4, heads=8)
                #self.bn1 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.conv2 = GATConv(32 * 4 * 8, 32 * 4, heads=8)
                #self.bn2 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.conv3 = GATConv(32 * 4 * 8, 32 * 4, heads=8)
                #self.bn3 = torch.nn.BatchNorm1d(32 * 4 * 8)
                self.lin1 = Linear(32 * 4 * 8, 20)
                self.lin2 = Linear(20, num_classes)
                #self.dropout = torch.nn.Dropout(p=0.6)

                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            def forward(self, x, edge_index, batch, edge_mask=None, return_intermediate=False, return_node_embeddings=False):
                x = self.conv1(x, edge_index)
                #x = self.bn1(x)
                x1 = F.relu(x)
                #x = self.dropout(x)
                x2 = self.conv2(x1, edge_index)
                #x = self.bn2(x)
                x2 = F.relu(x2)
                #x = self.dropout(x)
                x3 = self.conv3(x2, edge_index)
                #x = self.bn3(x)
                x3 = F.relu(x3)
                #x = self.dropout(x)

                if return_node_embeddings:
                    print("x1 shape:", x.shape)
                    print("x2 shape:", x1.shape)
                    print("x3 shape:", x2.shape)
                    print("x4 shape:", x3.shape)
                    return (x, x1, x2, x3)

                x_max_pool = global_max_pool(x3, batch)
                x_2 = F.relu(self.lin1(x_max_pool))
                x_3 = self.lin2(x_2)
                            
                if return_intermediate:
                    return F.log_softmax(x_3, dim=-1), (x, x1, x2, x3, x_max_pool, x_2, x_3)
                else: 
                    return F.log_softmax(x_3, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=64)

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        for epoch in range(1, 121):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy()) for f in zip(*features)])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().cpu().numpy(), f[6].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features


class GATV2Framework:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv1 = GATv2Conv(num_features, 10)
                self.conv2 = GATv2Conv(10, 20)
                self.conv3 = GATv2Conv(20, 10)
                self.lin1 = Linear(10, 20)
                self.lin2 = Linear(20, num_classes)

            def forward(self, x, edge_index, batch):
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = global_mean_pool(x, batch)  # Global pooling for graph classification
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)

        self.model = Net(10, self.dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=6428, shuffle=True)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=6428, shuffle=False)
        
    def train(self):   
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach() * data.num_graphs
        return total_loss.item() / len(self.train_loader.dataset)

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
            total_loss += loss.detach() * data.num_graphs
        return total_correct / len(loader.dataset), total_loss.item() / len(loader.dataset)
    
    def iterate(self, num_epochs=100):
        for epoch in range(1, num_epochs+1):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')


from torch_geometric.nn import MLP, GINConv
class GIN_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)

                self.lin1 = Linear(30,30)
                self.lin2 = Linear(30,out_channels)

                

            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = x.relu()
                x = self.conv2(x, edge_index)
                x = x.relu()

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)
     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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

        for epoch in range(1, 1001):
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

class GIN_framework3:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)
                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)

            def forward(self, x, edge_index, batch, return_intermediate=False, return_node_embeddings=False):
                x1 = self.conv1(x, edge_index).relu()
                x2 = self.conv2(x1, edge_index).relu()

                if return_node_embeddings:
                    return (x1, x2)
                
                x_global = global_mean_pool(x2, batch)
                x_lin1 = F.relu(self.lin1(x_global))
                out = self.lin2(x_lin1)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x_global, x_lin1, out)
                else:
                    return F.log_softmax(out, dim=-1)
            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=64)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=64)
            
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

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)  # Store the features for later analysis

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected

    def iterate(self):

        for epoch in range(1, 201):
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

    def evaluate_with_features(self):

        train_acc,train_loss,train_features = self.test_with_features(self.train_loader)
        test_acc,test_loss,test_features = self.test_with_features(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
        print(f'Number of features collected: {len(train_features)}')

        return train_features, test_features
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy()) for f in zip(*features)])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features
    
class GIN_framework4:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)
                self.mlp3 = torch.nn.Linear(30, 30)
                self.conv3 = GINConv(self.mlp3)
                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)

            def forward(self, x, edge_index, batch, return_intermediate=False, return_node_embeddings=False):
                x1 = self.conv1(x, edge_index).relu()
                x2 = self.conv2(x1, edge_index).relu()
                x3 = self.conv3(x2, edge_index).relu()

                if return_node_embeddings:
                    return (x1, x2, x3)
                
                x_global = global_mean_pool(x3, batch)
                x_lin1 = F.relu(self.lin1(x_global))
                out = self.lin2(x_lin1)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x3, x_global, x_lin1, out)
                else:
                    return F.log_softmax(out, dim=-1)
            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=1)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=1)
            
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

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)  # Store the features for later analysis

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected

    def iterate(self):

        for epoch in range(1, 201):
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

    def evaluate_with_features(self):

        train_acc,train_loss,train_features = self.test_with_features(self.train_loader)
        test_acc,test_loss,test_features = self.test_with_features(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
        print(f'Number of features collected: {len(train_features)}')

        return train_features, test_features
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])

        return train_features, test_features
    
class GIN_framework4_L2:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)
                self.mlp3 = torch.nn.Linear(30, 30)
                self.conv3 = GINConv(self.mlp3)
                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)

            def forward(self, x, edge_index, batch, return_intermediate=False, return_node_embeddings=False):
                x1 = self.conv1(x, edge_index).relu()
                x2 = self.conv2(x1, edge_index).relu()
                x3 = self.conv3(x2, edge_index).relu()

                if return_node_embeddings:
                    return (x1, x2, x3)
                
                x_global = global_mean_pool(x3, batch)
                x_lin1 = F.relu(self.lin1(x_global))
                out = self.lin2(x_lin1)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x3, x_global, x_lin1, out)
                else:
                    return F.log_softmax(out, dim=-1)
            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001, weight_decay=1e-2)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=1)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=1)
            
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

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)  # Store the features for later analysis

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected

    def iterate(self):

        for epoch in range(1, 201):
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

    def evaluate_with_features(self):

        train_acc,train_loss,train_features = self.test_with_features(self.train_loader)
        test_acc,test_loss,test_features = self.test_with_features(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
        print(f'Number of features collected: {len(train_features)}')

        return train_features, test_features
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])

        return train_features, test_features


class GIN_framework4_dropout:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Linear(in_channels, 30)
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Linear(30, 30)
                self.conv2 = GINConv(self.mlp2)
                self.mlp3 = torch.nn.Linear(30, 30)
                self.conv3 = GINConv(self.mlp3)
                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)

                self.dropout = torch.nn.Dropout(p=0.2)


            def forward(self, x, edge_index, batch, return_intermediate=False, return_node_embeddings=False):
                x1 = self.conv1(x, edge_index).relu()
                x1 = self.dropout(x1)
                x2 = self.conv2(x1, edge_index).relu()
                x2 = self.dropout(x2)
                x3 = self.conv3(x2, edge_index).relu()
                x3 = self.dropout(x3)

                if return_node_embeddings:
                    return (x1, x2, x3)
                
                x_global = global_mean_pool(x3, batch)
                x_lin1 = F.relu(self.lin1(x_global))
                x_lin1 = self.dropout(x_lin1)
                out = self.lin2(x_lin1)

                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x3, x_global, x_lin1, out)
                else:
                    return F.log_softmax(out, dim=-1)
            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=1)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=1)
            
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

    @torch.no_grad()
    def test_with_features(self, loader):
        self.model.eval()
        features_collected = []

        total_correct = 0
        total_loss = 0
        print("len(loader.dataset): ",len(loader.dataset))

        for data in loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
            print("features: ",features)
            features_collected.append(features)  # Store the features for later analysis

            total_correct += int((out.argmax(-1) == data.y).sum())
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset), total_loss / len(loader.dataset), features_collected

    def iterate(self):

        for epoch in range(1, 201):
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

    def evaluate_with_features(self):

        train_acc,train_loss,train_features = self.test_with_features(self.train_loader)
        test_acc,test_loss,test_features = self.test_with_features(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
        print(f'Number of features collected: {len(train_features)}')

        return train_features, test_features
    
    @torch.no_grad()
    def evaluate_with_features2(self, return_node_embeddings=False):
        self.model.eval()
        train_features = []
        test_features = []

        if return_node_embeddings:
            for data in self.train_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                print("len of features: ",len(features))
                print("features[0].shape: ",features[0].shape)
                print("features[1].shape: ",features[1].shape)

                train_features.append([f.cpu().numpy() for f in features])
                #check shape of feature 0 in train_features
                # print("train_features[0].shape: ",train_features[0].shape)
                print("train_features[0][0].shape: ",train_features[0][0].shape)

            for data in self.test_loader:
                data = data.to(self.device)
                features = self.model(data.x, data.edge_index, data.batch, return_node_embeddings=True)
                test_features.append([f.cpu().numpy() for f in features])

        else:
            # Extract features for training data
            for data in self.train_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                train_features.append([f.cpu().numpy() for f in features])

            # Extract features for test data
            for data in self.test_loader:
                data = data.to(self.device)
                out, features = self.model(data.x, data.edge_index, data.batch, return_intermediate=True)
                # test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy(), f[5].cpu().numpy(), f[6].cpu().numpy(), f[7].cpu().numpy()) for f in zip(*features)])
                test_features.append([f.cpu().numpy() for f in features])

        return train_features, test_features   

        
class GINFramework2:
    def __init__(self, dataset, device=None):   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64)
                )
                self.conv1 = GINConv(self.mlp1)
                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64)
                )
                self.conv2 = GINConv(self.mlp2)

                self.lin1 = Linear(64, 64)
                self.lin2 = Linear(64, out_channels)

            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = x.relu()
                x = self.conv2(x, edge_index)
                x = x.relu()
                x = global_mean_pool(x, batch)
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=10)

        self.train_loader = DataLoader(self.dataset[self.train_idx], batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx], batch_size=256)
            
    def train(self):   
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        for epoch in range(1, 1001):
            loss = self.train()
            train_acc, train_loss = self.test(self.train_loader)
            test_acc, test_loss = self.test(self.test_loader)
            if epoch % 20 == 0:
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
        
    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):
        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')



from torch_geometric.nn import ChebConv

class Cheb_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset


        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.conv1 = ChebConv(in_channels,30,K=5)
                self.conv2 = ChebConv(30,30,K=5)
                self.conv3 = ChebConv(30,30,K=5)
                self.lin1 = Linear(30,30)
                self.lin2 = Linear(30,out_channels)


            def forward(self, x, edge_index, batch,edge_weight=None):
                x = self.conv1(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv2(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv3(x, edge_index,edge_weight)

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)


     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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

        for epoch in range(1, 1001):
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


class RGCN_framework:
    def __init__(self, dataset, device=None, num_relations=2):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.dataset = dataset
        self.num_relations = num_relations

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.conv1 = RGCNConv(in_channels, 30, num_relations)
                self.conv2 = RGCNConv(30, 30, num_relations)
                self.lin1 = torch.nn.Linear(30, 30)
                self.lin2 = torch.nn.Linear(30, out_channels)

            def forward(self, x, edge_index, edge_type, batch, return_intermediate=False):
                x1 = F.relu(self.conv1(x, edge_index, edge_type))
                x2 = F.relu(self.conv2(x1, edge_index, edge_type))

                
                x_global = global_max_pool(x2, batch)
                x_lin1 = F.relu(self.lin1(x_global))
                out = self.lin2(x_lin1)
                
                if return_intermediate:
                    return F.log_softmax(out, dim=-1), (x1, x2, x_global, x_lin1, out)
                else:
                    return F.log_softmax(out, dim=-1)

        self.model = Net(dataset.num_features, dataset.num_classes).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        idx = torch.arange(len(dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=dataset.data.y, random_state=10)

        self.train_loader = DataLoader(dataset[self.train_idx], batch_size=256, shuffle=True)
        self.test_loader = DataLoader(dataset[self.test_idx], batch_size=256, shuffle=False)

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

        

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = F.nll_loss(output, data.y)
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
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved in: {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logging.info(f"Model loaded from: {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def evaluate(self):

        train_acc, train_loss = self.test(self.train_loader)
        test_acc, test_loss = self.test(self.test_loader)
        logging.info(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')

    def evaluate_with_features(self):

        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            train_features.extend([(f[2].cpu().numpy(), f[3].cpu().numpy()) for f in zip(*features)])

        # Extract features for test data
        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            test_features.extend([(f[2].cpu().numpy(), f[3].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features
    
    @torch.no_grad()
    def evaluate_with_features2(self):
        self.model.eval()
        train_features = []
        test_features = []

        # Extract features for training data
        for data in self.train_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            train_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy()) for f in zip(*features)])

        # Extract features for test data
        for data in self.test_loader:
            data = data.to(self.device)
            out, features = self.model(data.x, data.edge_index, data.edge_type, data.batch, return_intermediate=True)
            test_features.extend([(f[0].cpu().numpy(), f[1].cpu().numpy(), f[2].cpu().numpy(), f[3].cpu().numpy(), f[4].cpu().numpy()) for f in zip(*features)])

        return train_features, test_features

# Set2Set

from torch_geometric.nn import Set2Set

class Set2set_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 60)
                self.conv2 = GCNConv(60, 60)
                self.conv3 = GCNConv(60, 60)
                self.conv4 = GCNConv(60, 60)
                self.set2set = Set2Set(60,7)
                self.lin1 = Linear(120,60)
                self.lin2 = Linear(60,10)
                self.lin3 = Linear(10,num_classes)

            def forward(self,x,edge_index,batch,edge_mask=None):

                x = F.relu(self.conv1(x, edge_index,edge_mask))
                x = F.relu(self.conv2(x, edge_index,edge_mask))
                x = F.relu(self.conv3(x, edge_index,edge_mask))
                x = F.relu(self.conv4(x, edge_index,edge_mask))
                #x = F.relu(self.conv3(x, edge_index,edge_mask))
                #x = self.conv4(x, edge_index,edge_mask)

                x = self.set2set(x,batch)
                
                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                x = self.lin3(x)


                return F.log_softmax(x, dim=-1)


            

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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
            loss = F.nll_loss(output, data.y.view(-1))
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

        for epoch in range(1, 5601):
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
            



# higher order
from torch_geometric.nn import GraphConv

class HO_framework:
    def __init__(self,dataset,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Net, self).__init__()
                self.conv1 = GraphConv(in_channels,30)
                self.conv2 = GraphConv(30,30)

                self.lin1 = Linear(30, 30)
                self.lin2 = Linear(30, out_channels)

            def forward(self, x, edge_index, batch,edge_weight=None):
                x = self.conv1(x, edge_index,edge_weight)
                x = x.relu()
                x = self.conv2(x, edge_index,edge_weight)
                x = x.relu()

                x = global_mean_pool(x, batch)

                x = F.relu(self.lin1(x))
                x = self.lin2(x)

                return F.log_softmax(x, dim=-1)
     

        self.model = Net(10,self.dataset.num_classes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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


           
##### DIFFPOOL
from math import ceil
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj
import numpy as np


class Diffpool_framework:
    def __init__(self,dataset,max_nodes,device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.max_nodes = max_nodes
        self.dataset = dataset
        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y,random_state=10)

        self.train_loader = DenseDataLoader(self.dataset[self.train_idx],batch_size=64)
        self.test_loader = DenseDataLoader(self.dataset[self.test_idx],batch_size=64)

        class GNN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels,lin=True):
                super().__init__()

                self.conv1 = DenseGCNConv(in_channels, hidden_channels)
                self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
                self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv3 = DenseGCNConv(hidden_channels, out_channels)
                self.bn3 = torch.nn.BatchNorm1d(out_channels)
                if lin is True:
                    self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                            out_channels)
                else:
                    self.lin = None

            def bn(self, i, x):
                batch_size, num_nodes, num_channels = x.size()

                x = x.view(-1, num_channels)
                x = getattr(self, f'bn{i}')(x)
                x = x.view(batch_size, num_nodes, num_channels)
                return x

            def forward(self, x, adj, mask=None):
                batch_size, num_nodes, in_channels = x.size()

                x0 = x
                x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
                x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
                x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

                x = torch.cat([x1, x2, x3], dim=-1)

                if self.lin is not None:
                    x = F.relu(self.lin(x))

                return x


        class Net(torch.nn.Module):
            def __init__(self,max_nodes,dataset):
                super().__init__()
                self.max_nodes = max_nodes
                num_nodes = ceil(0.25 * max_nodes)
                self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
                self.gnn1_embed = GNN(dataset.num_features, 64, 64, lin=False)

                num_nodes = ceil(0.25 * num_nodes)
                self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
                self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

                self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

                self.lin1 = torch.nn.Linear(3 * 64, 64)
                self.lin2 = torch.nn.Linear(64, dataset.num_classes)

            def forward(self, x, adj, mask=None,batch=None):

                s = self.gnn1_pool(x, adj, mask)
                x = self.gnn1_embed(x, adj, mask)

                x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

                s = self.gnn2_pool(x, adj)
                x = self.gnn2_embed(x, adj)

                x, adj, l2, e2 = dense_diff_pool(x, adj, s)

                x = self.gnn3_embed(x, adj)

                x = x.mean(dim=1)
                x = F.relu(self.lin1(x))
                x = self.lin2(x)
                return F.log_softmax(x, dim=-1)
            
        self.model = Net(self.max_nodes,self.dataset).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


    def train(self,epoch):
        self.model.train()
        loss_all = 0
        for data in self.train_loader:
            self.data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.adj.double(), data.mask)
            
            loss = F.nll_loss(output, data.y.view(-1))
            loss.backward()
            loss_all += data.y.size(0) * loss.item()
            self.optimizer.step()
        return loss_all / len(self.train_loader.dataset)


    @torch.no_grad()
    def test(self,loader):
        self.model.eval()
        correct = 0
        for data in loader:
            data = data.to(self.device)

            out = self.model(data.x, data.adj.double(), data.mask)
            loss = F.nll_loss(out,data.y.view(-1))
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        return loss,correct / len(loader.dataset)
    def iterate(self):
        for epoch in range(1, 5001):
            train_loss = self.train(epoch)
            test_loss,test_acc = self.test(self.test_loader)
            _,train_acc = self.test(self.train_loader)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f},Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
   


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):
        train_loss,train_acc = self.test(self.train_loader)
        test_loss,test_acc = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            

            
            
# pooling
from torch_geometric.nn import DenseGCNConv, GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

class Pool_framework:
    def __init__(self,dataset, max_nodes, device=None):   

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"

        self.dataset = dataset
        self.max_nodes = max_nodes

        class Net(torch.nn.Module):
            def __init__(self, in_channels, out_channels, max_nodes, hidden_channels=32):
                super().__init__()

                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv12 = GCNConv(hidden_channels, hidden_channels)
                self.conv13 = GCNConv(hidden_channels, hidden_channels)
                num_nodes = ceil(0.5 * max_nodes)
                self.pool1 = Linear(hidden_channels, num_nodes)

                self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
                num_nodes = ceil(0.5 * num_nodes)
                self.pool2 = Linear(hidden_channels, num_nodes)

                self.conv3 = DenseGCNConv(hidden_channels, hidden_channels)

                self.lin1 = Linear(hidden_channels*3, hidden_channels)
                self.lin2 = Linear(hidden_channels, out_channels)

            def forward(self,x, edge_index, batch,edge_weight=None,train=False):
                x = x.double()
                
                x = self.conv1(x, edge_index).relu()
                x = self.conv12(x, edge_index).relu()
                x = self.conv13(x, edge_index).relu()

                x, mask = to_dense_batch(x, batch)
                adj = to_dense_adj(edge_index, batch).double()

                s = self.pool1(x)
#                 print(s.dtype, adj.dtype, mask.dtype, x.dtype)
                x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

                x = self.conv2(x, adj).relu()
                s = self.pool2(x)

                x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

                x = self.conv3(x, adj)

                x1 = x.mean(dim=1)
                x2 = x.sum(dim=1)
                x3 = x.max(dim=1).values
                x = torch.cat([x1, x2, x3], dim=1)
                x = self.lin1(x).relu()
                x = self.lin2(x)
                if train:
                    return F.log_softmax(x, dim=-1) , mc1 + mc2, o1 + o2
                else:
                    return F.log_softmax(x, dim=-1)
     

        self.model = Net(self.dataset.num_features, self.dataset.num_classes, self.max_nodes).to(self.device).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001) #0.0001

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
            output , mc_loss , o_loss = self.model(data.x, data.edge_index, data.batch, train=True)
            loss = F.nll_loss(output, data.y.view(-1)) + mc_loss + o_loss
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
        for epoch in range(1, 701):
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
import copy
import tqdm
import os
import numpy as np

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from src import data_loader
from src import utils
from src import constants

class GatedGraphConv(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = torch.nn.GRUCell(in_channels, out_channels, bias=False)
    
    def forward(self, x, edge_index):
        message = self.propagate(edge_index, x=x, size=None)
        return self.gru(message, x)
    
    def message(self, x_j):
        return x_j
    
    def message_and_aggregate(self, adj_t, x):
        return torch.matmul(adj_t, x, reduce=self.aggr)

class GatedGNN(torch.nn.Module):
    def __init__(self, config):
        super(GatedGNN, self).__init__()
        self.hidden_channels = config["hidden_channels"]
        self.num_tools = config["num_tools"]
        self.emb_dropout = config["emb_dropout"]
        self.dropout = config["dropout"]
        
        if config["model_type"] == "graph":
            self.num_tools += 1 # Add one for the masked node
        
        self.combined_channels = self.hidden_channels
        
        self.embedding = torch.nn.Embedding(self.num_tools, self.hidden_channels)
        
        self.graph = GatedGraphConv(self.combined_channels, self.combined_channels)
        self.dropout_one = torch.nn.Dropout(self.dropout)
        
        self.linear_one = torch.nn.Linear(self.combined_channels, self.combined_channels, bias=False)
        self.linear_two = torch.nn.Linear(self.combined_channels, self.combined_channels, bias=True)
        self.q = torch.nn.Linear(self.combined_channels, self.combined_channels, bias=True)
        
        self.linear_transform = torch.nn.Linear(self.combined_channels * 2, self.combined_channels, bias=False)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        emb = self.get_embedding(x)
        
        h = self.graph(emb, edge_index)
        h = self.dropout_one(h)
        
        w_l, w_g = self.get_workflow_reps(h, batch)
        
        w = torch.cat([w_l, w_g], dim=1)
        w = self.linear_transform(w)
        
        logits = torch.matmul(self.embedding.weight, w.T).T
        return logits
    
    def get_embedding(self, x):
        emb = self.embedding(x[:,0].long())
        emb = torch.nn.functional.dropout2d(emb.permute(1, 0), p=self.emb_dropout, training=self.training).permute(1, 0)
        return emb
    
    def get_workflow_reps(self, h, batch):
        split_sections = list(torch.bincount(batch).cpu())
        h_split = torch.split(h, split_sections, dim=0)
        
        # Stack last elements of each batch
        w_l = torch.stack([h_split[i][-1] for i in range(len(h_split))])
        
        # Stack last elements of each batch to match size of the current split
        w_g_r = torch.cat([h_split[i][-1].repeat(len(h_split[i]), 1) for i in range(len(h_split))])
        
        q1 = self.linear_one(w_g_r)
        q2 = self.linear_two(h)
        alpha = self.q(torch.sigmoid(q1 + q2))
        a = alpha * h
        
        # Split and sum by batch
        a_split = torch.split(a, split_sections, dim=0)
        w_g = torch.stack([torch.sum(a_split[i], dim=0) for i in range(len(a_split))])
        
        return w_l, w_g
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

def get_metrics(logits, data, config, all_metrics=True, by_length=None):
    pred = torch.argmax(logits, dim=1)
    acc_by_item = pred == data.y
    acc = torch.sum(acc_by_item).item() / len(data.y)
    
    top_k = 0
    mrr_k = 0
    
    if all_metrics:
        k = config["top_k"]
        mrr_k = config["mrr_k"]
        
        graph_lengths = torch.bincount(data.batch)
        if config["model_type"] == "graph":
            graph_lengths -= 1 # Don't count the masked node
        
        top_k_preds = torch.topk(logits, k=k, dim=1)[1]
        top_k_by_item = torch.sum(top_k_preds == data.y.view(-1, 1), dim=1)
        top_k = torch.sum(top_k_by_item).item() / len(data.y)
        
        mrr_top_k_preds = torch.topk(logits, k=mrr_k, dim=1)[1]
        mrr_top_k_by_item = torch.sum(mrr_top_k_preds == data.y.view(-1, 1), dim=1)
        
        correct_index = torch.where(mrr_top_k_preds == data.y.view(-1, 1))
        mrr_k_by_item = torch.zeros(len(data.y), dtype=torch.float, device=config["device"])
        mrr_k_by_item[correct_index[0]] = 1 / (correct_index[1].float() + 1)
        mrr_k_by_item = torch.where(mrr_top_k_by_item == 0, mrr_top_k_by_item.float(), mrr_k_by_item)
        mrr_k = torch.sum(mrr_k_by_item).item() / len(data.y)
        
        for i in range(len(graph_lengths)):
            curr_len = graph_lengths[i].item()
            if curr_len not in by_length:
                by_length[curr_len] = {"acc": [], "top_k": [], "mrr_k": []}
            
            by_length[curr_len]["acc"].append(acc_by_item[i].item())
            by_length[curr_len]["top_k"].append(top_k_by_item[i].item())
            by_length[curr_len]["mrr_k"].append(mrr_k_by_item[i].item())
    
    return acc, top_k, mrr_k, by_length
        

def evaluate_model(model, loader, config, all_metrics=True, use_tqdm=True):
    model.eval()
    
    total_acc = 0
    total_top_k = 0
    total_mrr_k = 0
    
    by_length = {}
    
    loader_iter = tqdm.tqdm(loader, desc="Evaluating") if use_tqdm else loader
    for _, data in enumerate(loader_iter):
        data = data.to(config["device"])
        
        with torch.no_grad():
            logits = model(data)
        
        acc, top_k, mrr_k, by_length = get_metrics(logits, data, config, all_metrics, by_length)
        
        total_acc += acc
        total_top_k += top_k
        total_mrr_k += mrr_k
    
    if all_metrics:
        for key in list(by_length.keys()):
            for metric, values in by_length[key].items():
                by_length[key][metric] = np.mean(values)
            
        return total_acc / len(loader), total_top_k / len(loader), total_mrr_k / len(loader), by_length
    else:
        return total_acc / len(loader)

def train_model(config, use_tqdm=True):
    if config["model_type"] == "graph":
        train_dataset = data_loader.GraphDataset(root=config["model_path"], name="train_data")
        val_dataset = data_loader.GraphDataset(root=config["model_path"], name="val_data")
    else:
        train_dataset = data_loader.PathDataset(root=config["model_path"], name="train_data")
        val_dataset = data_loader.PathDataset(root=config["model_path"], name="val_data")
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = GatedGNN(config).to(config["device"])
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["l2_penalty"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    
    val_accs = []
    losses = []
    
    best_acc = 0
    best_epoch = 0
    best_model = None
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        loader_iter = tqdm.tqdm(train_loader, desc="Train Epoch {}".format(epoch)) if use_tqdm else train_loader
        for _, data in enumerate(loader_iter):
            data = data.to(config["device"])
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() / len(data.y)
        
        scheduler.step()
        
        epoch_loss /= len(train_loader)
        val_acc = evaluate_model(model, val_loader, config, all_metrics=False, use_tqdm=use_tqdm)
        print("Epoch: {}, Val Acc: {}".format(epoch, val_acc))
        
        losses.append(epoch_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
    
    return best_model, best_epoch, best_acc, val_accs, losses

def save_model(model, config):
    file_name = config["model_name"]
    path = os.path.join(config["model_path"], file_name)
    utils.make_dir_for_file(path)
    torch.save(model.state_dict(), path)

def load_model(config):
    file_name = config["model_name"]
    path = os.path.join(config["model_path"], file_name)
    model = GatedGNN(config).to(config["device"])
    model.load_state_dict(torch.load(path))
    return model
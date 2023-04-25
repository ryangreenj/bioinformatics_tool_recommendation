import torch
import torch_geometric
import pickle
import numpy as np
import pandas as pd
import os

from src import utils
from src import constants

class GraphDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.file_name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.file_name}.pickle"]

    @property
    def processed_file_names(self):
        return [f"{self.file_name}.pt"]

    def download(self):
        pass

    def process(self):
        info = utils.load_json(os.path.join(self.root, "info.json"))
        edam_ont_annots = utils.load_json(constants.EDAM_ONTOLOGY_PROCESSED)

        topic_size = len(edam_ont_annots["topic"])
        data_size = len(edam_ont_annots["data"])
        format_size = len(edam_ont_annots["format"])
        operation_size = len(edam_ont_annots["operation"])
        
        data_file = f"{self.raw_dir}/{self.raw_file_names[0]}"
        
        # Load prepared data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        data_list = []
        
        # Convert to torch_geometric.data.Data
        for workflow_entry in data:
            for graph in workflow_entry["partial_graphs"]:
                G = graph["graph"]
                
                x = []
                y = graph["y"]
                
                # Build graph from node features and edges
                next_id = 0
                tool_id_to_graph_id = {}
                last_nodes = [] # Nodes with no outgoing edges, i.e. last nodes in a workflow before the masked node
                for node_id in G.nodes():
                    node = G.nodes[node_id]
                    tool_id_to_graph_id[node["tool_id"]] = next_id
                    node_features = [node["tool_id"]]
                    
                    # EDAM ontology vectors
                    topic_vec = np.zeros(topic_size)
                    for topic in node["topic"]:
                        topic_vec[topic] = 1
                    node_features.extend(topic_vec)

                    data_vec = np.zeros(data_size)
                    for data in node["data"]:
                        data_vec[data] = 1
                    node_features.extend(data_vec)

                    format_vec = np.zeros(format_size)
                    for format in node["format"]:
                        format_vec[format] = 1
                    node_features.extend(format_vec)

                    operation_vec = np.zeros(operation_size)
                    for operation in node["operation"]:
                        operation_vec[operation] = 1
                    node_features.extend(operation_vec)
                    
                    node_features.extend(node["embedding"])
                    x.append(node_features)
                    
                    if G.out_degree(node_id) == 0:
                        last_nodes.append(next_id)
                    next_id += 1
                
                senders, receivers = [], []
                
                for edge in G.edges():
                    senders.append(tool_id_to_graph_id[G.nodes[edge[0]]["tool_id"]])
                    receivers.append(tool_id_to_graph_id[G.nodes[edge[1]]["tool_id"]])
                
                # Masked node receives a dummy feature vector
                x_masked = [info["num_tools"]]
                x_masked.extend(np.zeros(info["embedding_size"] + topic_size + data_size + format_size + operation_size))
                x.append(x_masked)
                
                # Add edges from last nodes to masked node
                for last_node in last_nodes:
                    senders.append(last_node)
                    receivers.append(next_id)
                
                x = torch.tensor(x, dtype=torch.float)
                edge_index = torch.tensor([senders, receivers], dtype=torch.long)
                y = torch.tensor(y, dtype=torch.long)
                data_list.append(torch_geometric.data.Data(x=x, edge_index=edge_index, y=y))
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PathDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.file_name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.file_name}.pickle"]

    @property
    def processed_file_names(self):
        return [f"{self.file_name}.pt"]

    def download(self):
        pass

    def process(self):
        data_file = f"{self.raw_dir}/{self.raw_file_names[0]}"
        
        # Load prepared data from pickle file
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        data_list = []
        
        # Convert to torch_geometric.data.Data
        for workflow_entry in data:
            for path in workflow_entry["partial_paths"]:
                x = path[:-1]
                y = path[-1][0] # Last step in path is ground truth
                
                steps = [node[0] for node in x]
                id_to_embedding = {node[0]: node[1] for node in x}
                indices, x = pd.factorize(steps)
                
                senders, receivers = indices[:-1], indices[1:]
                
                # Associate embedding with tool instance in sequence
                full_x = []
                for item in x:
                    seq = [item]
                    seq.extend(id_to_embedding[item])
                    full_x.append(seq)
                
                full_x = torch.tensor(full_x, dtype=torch.float)
                edge_index = torch.tensor([senders, receivers], dtype=torch.long)
                y = torch.tensor(y, dtype=torch.long)
                data_list.append(torch_geometric.data.Data(x=full_x, edge_index=edge_index, y=y))
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def add_data_config(config):
    # Load info.json to get number of tools and embedding size used for model and other evaluation
    info = utils.load_json(os.path.join(config["model_path"], "info.json"))
    config["num_tools"] = info["num_tools"]
    config["description_size"] = info["embedding_size"]
    return config
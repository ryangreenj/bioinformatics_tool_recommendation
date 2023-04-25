import networkx as nx
import random
import os
import tqdm

from src import constants
from src import utils

def filter_workflows(workflows):
    # Filter out workflows that are too short or too long
    return [workflow for workflow in workflows if len(workflow["nodes"]) >= constants.MIN_WORKFLOW_LENGTH and len(workflow["nodes"]) <= constants.MAX_WORKFLOW_LENGTH]

def build_tool_list(workflows, toolbox):
    tool_name_to_id = {}
    tool_id_to_type = {}
    next_id = 0
    
    # Associate each tool with a unique ID, and ensure that all tools are in the toolbox
    for workflow in tqdm.tqdm(workflows, desc="Building tool list"):
        for node in workflow["nodes"]:
            id = utils.get_node_id(node)
            t = utils.get_node_type(node)
            
            if id not in toolbox:
                raise Exception(f"Tool {id} not found in toolbox. Please process the toolbox first.")
            
            if id not in tool_name_to_id:
                tool_name_to_id[id] = next_id
                tool_id_to_type[next_id] = t
                next_id += 1
    
    return tool_name_to_id, tool_id_to_type

def flatten_to_type(tool_name_to_id, tool_id_to_type, types=["tool"]):
    # Remove nodes that are not of the specified types, generally all but tools
    tool_name_to_id_flattened = {}
    tool_id_to_type_flattened = {}
    next_id = 0
    
    # Also flatten the associated dictionaries
    for tool_name, tool_id in tqdm.tqdm(tool_name_to_id.items(), desc="Flattening tool list"):
        if tool_id_to_type[tool_id] in types:
            tool_name_to_id_flattened[tool_name] = next_id
            tool_id_to_type_flattened[next_id] = tool_id_to_type[tool_id]
            next_id += 1
    
    return tool_name_to_id_flattened, tool_id_to_type_flattened

def build_workflow_dag(workflow, tool_name_to_id, toolbox, types=["tool"]):
    G = nx.DiGraph()
    
    # Map nodes with tool IDs and description embeddings
    node_index = 0
    for node in workflow["nodes"]:
        tool_type = utils.get_node_type(node)
        if tool_type not in types:
            node_index += 1
            continue
        
        tool_name = utils.get_node_id(node)
        G.add_node(node_index, tool_id=tool_name_to_id[tool_name], tool_type=tool_type, embedding=toolbox[tool_name]["embedding"])
        node_index += 1
    
    # Map edges between nodes
    for from_node in workflow["edges"]:
        from_node_int = int(from_node) # JSON keys are strings
        for to_node in workflow["edges"][from_node]:
            to_node_string = str(to_node) # JSON values are ints
            if utils.get_node_type(workflow["nodes"][from_node_int]) not in types:
                continue
            if utils.get_node_type(workflow["nodes"][to_node]) not in types:
                # Check if bypass edge should be added
                for to_node2 in workflow["edges"][to_node_string]:
                    to_node2
                    if utils.get_node_type(workflow["nodes"][to_node2]) not in types:
                        continue
                    G.add_edge(from_node_int, to_node2)
            else:
                G.add_edge(from_node_int, to_node)
    
    return G

def build_unique_workflow_dags(workflows, tool_name_to_id, toolbox, types=["tool"]):
    # Use graph hash to filter out duplicates
    graph_hashes = []
    unique_graph_dicts = []
    
    for workflow in tqdm.tqdm(workflows, desc="Building workflow DAGs"):
        G = build_workflow_dag(workflow, tool_name_to_id, toolbox, types)
        graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="tool_id") # Strong guarantee of uniqueness between non-isomorphic graphs
        
        # Only keep unique graphs
        if graph_hash not in graph_hashes:
            graph_hashes.append(graph_hash)
            graph_dict = { "graph": G }
            
            # Other information that can help inspection of results, not present in EuGalaxy dataset
            if "name" in workflow:
                graph_dict["name"] = workflow["name"]
            if "server" in workflow:
                graph_dict["server"] = workflow["server"]
            if "workflow_id" in workflow:
                graph_dict["workflow_id"] = workflow["workflow_id"]
            
            unique_graph_dicts.append(graph_dict)
    
    return unique_graph_dicts

def get_all_dag_paths(G):
    # Get all linear sequences in a DAG from all sources to all sinks
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    
    paths = []
    for source in sources:
        for sink in sinks:
            paths.extend(nx.all_simple_paths(G, source, sink))
    
    return paths

def build_workflow_paths_sequence(workflow_dict):
    # TODO: EDAM data not currently supported for path
    G = workflow_dict["graph"]
    paths = get_all_dag_paths(G)
    path_sequences = []
    for path in paths:
        path_sequence = []
        
        # Skip paths that are too short to have an input and ground truth
        if len(path) < 2:
            continue
        
        for node in path:
            path_sequence.append((G.nodes[node]["tool_id"], G.nodes[node]["embedding"]))
        
        path_sequences.append(path_sequence)
    
    return path_sequences

def split_data(workflow_dicts):
    # Split workflows randomly into train, test, val splits as specified in constants
    seed = random.randint(0, 1000000)
    workflow_ids = list(range(len(workflow_dicts)))
    random.Random(seed).shuffle(workflow_ids)
    
    train_ids = workflow_ids[:int(len(workflow_ids) * constants.TRAIN_SPLIT)]
    val_ids = workflow_ids[int(len(workflow_ids) * constants.TRAIN_SPLIT):int(len(workflow_ids) * (constants.TRAIN_SPLIT + constants.VAL_SPLIT))]
    test_ids = workflow_ids[int(len(workflow_ids) * (constants.TRAIN_SPLIT + constants.VAL_SPLIT)):]
    
    train_data = [workflow_dicts[i] for i in train_ids]
    val_data = [workflow_dicts[i] for i in val_ids]
    test_data = [workflow_dicts[i] for i in test_ids]
    
    return train_data, val_data, test_data

def get_partial_graphs(G):
    # Get all sub-graphs for a given graph, where each sub-graph is the reverse DFS tree of a node
    partial_graphs = []
    for node in G.nodes:
        # Use reverse DFS to get partial graphs
        if G.in_degree(node) == 0:
            continue
        
        reverse_dfs_tree = nx.dfs_tree(G.reverse(), node).reverse()
        reverse_dfs_tree.add_nodes_from((n, G.nodes[n]) for n in reverse_dfs_tree.nodes)
        reverse_dfs_tree.remove_node(node) # This is the ground truth, which will become masked node at "recommendation position".
        
        partial_graphs.append({ "graph": reverse_dfs_tree, "y": G.nodes[node]["tool_id"] })
    
    return partial_graphs

def get_partial_paths(path_sequences):
    # Get all partial paths for a given path sequence, where each partial path is a sub-sequence of the path length 2 to n
    partial_paths = []
    unique_paths = []
    for path in path_sequences:
        for i in range(2, (len(path) + 1)):
            path_tuple = tuple([x[0] for x in path[:i]])
            if path_tuple not in unique_paths:
                unique_paths.append(path_tuple)
                partial_paths.append(path[:i])
    
    return partial_paths

def prepare(workflows_path, data_output_path):
    # Driver function to prepare data for models
    
    # Load workflows and toolbox
    print("Preparing data for models...")
    workflows = utils.load_json(workflows_path)
    toolbox, embedding_size = utils.load_toolbox(include_embeddings=True, include_edam=True, edam_embeddings=True)
    
    # Filter workflows and build tool list
    print("Length of workflows before filtering: {}".format(len(workflows)))
    workflows = filter_workflows(workflows)
    print("Length of workflows after filtering: {}".format(len(workflows)))
    tool_name_to_id, tool_id_to_type = build_tool_list(workflows, toolbox)
    tool_name_to_id, tool_id_to_type = flatten_to_type(tool_name_to_id, tool_id_to_type)
    
    num_tools = len(tool_name_to_id)
    
    # Build unique workflow DAGs
    workflow_dicts = build_unique_workflow_dags(workflows, tool_name_to_id, toolbox)
    
    # Build paths for each workflow DAG
    for workflow_dict in workflow_dicts:
        workflow_dict["path_sequences"] = build_workflow_paths_sequence(workflow_dict)
    
    # Split data into train, val, test
    train_data, val_data, test_data = split_data(workflow_dicts)
    
    # Build partial graphs and paths for each workflow DAG
    for data in [train_data, val_data, test_data]:
        for workflow_dict in tqdm.tqdm(data, desc="Building partial graphs and paths"):
            workflow_dict["partial_graphs"] = get_partial_graphs(workflow_dict["graph"])
            workflow_dict["partial_paths"] = get_partial_paths(workflow_dict["path_sequences"])
    
    # Save data
    raw_output_path = os.path.join(data_output_path, "raw")
    print("Saving data...")
    utils.dump_pickle(train_data, os.path.join(raw_output_path, "train_data.pickle"))
    utils.dump_pickle(val_data, os.path.join(raw_output_path, "val_data.pickle"))
    utils.dump_pickle(test_data, os.path.join(raw_output_path, "test_data.pickle"))
    
    info = { "num_tools": num_tools, "tool_name_to_id": tool_name_to_id, "embedding_size": embedding_size }
    utils.dump_json(info, os.path.join(data_output_path, "info.json"))
    
    print("Done processing.")
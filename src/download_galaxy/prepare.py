import os
import tqdm

from src import constants
from src import utils

def process_subworkflow(subworkflow, nodes, edges, id_map, next_id):
    sub_nodes = []
    sub_edges = {}

    sub_id_map = {}
    sub_next_id = 0

    for step in list(subworkflow["subworkflow"]["steps"]):
        step = subworkflow["subworkflow"]["steps"][step]
        sub_nodes, sub_edges, sub_id_map, sub_next_id = process_step(step, sub_nodes, sub_edges, sub_id_map, sub_next_id)
    
    nodes.extend(sub_nodes)

    for from_sub_edge, to_sub_edges in sub_edges.items():
        from_sub_edge += next_id
        for to_sub_edge in to_sub_edges:
            to_sub_edge += next_id
            if from_sub_edge not in edges:
                edges[from_sub_edge] = [to_sub_edge]
            else:
                edges[from_sub_edge].append(to_sub_edge)
    
    id_map[subworkflow["id"]] = next_id + sub_next_id - 1

    for in_connection in list(subworkflow["input_connections"]):
        in_connection = subworkflow["input_connections"][in_connection]

        if type(in_connection) != list:
            in_connection = [in_connection]
        for connection in in_connection:
            from_id = id_map[connection["id"]]
            to_id = connection["input_subworkflow_step_id"] + next_id
            if from_id not in edges:
                edges[from_id] = [to_id]
            else:
                edges[from_id].append(to_id)
    
    next_id += sub_next_id

    return nodes, edges, id_map, next_id

def process_step(step, nodes, edges, id_map, next_id):
    curr_step = {}

    this_id = next_id
    
    # Extract relevant information from step to build node
    curr_step["type"] = step["type"]
    if "tool_state" in step and step["tool_state"] is not None:
        curr_step["tool_state"] = step["tool_state"]
    curr_step["annotation"] = step["annotation"]
    if "label" in step and step["label"] is not None:
        curr_step["label"] = step["label"]
    curr_step["name"] = step["name"]
    
    # Galaxy workflows can contain subworkflows (that may contain subworkflows, etc.) so we need to recursively process them
    if step["type"] == "subworkflow":
        nodes, edges, id_map, next_id = process_subworkflow(step, nodes, edges, id_map, next_id)
    else:
        # We have a single step, associate the in-edges
        for in_connection in list(step["input_connections"]):
            in_connection = step["input_connections"][in_connection]
            
            if type(in_connection) != list:
                in_connection = [in_connection]
            for connection in in_connection:
                from_id = id_map[connection["id"]]
                if from_id not in edges:
                    edges[from_id] = [this_id]
                else:
                    edges[from_id].append(this_id)
        
        # Get tool information
        if step["type"] != "data_input" and step["type"] != "data_collection_input" and step["type"] != "parameter_input":
            curr_step["tool_id"] = step["tool_id"]
            if "tool_shed_repository" in step:
                curr_step["tool_name"] = step["tool_shed_repository"]["name"]
                curr_step["tool_shed"] = step["tool_shed_repository"]["tool_shed"]
            
        nodes.append(curr_step)
        id_map[step["id"]] = this_id
        next_id += 1
    
    return nodes, edges, id_map, next_id

def process_file(file_path):
    workflow = utils.load_json(file_path)
    
    # Associate relevant information
    name = workflow["name"]
    tags = workflow["tags"]
    license = workflow["license"] if "license" in workflow else None
    split_path = os.path.split(file_path)
    workflow_id = split_path[1].split(".")[0]
    server = os.path.split(split_path[0])[1]

    nodes = []
    edges = {}

    id_map = {}
    next_id = 0
    
    # Process each step to build list of nodes and edges
    try:
        for step in list(workflow["steps"]):
            step = workflow['steps'][step]
            nodes, edges, id_map, next_id = process_step(step, nodes, edges, id_map, next_id)
    except:
        print(f"Error processing file: {file_path}  Step: {step['id']}")
        raise
    
    return {"name": name, "tags": tags, "license": license, "nodes": nodes, "edges": edges, "server": server, "workflow_id": workflow_id}

def run():
    workflows = []
    num_nodes = 0
    num_edges = 0
    print("Processing Galaxy files for AllGalaxy dataset...")
    galaxy_files = utils.get_files_of_ext(constants.RAW_GALAXY_FILES, "ga")
    for file in tqdm.tqdm(galaxy_files):
        workflow = process_file(file)
        num_nodes += len(workflow["nodes"])
        for from_edge, to_edges in workflow["edges"].items():
            num_edges += len(to_edges)
        workflows.append(workflow)
    
    utils.dump_json(workflows, os.path.join(constants.PROCESSED_WORKFLOWS_LOC, "all_galaxy.json"))
    
    print("Done processing.")
    print(f"Number of workflows: {len(workflows)}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
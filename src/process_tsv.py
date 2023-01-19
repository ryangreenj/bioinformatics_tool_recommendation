import pandas
import tqdm
import json
import os

from src import constants
from src import utils

def load_tsv(path, req_published=False, req_not_deleted=False, req_error_free=False):
    df = pandas.read_csv(path, sep='\t', header=None, names=["wf_id", "wf_updated", "in_id", "in_tool", "in_tool_v", "out_id", "out_tool", "out_tool_v", "published", "deleted", "has_errors"])
    
    # Reformat data and cleanup some errors
    df.loc[df["published"] == "t", "published"] = True
    df.loc[df["published"] == "f", "published"] = False
    df.loc[df["deleted"] == "t", "deleted"] = True
    df.loc[df["deleted"] == "f", "deleted"] = False
    df.loc[df["has_errors"] == "t", "has_errors"] = True
    df.loc[df["has_errors"] == "f", "has_errors"] = False
    df.loc[df["has_errors"].isnull(), "has_errors"] = False

    if req_published:
        df = df[df["published"] == True]
    
    if req_not_deleted:
        df = df[df["deleted"] == False]
    
    if req_error_free:
        df = df[df["has_errors"] == False]

    df.loc[df["in_tool"].isnull(), "in_tool"] = "Input dataset" # Assume all null tools are input datasets
    df = df[df['out_tool'].notna()]

    return df

def extract_workflows(df):
    workflows = []
    
    # Every row is a node, so we can group by workflow id to process a workflow at a time
    dfs = dict(tuple(df.groupby("wf_id")))
    for wf_id in tqdm.tqdm(dfs):
        next_id = 0
        id_to_new_id = dict()
        nodes = []
        edges = dict()
        
        # From nodes dataframe
        from_dfs = dict(tuple(dfs[wf_id].groupby("in_id")))
        for in_id in from_dfs:
            if in_id not in id_to_new_id:
                id_to_new_id[in_id] = next_id
                next_id += 1
                type = "tool" if from_dfs[in_id]["in_tool"].iloc[0] != "Input dataset" else "data_input"
                nodes.append({ "name": from_dfs[in_id]["in_tool"].iloc[0], "type": type })
            
            new_in_id = id_to_new_id[in_id]
            
            # To nodes dataframe
            to_dfs = dict(tuple(from_dfs[in_id].groupby("out_id")))
            for out_id in to_dfs:
                if out_id not in id_to_new_id:
                    id_to_new_id[out_id] = next_id
                    next_id += 1
                    type = "tool" if to_dfs[out_id]["out_tool"].iloc[0] != "Input dataset" else "data_input"
                    nodes.append({ "name": to_dfs[out_id]["out_tool"].iloc[0], "type": type })
                
                new_out_id = id_to_new_id[out_id]

                if new_in_id not in edges:
                    edges[new_in_id] = []
                edges[new_in_id].append(new_out_id)
        
        # Remove nodes with no edges
        used_nodes = set()
        for in_id in edges:
            used_nodes.add(in_id)
            for out_id in edges[in_id]:
                used_nodes.add(out_id)
        
        # Ensure nodes are used and in order
        new_nodes = []
        new_id_to_new_id = dict()
        new_next_id = 0
        for node_id in range(len(nodes)):
            if node_id in used_nodes:
                new_nodes.append(nodes[node_id])
                new_id_to_new_id[node_id] = new_next_id
                new_next_id += 1
        
        # Update edges to use new node ids
        new_edges = dict()
        for in_id in edges:
            new_in_id = new_id_to_new_id[in_id]
            for out_id in edges[in_id]:
                new_out_id = new_id_to_new_id[out_id]
                if new_in_id not in new_edges:
                    new_edges[new_in_id] = []
                new_edges[new_in_id].append(new_out_id)
        
        if len(new_nodes) > 1:
            workflow = { "nodes": new_nodes, "edges": new_edges }
            workflows.append(workflow)
    
    return workflows

def run():
    print("Processing TSV files for EuGalaxy dataset...")
    workflows = []
    
    tsv_files = utils.get_files_of_ext(constants.TSV_FILES, "tsv")
    for file in tsv_files:
        df = load_tsv(file, req_published=False, req_not_deleted=True, req_error_free=True)
        workflows.extend(extract_workflows(df))
    
    if len(workflows) > 0:
        utils.dump_json(workflows, os.path.join(constants.PROCESSED_WORKFLOWS_LOC, "eu_galaxy.json"))
    
    print("Done processing.")
import os
import json
import numpy as np
from glob import glob
import pickle

from src import constants

def make_dir(path):
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

def make_dir_for_file(path):
    # Create directory for file if it doesn't exist
    make_dir(os.path.dirname(path))

def dump_json(obj, path):
    # Dump object to json file
    make_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    # Load json file
    with open(path, "r") as f:
        return json.load(f)

def dump_numpy(obj, path):
    # Dump numpy array to file
    make_dir_for_file(path)
    np.save(path, obj)
    
def load_numpy(path):
    # Load numpy array from file
    return np.load(path)

def dump_pickle(obj, path):
    # Dump object to pickle file
    make_dir_for_file(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    # Load pickle file
    with open(path, "rb") as f:
        return pickle.load(f)

def get_files_of_ext(path, file_ext):
    # Get all files of a certain extension in a directory
    file_ext = file_ext.replace(".", "")
    return [y for x in os.walk(path) for y in glob(os.path.join(x[0], f"*.{file_ext}"))]

def get_guid_without_version(guid):
    # Get tool guid string without version
    return "/".join(guid.split("/")[:-1])

def get_id_from_guid(guid, has_version=False):
    # Get tool id from guid
    if has_version:
        return guid.split("/")[-2]
    return guid.split("/")[-1]

def is_guid(input):
    return "/" in input

def get_node_id(node):
    # Get node id from node dict based on the available fields
    if "tool_id" in node:
        id = node["tool_id"]
    elif "tool_name" in node:
        id = node["tool_name"]
    elif "label" in node:
        id = node["label"]
    else:
        id = node["name"]
    
    if is_guid(id):
        id = get_id_from_guid(id, has_version=True)
    return id

def get_node_type(node):
    return node["type"]

def load_toolbox(include_embeddings=True, include_edam=False, edam_embeddings=False):
    # Construct toolbox from json file and associate embeddings
    if include_edam:
        toolbox = load_json(constants.TOOL_LIST_BIOTOOLS)
    else:
        toolbox = load_json(constants.TOOL_LIST)
    if include_embeddings:
        if include_edam:
            tool_embeddings = load_numpy(constants.DESCRIPTION_EMBEDDINGS_BIOTOOLS)
        else:
            tool_embeddings = load_numpy(constants.DESCRIPTION_EMBEDDINGS)
        embedding_size = tool_embeddings.shape[1]
    
        if (len(toolbox) != len(tool_embeddings)):
            raise Exception("Toolbox and embeddings have different lengths. Please process the toolbox first.")
    
        # Map embeddings to tools
        for i, tool in enumerate(toolbox):
            toolbox[tool]["embedding"] = tool_embeddings[i]
    
        return toolbox, embedding_size
    else:
        return toolbox
import os
import json
import glob
import tqdm
import pandas as pd

from src import constants
from src import utils

def generate_aliases(name):
    aliases = set()
    aliases.add(name.lower())
    aliases.add(name.replace(' ', '').replace('_', '').replace('-', '').lower())
    no_symbols = name.replace('.', '').replace('(', '').replace(')', '').lower()
    aliases.add(no_symbols)
    aliases.add(no_symbols.replace(' ', '').replace('_', '').replace('-', ''))
    
    # Remove last char if it is a number
    if no_symbols[-1].isdigit():
        aliases.add(no_symbols[:-1])
        aliases.add(no_symbols[:-1].replace(' ', '').replace('_', '').replace('-', ''))
    
    return list(aliases)
    
def load_biotools_names_to_json():
    folder_names = os.listdir(constants.BIOTOOLS_FILES)
    name_to_json = {}
    
    for folder in tqdm.tqdm(folder_names, desc="Loading biotools data"):
        if os.path.isdir(os.path.join(constants.BIOTOOLS_FILES, folder)):
            biotools_files = glob.glob(os.path.join(constants.BIOTOOLS_FILES, folder, "*.biotools.json"))
            if len(biotools_files) > 0:
                aliases = generate_aliases(folder)
                for alias in aliases:
                    if alias not in name_to_json:
                        name_to_json[alias] = biotools_files[0]
                try:
                    with open(biotools_files[0], "r", errors="ignore") as f:
                        jsonData = json.load(f)
                except:
                    biotools_files[0] = "\\\\?\\" + biotools_files[0]
                    with open(biotools_files[0], "r", errors="ignore") as f:
                        jsonData = json.load(f)
                if "name" in jsonData:
                    aliases = generate_aliases(jsonData["name"])
                    for alias in aliases:
                        if alias not in name_to_json:
                            name_to_json[alias] = biotools_files[0]
    
    return name_to_json

def match_tools(toolbox, name_to_json):
    #matched_tools = []
    total = 0
    matched = 0
    tool_name_to_json = {}
    for tool in tqdm.tqdm(toolbox, desc="Processing toolbox"):
        
        split_by_under = tool.split("_")
        
        name_cands = ["_".join(split_by_under[:i]) for i in range(1, len(split_by_under) + 1)]
        
        # Want to try to match the others first
        hyphen_to_under = tool.replace("-", "_")
        split_by_under = hyphen_to_under.split("_")
        name_cands += ["_".join(split_by_under[:i]) for i in range(1, len(split_by_under) + 1)]
        
        for name in name_cands:
            if any(char.isdigit() for char in name):
                name = ''.join([i for i in name if not i.isdigit()])
                name_cands.append(name)
        
        # Reverse
        name_cands = name_cands[::-1]
        
        for name in name_cands:
            name = name.lower()
            if name in name_to_json:
                matched += 1
                tool_name_to_json[tool] = name_to_json[name]
                break
            
            name = name.replace(' ', '').replace('_', '').replace('-', '')
            if name in name_to_json:
                matched += 1
                tool_name_to_json[tool] = name_to_json[name]
                break
            
            name = name.replace('.', '').replace('(', '').replace(')', '')
            if name in name_to_json:
                matched += 1
                tool_name_to_json[tool] = name_to_json[name]
                break
        
        total += 1
    
    #print(matched_tools)
    print(f"Matched {matched} out of {total} ({matched/total*100:.2f}%)")
    
    return tool_name_to_json

def associate_edam_ontology(all_topic, all_data, all_format, all_operation):
    edam_ontology = pd.read_csv(constants.EDAM_ONTOLOGY_CSV)
    
    ont_annot = {"topic": {}, "data": {}, "format": {}, "operation": {}}
    
    next_topic = 0
    for topic in all_topic:
        try:
            name = edam_ontology[edam_ontology["http://data.bioontology.org/metadata/prefixIRI"] == topic]["Preferred Label"].values[0]
        except:
            print(topic)
            raise
        ont_annot["topic"][topic] = {"name": name, "id": next_topic}
        next_topic += 1
    
    next_data = 0
    for data in all_data:
        name = edam_ontology[edam_ontology["http://data.bioontology.org/metadata/prefixIRI"] == data]["Preferred Label"].values[0]
        ont_annot["data"][data] = {"name": name, "id": next_data}
        next_data += 1
    
    next_format = 0
    for format in all_format:
        name = edam_ontology[edam_ontology["http://data.bioontology.org/metadata/prefixIRI"] == format]["Preferred Label"].values[0]
        ont_annot["format"][format] = {"name": name, "id": next_format}
        next_format += 1
    
    next_operation = 0
    for operation in all_operation:
        name = edam_ontology[edam_ontology["http://data.bioontology.org/metadata/prefixIRI"] == operation]["Preferred Label"].values[0]
        ont_annot["operation"][operation] = {"name": name, "id": next_operation}
        next_operation += 1
    
    return ont_annot
    

def extract_edam_data(toolbox, tool_name_to_json):
    all_topic = set()
    all_data = set()
    all_format = set()
    all_operation = set()
    
    for tool in tqdm.tqdm(toolbox, desc="Extracting EDAM data"):
        if tool not in tool_name_to_json:
            continue
        
        with open(tool_name_to_json[tool], "r", errors="ignore") as f:
            json_data = json.load(f)
        
        if "description" in json_data:
            toolbox[tool]["biotools_description"] = json_data["description"]
        
        if "topic" in json_data:
            topics = set()
            for topic in json_data["topic"]:
                topics.add(topic["uri"].split("/")[-1])
            toolbox[tool]["biotools_topics"] = list(topics)
            all_topic.update(topics)
        
        if "function" in json_data:
            input_data = set()
            input_format = set()
            operations = set()
            output_data = set()
            output_format = set()
            
            for function in json_data["function"]:
                if "input" in function:
                    for inp in function["input"]:
                        if "data" in inp:
                            input_data.add(inp["data"]["uri"].split("/")[-1])
                        if "format" in inp:
                            for format in inp["format"]:
                                input_format.add(format["uri"].split("/")[-1])
                
                if "operation" in function:
                    for operation in function["operation"]:
                        operations.add(operation["uri"].split("/")[-1])
                
                if "output" in function:
                    for out in function["output"]:
                        if "data" in out:
                            output_data.add(out["data"]["uri"].split("/")[-1])
                        if "format" in out:
                            for format in out["format"]:
                                output_format.add(format["uri"].split("/")[-1])
            
            if len(input_data) > 0:
                toolbox[tool]["biotools_input_data"] = list(input_data)
                all_data.update(input_data)
            if len(input_format) > 0:
                toolbox[tool]["biotools_input_format"] = list(input_format)
                all_format.update(input_format)
            if len(operations) > 0:
                toolbox[tool]["biotools_operations"] = list(operations)
                all_operation.update(operations)
            if len(output_data) > 0:
                toolbox[tool]["biotools_output_data"] = list(output_data)
                all_data.update(output_data)
            if len(output_format) > 0:
                toolbox[tool]["biotools_output_format"] = list(output_format)
                all_format.update(output_format)
                
    return toolbox, associate_edam_ontology(all_topic, all_data, all_format, all_operation)

def process():
    toolbox = utils.load_toolbox(include_embeddings=False, include_edam=False)
    
    name_to_json = load_biotools_names_to_json()
    
    tool_name_to_json = match_tools(toolbox, name_to_json)
    
    toolbox, ont_annot = extract_edam_data(toolbox, tool_name_to_json)
    
    utils.dump_json(toolbox, constants.TOOL_LIST_BIOTOOLS)
    utils.dump_json(ont_annot, constants.EDAM_ONTOLOGY_PROCESSED)
import tqdm

from src import constants
from src import utils

def collect_tools_from_repos():
    repositories = utils.load_json(constants.PROCESSED_REPOSITORIES)
    
    tools = {}
    
    for repository in tqdm.tqdm(repositories.values(), desc="Retrieving descriptions"):
        if not repository["processed"]:
            continue
    
        if "tools" not in repository:
            continue
        
        # Obtain descriptions for all tools from processed repository
        for tool in repository["tools"]:
            # If there is only one tool in the repository, often use the repository description as seen from data
            if len(repository["tools"]) == 1:
                if tool["description"] == "":
                    desc = tool["name"] + ". " + repository["description"]
                else:
                    desc = tool["name"] + " " + tool["description"] + ". " + repository["description"]
            else:
                desc = tool["name"] + " " + tool["description"]
            
            desc = desc.strip()
            id = tool["id"]
            processed_id = id.replace("/", "_") # Some tools have a slash in their id, but it is not a guid at this point
            
            # Some tools have same name, take description of one with most downloads
            if processed_id in tools:
                if tools[processed_id]["downloads"] >= repository["downloads"]:
                    continue
            
            tools[processed_id] = { "description": desc, "downloads": repository["downloads"] }
    
    return tools

def add_extra_tools(tools):
    # Not all tools in workflows are present in Toolshed repositories, so we add them here
    processed_workflows = utils.get_files_of_ext(constants.PROCESSED_WORKFLOWS_LOC, "json")
    
    for file in processed_workflows:
        workflows = utils.load_json(file)
        
        for workflow in tqdm.tqdm(workflows, desc="Adding extras"):
            for node in workflow["nodes"]:
                id = utils.get_node_id(node)
                
                if id in tools:
                    continue
                
                tools[id] = { "description": node["name"], "downloads": 0 }
    
    return tools

def preprocess_descriptions(tools):
    # Remove punctuation and make all lowercase
    for id in tqdm.tqdm(tools, desc="Preprocessing descriptions"):
        desc = tools[id]["description"]
        desc = desc.replace("(", " ").replace(")", " ").replace(",", " ").replace(":", " ").replace(";", " ").replace("-", " ").replace("_", " ").replace("/", " ").replace("\\", " ").replace("  ", " ")
        tools[id]["description"] = desc.lower().strip()
    return tools

def create_tool_list():
    # Create toolbox from downloaded tool repositories
    tools = collect_tools_from_repos()
    tools = add_extra_tools(tools)
    tools = preprocess_descriptions(tools)
    utils.dump_json(tools, constants.TOOL_LIST)
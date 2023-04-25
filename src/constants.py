import os

# High-level locations
REPO_LOC = os.path.dirname(os.path.dirname(__file__))
DATA_LOC = os.path.join(REPO_LOC, "data")
OUT_LOC = os.path.join(REPO_LOC, "out")

# Data locations
WORKFLOW_SOURCES = os.path.join(DATA_LOC, "workflow_sources.txt")
TOOL_SOURCES = os.path.join(DATA_LOC, "tool_sources.txt")
RAW_GALAXY_FILES = os.path.join(DATA_LOC, "raw_galaxy_files")
TSV_FILES = os.path.join(DATA_LOC, "tsv_files")

# Processed data locations
PROCESSED_WORKFLOWS_LOC = os.path.join(OUT_LOC, "processed_workflows")

TOOLBOX_LOC = os.path.join(OUT_LOC, "toolbox")
PROCESSED_REPOSITORIES = os.path.join(TOOLBOX_LOC, "processed_repositories.json")
TOOL_LIST = os.path.join(TOOLBOX_LOC, "tool_list.json")
DESCRIPTION_EMBEDDINGS = os.path.join(TOOLBOX_LOC, "description_embeddings.npy")

# Download options
GALAXY_API_WAIT = 5
TOOLSHED_API_WAIT = 5

# Filter options
MIN_WORKFLOW_LENGTH = 2
MAX_WORKFLOW_LENGTH = 50

# Data options
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Training options
NUM_OPTIMIZE_ITERATIONS = 10

# Automatic evaluation options
HITRATE_K = 3
MRR_K = 5

# Model options and locations
MODEL_TYPE = "graph" # "graph" or "path"
MODEL_DATA = "all_galaxy" # "all_galaxy" or "eu_galaxy" are provided, can be extended to any processed workflow set
MODEL_NAME = "allgalaxy_graph" # Folder path in OUT_LOC, 'out' folder by default

# Recommendation options
TOOLS_TO_RECOMMEND = 5 # Number of tools to recommend
CANDIDATES_TO_SHOW = 10 # Number of candidate tools to show if input name could not be matched fully, model requires identical match
MATCH_INTERACTIVELY = True # If True, show candidates and ask user to select one, otherwise show candidates and exit
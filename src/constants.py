import os

REPO_LOC = os.path.dirname(os.path.dirname(__file__))
DATA_LOC = os.path.join(REPO_LOC, "data")
OUT_LOC = os.path.join(REPO_LOC, "out")


WORKFLOW_SOURCES = os.path.join(DATA_LOC, "workflow_sources.txt")
TOOL_SOURCES = os.path.join(DATA_LOC, "tool_sources.txt")
RAW_GALAXY_FILES = os.path.join(DATA_LOC, "raw_galaxy_files")
TSV_FILES = os.path.join(DATA_LOC, "tsv_files")


PROCESSED_WORKFLOWS_LOC = os.path.join(OUT_LOC, "processed_workflows")

TOOLBOX_LOC = os.path.join(OUT_LOC, "toolbox")
PROCESSED_REPOSITORIES = os.path.join(TOOLBOX_LOC, "processed_repositories.json")
TOOL_LIST = os.path.join(TOOLBOX_LOC, "tool_list.json")
DESCRIPTION_EMBEDDINGS = os.path.join(TOOLBOX_LOC, "description_embeddings.npy")

GALAXY_API_WAIT = 5
TOOLSHED_API_WAIT = 5

MIN_WORKFLOW_LENGTH = 2
MAX_WORKFLOW_LENGTH = 50

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

NUM_OPTIMIZE_ITERATIONS = 10

HITRATE_K = 3
MRR_K = 5

MODEL_TYPE = "graph" # "graph" or "path"
MODEL_VARIANT = "none" # "none", "no_nlp", or "no_attn"
MODEL_DATA = "all_galaxy" # "all_galaxy" or "eu_galaxy"
MODEL_NAME = "allgalaxy_graph"
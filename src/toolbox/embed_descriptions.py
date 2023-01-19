from sentence_transformers import SentenceTransformer

from src import constants
from src import utils

def run():
    tools = utils.load_json(constants.TOOL_LIST)
    
    print("Loading PubMedBERT model...")
    model = SentenceTransformer('pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb')
    
    descriptions = [tools[id]["description"] for id in tools]
    
    print("Embedding tool descriptions...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    utils.dump_numpy(embeddings, constants.DESCRIPTION_EMBEDDINGS)
    
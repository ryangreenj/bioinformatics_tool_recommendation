from sentence_transformers import SentenceTransformer
import tqdm
import numpy as np

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


def embed_biotools():
    # Do sum pooling of embeddings of all available entries
    tools = utils.load_toolbox(include_embeddings=False, include_edam=True)
    edam_ontology = utils.load_json(constants.EDAM_ONTOLOGY_PROCESSED)

    all_sentences = {}
    for tool in tools:
        to_embed = []
        to_embed.append(tools[tool]["description"])
        if "biotools_description" in tools[tool]:
            to_embed.append(tools[tool]["biotools_description"])

        input = ""
        if "biotools_input_format" in tools[tool]:
            input += ", ".join([edam_ontology["format"][entry]["name"] for entry in tools[tool]["biotools_input_format"]])
            input += ". "

        if "biotools_input_data" in tools[tool]:
            input += ", ".join([edam_ontology["data"][entry]["name"] for entry in tools[tool]["biotools_input_data"]])
            input += ". "

        if input != "":
            input = "Input: " + input.strip()
            to_embed.append(input)

        output = ""
        if "biotools_output_format" in tools[tool]:
            output += ", ".join([edam_ontology["format"][entry]["name"] for entry in tools[tool]["biotools_output_format"]])
            output += ". "

        if "biotools_output_data" in tools[tool]:
            output += ", ".join([edam_ontology["data"][entry]["name"] for entry in tools[tool]["biotools_output_data"]])
            output += ". "

        if output != "":
            output = "Output: " + output.strip()
            to_embed.append(output)

        topics = ""
        if "biotools_topics" in tools[tool]:
            topics += ", ".join([edam_ontology["topic"][entry]["name"] for entry in tools[tool]["biotools_topics"]])
            topics += ". "

        if topics != "":
            topics = "Topics: " + topics.strip()
            to_embed.append(topics)

        operations = ""
        if "biotools_operations" in tools[tool]:
            operations += ", ".join([edam_ontology["operation"][entry]["name"] for entry in tools[tool]["biotools_operations"]])
            operations += ". "

        if operations != "":
            operations = "Operations: " + operations.strip()
            to_embed.append(operations)

        all_sentences[tool] = to_embed

    print("Loading PubMedBERT model...")
    model = SentenceTransformer('pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb')

    print("Embedding tool descriptions...")
    embeddings = []

    for tool in tqdm.tqdm(all_sentences):
        tool_embeddings = model.encode(all_sentences[tool], show_progress_bar=False)
        embeddings.append(np.sum(tool_embeddings, axis=0))

    embeddings = np.array(embeddings)
    utils.dump_numpy(embeddings, constants.DESCRIPTION_EMBEDDINGS_BIOTOOLS)
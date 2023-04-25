import os
import sys
sys.path.append(os.getcwd())

import torch_geometric

from src import constants
from src import process_tsv
from src import utils
from src import prepare_data
from src import data_loader
from src import optimize_hyper
from src import model
from src import get_recommendation

import download_galaxy.download
import download_galaxy.prepare

import toolbox.download_repositories
import toolbox.process_toolbox
import toolbox.embed_descriptions

def download_workflow_data():
    ### DOWNLOAD AND PROCESS WORKFLOW DATA
    download_galaxy.download.run()
    download_galaxy.prepare.run()
    process_tsv.run() # For EuGalaxy tsv files

def download_toolbox_data():
    ## DOWNLOAD AND EMBED TOOLBOX NAMES AND DESCRIPTIONS
    toolbox.download_repositories.run()
    toolbox.process_toolbox.create_tool_list()
    toolbox.embed_descriptions.run()

def prepare_data_splits(model_base_loc, optimize_base_loc):
    ### PREPARE DATA AND SPLITS
    prepare_data.prepare(os.path.join(constants.PROCESSED_WORKFLOWS_LOC, "{}.json".format(constants.MODEL_DATA)), model_base_loc)
    prepare_data.prepare(os.path.join(constants.PROCESSED_WORKFLOWS_LOC, "{}.json".format(constants.MODEL_DATA)), optimize_base_loc)

def optimize_hyperparameters(base_config):
    ## OPTIMIZE HYPERPARAMETERS OVER 10 ITERATIONS
    ranges = {
        "hidden_channels": [16, 64],
        "learning_rate": [0.0001, 0.01],
        "l2_penalty": [0.00001, 0.01],
        "step_size": [10, 30],
        "emb_dropout": [0.0, 0.5],
        "dropout": [0.0, 0.5],
        "batch_size": [32, 128],
        "epochs": [50, 100],
    }
    
    num_evals = constants.NUM_OPTIMIZE_ITERATIONS
    optimize_hyper.optimize(base_config, ranges, num_evals, "best_hyperparameters.json")

def train_model(base_config, model_base_loc, optimize_base_loc):
    ## TRAIN ONE MODEL
    config = base_config.copy()
    best_params = utils.load_json(os.path.join(optimize_base_loc, "best_hyperparameters.json"))
    
    for key, value in best_params.items():
        config[key] = value
    
    config["hidden_channels"] = int(config["hidden_channels"])
    config["step_size"] = int(config["step_size"])
    config["batch_size"] = int(config["batch_size"])
    config["epochs"] = int(config["epochs"])
    config["model_path"] = model_base_loc
    config = data_loader.add_data_config(config)
    
    best_model, best_epoch, best_acc, val_accs, losses = model.train_model(config)
    print("Best Epoch: {}, Best Val Acc: {}".format(best_epoch, best_acc))
    model.save_model(best_model, config)

def test_model(base_config, model_base_loc, optimize_base_loc):
    # TEST ONE MODEL
    config = base_config.copy()
    best_params = utils.load_json(os.path.join(optimize_base_loc, "best_hyperparameters.json"))
    
    for key, value in best_params.items():
        config[key] = value
    
    config["hidden_channels"] = int(config["hidden_channels"])
    config["step_size"] = int(config["step_size"])
    config["batch_size"] = int(config["batch_size"])
    config["epochs"] = int(config["epochs"])
    config["model_path"] = model_base_loc
    config = data_loader.add_data_config(config)
    
    if config["model_type"] == "graph":
        test_dataset = data_loader.GraphDataset(root=config["model_path"], name="test_data")
    else:
        test_dataset = data_loader.PathDataset(root=config["model_path"], name="test_data")
    
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    best_model = model.load_model(config)
    acc, top_k, mrr_k, by_length = model.evaluate_model(best_model, test_loader, config)
    print("HR1: {}, HR3: {}, MRR: {}".format(acc*100, top_k*100, mrr_k*100))

def main():
    model_base_loc = os.path.join(constants.OUT_LOC, constants.MODEL_NAME)
    optimize_base_loc = os.path.join(constants.OUT_LOC, "{}_optimize".format(constants.MODEL_NAME))
    
    base_config = {
        "device": "cuda",
        "model_type": constants.MODEL_TYPE,
        "hidden_channels": 32,
        "learning_rate": 0.001,
        "l2_penalty": 0.00001,
        "step_size": 30,
        "weight_decay": 0.1,
        "emb_dropout": 0.0,
        "dropout": 0.0,
        "epochs": 100,
        "batch_size": 100,
        "model_path": optimize_base_loc,
        "model_name": "model.pt",
        "top_k": constants.HITRATE_K,
        "mrr_k": constants.MRR_K,
    }
    
    if (sys.argv[1] == "download"):
        download_workflow_data()
        download_toolbox_data()
        return
    
    if (sys.argv[1] == "prepare"):
        prepare_data_splits(model_base_loc, optimize_base_loc)
        return
    
    if (sys.argv[1] == "optimize"):
        optimize_hyperparameters(base_config)
        return
    
    if (sys.argv[1] == "train"):
        train_model(base_config, model_base_loc, optimize_base_loc)
        return
    
    if (sys.argv[1] == "test"):
        test_model(base_config, model_base_loc, optimize_base_loc)
        return
    
    if (sys.argv[1] == "get_rec"):
        if (len(sys.argv) < 3):
            print("Please provide a space-separated list of tools as input")
            return
        
        input_sequence = sys.argv[2:]
        get_recommendation(base_config, model_base_loc, optimize_base_loc, input_sequence)
        return
    
    print ("Invalid argument. Please use one of the following:")
    print("download - Download and process workflow and toolbox data")
    print("prepare - Prepare data splits for training and testing the model based on the specified options")
    print("optimize - Optimize hyperparameters over {} iterations".format(constants.NUM_OPTIMIZE_ITERATIONS))
    print("train - Train a model using the optimized hyperparameters and specified options")
    print("test - Test a previously trained model")
    print("get_rec - Get recommendations for a given workflow, provide the input as a space-separated list of tools")

if __name__ == "__main__":
    main()
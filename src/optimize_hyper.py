import hyperopt
import numpy as np
import os

from src import utils
from src import data_loader
from src import constants

if (constants.MODEL_VARIANT == "no_nlp"):
    from src import model_no_nlp as model
elif (constants.MODEL_VARIANT == "no_attn"):
    from src import model_no_attn as model
else:
    from src import model as model

i = 0

def optimize(config, ranges, num_evals, output_name="best_hyperparameters.json"):
    params = config.copy()
    
    # Define hyperparaments and ranges
    params["hidden_channels"] = hyperopt.hp.quniform("hidden_channels", ranges["hidden_channels"][0], ranges["hidden_channels"][1], 1)
    params["learning_rate"] = hyperopt.hp.loguniform("learning_rate", np.log(ranges["learning_rate"][0]), np.log(ranges["learning_rate"][1]))
    params["l2_penalty"] = hyperopt.hp.loguniform("l2_penalty", np.log(ranges["l2_penalty"][0]), np.log(ranges["l2_penalty"][1]))
    params["step_size"] = hyperopt.hp.quniform("step_size", ranges["step_size"][0], ranges["step_size"][1], 1)
    params["emb_dropout"] = hyperopt.hp.uniform("emb_dropout", ranges["emb_dropout"][0], ranges["emb_dropout"][1])
    params["dropout"] = hyperopt.hp.uniform("dropout", ranges["dropout"][0], ranges["dropout"][1])
    params["batch_size"] = hyperopt.hp.quniform("batch_size", ranges["batch_size"][0], ranges["batch_size"][1], 1)
    params["epochs"] = hyperopt.hp.quniform("epochs", ranges["epochs"][0], ranges["epochs"][1], 1)
    
    def objective(params):
        global i
        i += 1
        model_config = params.copy()
        
        # Convert some parameters back to int
        model_config["hidden_channels"] = int(model_config["hidden_channels"])
        model_config["step_size"] = int(model_config["step_size"])
        model_config["batch_size"] = int(model_config["batch_size"])
        model_config["epochs"] = int(model_config["epochs"])
        
        print("Hyperparameter Optimize Iteration: {}".format(i))
        print("Hyperparameters: {}".format(model_config))
        
        model_config = data_loader.add_data_config(model_config)
        best_model, best_epoch, best_acc, val_accs, losses = model.train_model(model_config, use_tqdm=False)
        return min(losses)
    
    trials = hyperopt.Trials()
    best = hyperopt.fmin(objective, params, algo=hyperopt.tpe.suggest, max_evals=num_evals, trials=trials)
    utils.dump_json(best, os.path.join(config["model_path"], output_name))
    
    return best
    
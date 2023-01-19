# BTR: Bioinformatics Tool Recommendation System

Implementation of the Bioinformatics Tool Recommendation system for the paper in review.

Programing language: Python

Operating system: Any

License: MIT License

## Installation and Setup
### Prerequisites:
* Python 3.7+


### Download repository and install requirements
*    `git clone https://github.com/ryangreenj/bioinformatics_tool_recommendation.git`
*    `cd ./bioinformatics_tool_recommendation`
*    `pip install -r requirements.txt`

### Preparing data and configuration
AllGalaxy: The repository is configured to create the AllGalaxy dataset by default. Any additional Galaxy servers to download workflows from can be added to ./data/workflow_sources.txt. Additionaly Toolsheds to pull from can be added to ./data/tool_sources.txt.

EuGalaxy: Place any .tsv files containing workflow connections in the format of [Kumar et al. (2021)](https://doi.org/10.1093/gigascience/giaa152), which will become the "EuGalaxy" dataset.

For evaluation we use [workflow-connection-20-04.tsv](https://github.com/anuprulez/galaxy_tool_recommendation/blob/master/data/worflow-connection-20-04.tsv) downloaded from [their public GitHub repository](https://github.com/anuprulez/galaxy_tool_recommendation).

Model parameters can be customized by modifying the corresponding values on `./src/constants.py`

To download and process AllGalaxy data and EuGalaxy data (if applicable), along with toolbox information, run

    python ./src/main.py download

This will take some time depending on the number of seconds to wait between API calls specified in `./src/constants.py` (Default 5 seconds).

## Training and Evaluating the model
The code is by default configured to optimize and train an instance of BTR^g^~NLP+ATTN~ on the AllGalaxy dataset. This can be configured in `./src/constants.py`

Create the data splits for the model to use:

    python ./src/main.py prepare

Optimize hyperparameters for the model to use over 10 (default, configurable in ./src/constants.py) iterations:

    python ./src/main.py optimize

Train one model using the optimized parameters:

    python ./src/main.py train

Finally, evaluate the model and display the automatic evaluation metrics with:

    python ./src/main.py test

## Future Enhancements
This repository will be enhanced to support an interactive command-line method for obtaining recommendations from a trained model.

Ideally, most options can be converted to command-line arguments rather than constants.py file.
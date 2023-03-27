# Deep Learning – Project 1

## Image classification with convolutional neural networks

Codes for the Deep Learning project 1 about CNNs on CIFAR-10.

## Codes description

Folder `models` contains the implementation of the models used in the project. The models are implemented as classes that inherit from `torch.nn.Module`.

Script `train_model.py` contains a class that you can pass model, dataset and training hyperparameters and it will train the model and return the results. It is used by `tests_script.py`.

Script `tests_script.py` can be run from terminal to train a model of a given architecture with a given set of hyperparameters and upload the results to [neptune.ai](https://neptune.ai/). It has a parameter to repeat the training multiple times and also allows for trying different augmentation methods. 

## Other files

`ensembles*` files are related to creating ensembles and evaluating their accuracy.

`kaggle*` files are related to the Kaggle [CIFAR-10](https://www.kaggle.com/c/cifar-10) competition. The predicted labels achieved a score of `0.8119`

Folder `dl_trained_models` contains some of the models used for ensembling. Bigger models (ResNet18 architecture) were not uploaded.

# Reproducibility

This section contains description of how to repeat the experiments using the provided scripts.

## Environment

Experiments were run mainly in following settings:

- Ubuntu 20.04, Python 3.10.9, PyTorch 2.0.0, CUDA 11.7

Code was also tested in following settings:

- Ubuntu 22.04, Python 3.9.16, PyTorch 2.1.0, CUDA 11.7
- Debian 10, Python 3.9.16, PyTorch 1.12.1, CPU only

`requirements.txt` and `environment.yml` files contain the list of required packages installed in the first of the environments listed above.

Using pip:

> pip install -r requirements.txt

Using conda:

> conda env create -f environment.yml

## How to run the experiments

Script `experiment_baseline.sh` runs training of baseline models with default parameters and repeates the training 5 times. It also uploads the results to [neptune.ai](https://neptune.ai/). Default parameters include learning rate = 0.003, batch size = 16 and number of epochs = 25. Run with:

> ./experiment_baseline.sh

Script `experiment_hyperparameters.sh` runs training of the pretrained ResNet18 with different hyperparameters and augmenation methods. It also uploads the results to [neptune.ai](https://neptune.ai/). Run with:

> ./experiment_hyperparameters.sh

You may directly run the `tests_script.py` script with different parameters. To see the examples, open `experiment_baseline.sh`.

There is no separate script for ensembles. They were created and evauated in the `ensembles.ipynb` Jypyter Notebook.

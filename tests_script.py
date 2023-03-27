from models import resnet18, simple_nn, simple_cnn
import torch
from train_model import TrainModelFramework
import numpy as np
import torchvision
import sys
import neptune
from neptune.types import File
import time
from functools import partial
import os
import kornia


# script description is at the bottom


def load_data(data_augumentation, data_augumentation_param):
    transform_input = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if data_augumentation is not None:
        transform = torchvision.transforms.Compose(
        [transform_input,
        add_augumentation(data_augumentation, data_augumentation_param)])
    else:
        transform = transform_input

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_input)

    return trainset, testset

def add_augumentation(method, p):
    if method == 'random_rotation':
        return torchvision.transforms.RandomRotation((-p, p))
    elif method == 'random_crop':
        return torchvision.transforms.RandomCrop(p)
    elif method == 'color_jitter':
        return torchvision.transforms.ColorJitter(brightness=p)
    elif method == 'random_mix_up':
        return kornia.augmentation.RandomMixUpV2(p=p)
    return None

def update_seed():
    new_random_seed = np.random.randint(0, 100000)
    torch.manual_seed(new_random_seed)
    np.random.seed(new_random_seed)
    return new_random_seed

def train_single_model(model, epochs, lr, batch_size, momentum, l2_regularisation, regularisation_dropout, data_augumentation, data_augumentation_param):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    trainset, testset = load_data(data_augumentation, data_augumentation_param)
    model.to(device)
    torch.compile(model)
    trainer = TrainModelFramework(model, trainset, testset, device=device)
    trainer.train(epochs=epochs, lr=lr, batch_size=batch_size, momentum=momentum, regularisation_L2=l2_regularisation, regularisation_dropout=regularisation_dropout)
    return trainer

def neptune_start():
    run = neptune.init_run(
        project="PASTE_YOUR_PROJECT_NAME_HERE",
        api_token="PASTE_YOUR_API_TOKEN_HERE",
        source_files=[],
        capture_hardware_metrics=False,
        capture_stdout=True,
        capture_stderr=True,
    )   
    return run

def neptune_upload_results(run, selected_model_name, model_trainer_trained, run_number, seed, data_augumentation, data_augumentation_param):
    if data_augumentation is None: 
        data_augumentation = ''
    if data_augumentation_param is None: 
        data_augumentation_param = ''

    run['data_augumentation'] = data_augumentation
    run['data_augumentation_param'] = data_augumentation_param

    run['model_name'] = selected_model_name
    run['parameters'] = model_trainer_trained.params
    run['parameters/seed'] = seed

    run["training_history"].upload(File.as_html(model_trainer_trained.get_history()))
    # add training history by append to generate plots in neptune
    df = model_trainer_trained.get_history()
    for column_name in df.columns:
        if column_name != 'epoch':
            for i in range(len(df[column_name])):
                run[f'training_history_plots/{column_name}'].append(value=df[column_name].iloc[i], step=int(df['epoch'].iloc[i]))

    # save model checkpoints
    datetime = time.strftime("%Y%m%d-%H%M%S")
    for model_epoch, old_model in enumerate(model_trainer_trained.models_after_each_epoch):
        dir = os.path.join('checkpoints', f'{selected_model_name}_run_{run_number+1}_{datetime}')
        filename = f'{selected_model_name}_run_{run_number+1}_{datetime}_epoch_{model_epoch}.pt'
        os.makedirs(dir, exist_ok=True)
        path_to_old_model = os.path.join(dir, filename)
        torch.save(old_model, path_to_old_model)
        run[f"model_checkpoints/epoch_{model_epoch}"].upload(path_to_old_model)

    # add metrics again, but only for the last epoch
    metrics_df = model_trainer_trained.get_history()
    run['final_metrics/train_accuracy'] = metrics_df['train_accuracy'].iloc[-1]
    run['final_metrics/test_accuracy'] = metrics_df['test_accuracy'].iloc[-1]
    run['final_metrics/train_f1'] = metrics_df['train_f1'].iloc[-1]
    run['final_metrics/test_f1'] = metrics_df['test_f1'].iloc[-1]

def neptune_stop(run):
    run.stop()

def main(args):
    models = {'resnet18_pretrained': partial(resnet18.ResNet18, pretrained=True),
              'resnet18_not_pretrained': partial(resnet18.ResNet18, pretrained=False),
              'simple_nn': simple_nn.SimpleNN,
              'simple_cnn': simple_cnn.SimpleCNN}

    selected_model = models[args[0]]
    selected_model_name = args[0]
    repetitions = int(args[1])
    epochs = int(args[2])
    lr = float(args[3])
    if len(args) > 4:
        batch_size = int(args[4])
        momentum = float(args[5])
        l2_regularisation = float(args[6])
        regularisation_dropout = float(args[7])
        data_augumentation = str(args[8])
        data_augumentation_param = float(args[9])
    else:
        batch_size = 16
        momentum = 0.0
        l2_regularisation = 0.0
        regularisation_dropout = 0.0
        data_augumentation = None
        data_augumentation_param = None

    if data_augumentation == 'None':
        data_augumentation = None

    for i in range(repetitions):
        run = neptune_start()

        seed = update_seed()

        print(f'Training model: {selected_model_name}')
        model = selected_model()
        model_trainer_trained = train_single_model(model, epochs, lr, batch_size, momentum, l2_regularisation, regularisation_dropout, data_augumentation, data_augumentation_param)
        print(f'Finished training model: {selected_model_name}')

        neptune_upload_results(run, selected_model_name, model_trainer_trained, i, seed, data_augumentation, data_augumentation_param)
        print(f'Uploaded results for model: {selected_model_name}')
        neptune_stop(run)


if __name__ == '__main__':
    main(sys.argv[1:])


    # This script can be used to run multiple repetitions of the same model with the same parameters

    # first argument is model name: resnet18_pretrained, resnet18_not_pretrained, simple_nn or simple_cnn
    # second argument is number of repetitions
    # third argument is number of epochs

    # example usage (simple): 
    # python tests_script.py resnet18_pretrained 5 25 0.003
    # python tests_script.py resnet18_not_pretrained 5 25 0.003
    # python tests_script.py simple_nn 5 25 0.003
    # python tests_script.py simple_cnn 5 25 0.003

    # example usage (long version):
    # available augmentations (aug_name): random_rotation, random_crop, color_jitter 
    # # python tests_script.py model repetitions epochs lr batch_size momentum l2_regularisation regularisation_dropout aug_name aug_param
    # python tests_script.py resnet18_pretrained 1 10 0.0001 16 0.9 0.0001 0.0 None 0.0

    # simple usage is equivalent to
    # python tests_script.py resnet18_pretrained 5 25 0.003 16 0.0 0.0 0.0 None 0.0



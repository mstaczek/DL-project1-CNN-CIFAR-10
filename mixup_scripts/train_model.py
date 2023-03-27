import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import copy
import kornia

class TrainModelFramework():
    """Train model framework
    Given a model and a dataset, trains the model. 
    Specify epochs, learning rate, momentum, batch size.
    Specify also regularisation L2 or dropout.

    Parameters for __init__: 
    model (torch.nn.Module): model to train
    trainset (torch.utils.data.DataSet): trainset
    testset (torch.utils.data.DataSet): testset
    """

    def __init__(self, model, trainset, testset, device, mixup_train=False, mixup_param=None, shuffle=True):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.device = device
        self.shuffle = shuffle

        self.mixup_train = mixup_train
        self._mixup_train_y = None
        if mixup_train:
            self.mixup = kornia.augmentation.RandomMixUpV2(p=mixup_param, data_keys=["input", "class"])
            self.shuffle = False

    def _create_dataloader(self, batch_size):
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=self.shuffle, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=self.shuffle, num_workers=2)

    def _add_dropout(self, regularisation_dropout):
        if regularisation_dropout > 0:
            self.model.add_dropout(regularisation_dropout)

    def compute_metrics(self, data_loader, train=False):
        # compute accuracy, f1 and loss
        running_loss = 0.0
        real_labels = []
        predicted_labels = []

        with torch.no_grad():
            i = 0
            for data in data_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                real_labels.append(labels.cpu())
                predicted_labels.append(predicted.cpu())
        
                if (self.mixup_train and self._mixup_train_y is None) or (self.mixup_train and not train):
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                elif self.mixup_train and self._mixup_train_y is not None and train:
                    loss = self.criterion(outputs, self._mixup_train_y[i])
                else:
                    loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                i += 1

        real_labels = torch.cat(real_labels)
        predicted_labels = torch.cat(predicted_labels)

        if self.mixup_train:
            if train and self._mixup_train_y is not None:
                pred = predicted_labels
                y = torch.vstack(self._mixup_train_y).cpu()
                accuracy = torch.mean((1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()).item()
                f1 = 0
            else:
                accuracy = accuracy_score(real_labels, predicted_labels)
                f1 = f1_score(real_labels, predicted_labels, average='macro')
        else:
            accuracy = accuracy_score(real_labels, predicted_labels)
            f1 = f1_score(real_labels, predicted_labels, average='macro')

        loss = running_loss / len(data_loader)
        return {'accuracy': accuracy, 'f1': f1, 'loss': loss}
    
    def _at_epoch_end(self):
        metrics_train = self.compute_metrics(self.trainloader, train=True)
        self.train_loss_history.append(metrics_train['loss'])
        self.train_accuracy_history.append(metrics_train['accuracy'])
        self.train_f1_history.append(metrics_train['f1'])

        metrics_test = self.compute_metrics(self.testloader)
        self.test_loss_history.append(metrics_test['loss'])
        self.test_accuracy_history.append(metrics_test['accuracy'])
        self.test_f1_history.append(metrics_test['f1'])

        self.models_after_each_epoch.append(copy.deepcopy(self.model))

    def _print_latest_epoch(self, epoch):
        print(f'Epoch {epoch}/{self.params["epochs"]}: '+
              '[train] [test] '+
              f'Loss: [{self.train_loss_history[-1]:.4f}] [{self.test_loss_history[-1]:.4f}], '+
              f'Accuracy: [{self.train_accuracy_history[-1]:.2f}%] [{self.test_accuracy_history[-1]:.2f}%], ' +
              f'F1 macro: [{self.train_f1_history[-1]:.2f}] [{self.test_f1_history[-1]:.2f}]')

    def get_history(self):
        return pd.DataFrame({'epoch':[i for i in range(self.params['epochs']+1)],
                             'train_loss':self.train_loss_history,
                             'test_loss':self.test_loss_history,
                             'train_accuracy':self.train_accuracy_history,
                             'test_accuracy':self.test_accuracy_history,
                             'train_f1':self.train_f1_history,
                             'test_f1':self.test_f1_history})

    def get_params(self):
        return self.params

    def train(self, epochs, lr, momentum=0, batch_size=16, regularisation_L2=0, regularisation_dropout=0):

        self.params = {'epochs': epochs, 'lr': lr, 'momentum': momentum, 'batch_size': batch_size,
                       'regularisation_L2': regularisation_L2, 'regularisation_dropout': regularisation_dropout}

        self._create_dataloader(batch_size)
        self._add_dropout(regularisation_dropout)

        if self.mixup_train:
            def loss_mixup(logits, y):
                criterion = torch.nn.functional.cross_entropy
                loss_a = criterion(logits.float(), y[:, 0].long(), reduction='none')
                loss_b = criterion(logits.float(), y[:, 1].long(), reduction='none')
                return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
            self.criterion = loss_mixup
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=regularisation_L2)

        self.train_loss_history = []
        self.test_loss_history = []
        self.train_f1_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.test_f1_history = []

        self.models_after_each_epoch = []

        self.model.eval()
        self._at_epoch_end()
        self._print_latest_epoch(0)

        for epoch in range(epochs):
            self._mixup_train_y = None
            
            self.model.train()
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.mixup_train:
                    inputs, labels = self.mixup(inputs, labels)
                    inputs = torch.squeeze(inputs)
                    if self._mixup_train_y is None:
                        self._mixup_train_y = [labels]
                    else:
                        self._mixup_train_y.append(labels)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            self.model.eval()
            self._at_epoch_end()
            self._print_latest_epoch(epoch + 1)

        print('Finished Training')

        return self.get_history()
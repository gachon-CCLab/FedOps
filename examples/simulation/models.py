from typing import List, Tuple
import numpy as np
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch import optim
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm 

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Define Cifar100 Model    
class CIFAR100Classifier(nn.Module):
    def __init__(self, output_size):
        super(CIFAR100Classifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)  # 100 output classes for CIFAR-100

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
def client_train():
    def train(  # pylint: disable=too-many-arguments
        model: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Train the network on the training set.

        Parameters
        ----------
        model : nn.Module
            The neural network to train.
        trainloader : DataLoader
            The DataLoader containing the data to train the network on.
        device : torch.device
            The device on which the model should be trained, either 'cpu' or 'cuda'.
        epochs : int
            The number of epochs the model should be trained for.
        learning_rate : float
            The learning rate for the SGD optimizer.
        proximal_mu : float
            Parameter for the weight of the proximal term.
        """
        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
        criterion = nn.CrossEntropyLoss()
        global_params = [val.detach().clone() for val in model.parameters()]
        model.train()
        
        for _ in range(epochs):
            total_loss = 0.0
            correct=0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
            accuracy = correct / len(trainloader)
            average_loss = total_loss / len(trainloader)  # Calculate average loss
            

        return average_loss, accuracy
    
    return train
    
def server_test():
    def global_test(
        model: nn.Module, testloader: DataLoader, device: torch.device, wandb_use=False, run=None, server_round=None
    ) -> Tuple[float, float]:
        """Evaluate the network on the entire test set.

        Parameters
        ----------
        net : nn.Module
            The neural network to test.
        testloader : DataLoader
            The DataLoader containing the data to test the network on.
        device : torch.device
            The device on which the model should be tested, either 'cpu' or 'cuda'.

        Returns
        -------
        Tuple[float, float]
            The loss and the accuracy of the input model on the given data.
        """
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        total_loss=0.0
        model.eval()
        with torch.no_grad():
            with tqdm(testloader, unit='batch') as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
        
        accuracy = correct / len(testloader)
        average_loss = total_loss / len(testloader)  # Calculate average loss
        
        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        # f1 = f1_score(all_labels, all_predictions, average='weighted')
        # Add F1 score to metrics
        # metrics = {"f1_score": f1}
        metrics=None  
        
        if wandb_use:
            run.log({"GL_Loss": average_loss}, step=server_round)
            run.log({"GL_Accuracy": accuracy},step=server_round)
            if metrics!=None:
                # Log other metrics dynamically
                for metric_name, metric_value in metrics.items():
                    run.log({'GL_'+metric_name: metric_value}, step=server_round)

        return average_loss, accuracy, metrics
            
    return global_test


def client_test():
    def local_test(
        model: nn.Module, testloader: DataLoader, device: torch.device, cid: int, run=None, round=None
    ) -> Tuple[float, float]:
        """Evaluate the network on the entire test set.

        Parameters
        ----------
        net : nn.Module
            The neural network to test.
        testloader : DataLoader
            The DataLoader containing the data to test the network on.
        device : torch.device
            The device on which the model should be tested, either 'cpu' or 'cuda'.

        Returns
        -------
        Tuple[float, float]
            The loss and the accuracy of the input model on the given data.
        """
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        total_loss=0.0
        model.eval()
        with torch.no_grad():
            with tqdm(testloader, unit='batch') as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
        
        accuracy = correct / len(testloader)
        average_loss = total_loss / len(testloader)  # Calculate average loss
        
        
        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        # f1 = f1_score(all_labels, all_predictions, average='weighted')
        # Add F1 score to metrics
        # metrics = {"f1_score": f1}
        metrics=None  
        
        return average_loss, accuracy, metrics

    
    return local_test

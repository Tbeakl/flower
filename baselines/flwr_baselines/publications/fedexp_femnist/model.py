"""CNN model architecutre, training and testing functions for MNIST."""


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Convolutional Neural Network architecture as described in Reddi 2021
    paper :

    [ADAPTIVE FEDERATED OPTIMIZATION] (https://arxiv.org/pdf/2003.00295.pdf)
    """

    def __init__(self) -> None:

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dense1 = nn.Linear(9216,128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 62)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = self.dropout1(output_tensor)
        output_tensor = torch.flatten(output_tensor, start_dim=1)  # flatten all dimensions except batch
        output_tensor = F.relu(self.dense1(output_tensor))
        output_tensor = self.dropout2(output_tensor)
        output_tensor = self.dense2(output_tensor)
        return output_tensor

def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    gradient_clipping: bool = False,
    max_norm: float = 10
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    gradient_clipping : bool
        Whether or not gradient clipping should be used
    max_norm : float
        The max_norm which is used in gradient clipping if gradient clipping is happening
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    net.train()
    for _ in range(epochs):
        net = _training_loop(net, trainloader, device, criterion, optimizer, gradient_clipping=gradient_clipping, max_norm=max_norm)


def _training_loop(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    gradient_clipping: bool = False,
    max_norm: float = 10
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    gradient_clipping : bool
        Whether or not gradient clipping should be used
    max_norm : float
        The max_norm which is used in gradient clipping if gradient clipping is happening

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=max_norm)
        optimizer.step()
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
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
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

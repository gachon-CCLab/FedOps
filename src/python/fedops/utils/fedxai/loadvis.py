"""
Classes to load, visualize etc
"""

import numpy as np 
import torch 
torch.manual_seed(0)

import torch.nn as nn 
import matplotlib.pyplot as plt
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Loader(nn.Module):
    """
    Class to load the MNIST data into PyTorch dataloaders
        1. getdata(): Function to fetch the MNIST data and store into dataloaders.
    """

    def __init__(self, PATH, batch_size):
        super().__init__()
        self.path = PATH 
        self.batch_size = batch_size
        
    def __str__(self):
        return f"Trainloader: length = {len(self.getdata()[0])}  | Testloader: length = {len(self.getdata()[1])}\nBatch Size = {self.batch_size}"

    def getdata(self):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0, ), (1, ))
        ])

        trainset = datasets.MNIST(root=self.path, train=True, download=True, transform=trans)
        testset = datasets.MNIST(root=self.path, train=False, download=True, transform=trans)



        trainloader = DataLoader(dataset=trainset, batch_size=self.batch_size, shuffle=True)

        testloader = DataLoader(dataset=testset, batch_size=self.batch_size, shuffle=True)

        return trainloader, testloader

    # Function to get a single image from the dataloaders based on its ground truth.
    def getanimage(loader, class_idx, plot=False):

        image_batch, label_batch  = next(iter(loader))

        #getting the desired from the first batch 
        for i in range(len(label_batch)):
            if label_batch[i] == class_idx:
                image_tensor = image_batch[i]

        if plot == True:
            plt.title(f"Image of the digit {class_idx}")
            plt.imshow(image_tensor[0], cmap="Greys")
            plt.show()

        return image_tensor


class Visualizer:
    """
    Class to Visualize the images in MNIST
        1. visualize(): method to enable the plotting of the digits. 
        2. plotperf(): method to plot the losses / accuracies for the model 
    """
    def __init__(self):
        pass

    def visualize(self, loader):
        test =  list(enumerate(loader))
        
        # For the first batch 
        image_batch = test[0][1][0]
        label_batch = test[0][1][1]

        image = image_batch[0].view(28, 28)
        label = label_batch[0]

        plt.title(f"Image of the digit:{label.item()}")
        plt.imshow(image.numpy(), cmap="Blues")
        plt.show()


    def plotperf(self, loss_list, num_epochs, train=True):
        if train:
            xaxis = np.array(range(num_epochs))
            yaxis = np.array(loss_list)
            plt.title(f"Loss-Epoch plot for the training cycle")
            plt.plot(xaxis, yaxis)
            plt.grid()
            plt.show()



if __name__ == "__main__":
    
    loader = Loader(PATH="/home/agastya123/PycharmProjects/DeepLearning/MNIST-Grad-CAM/data/", batch_size=128)
    trainloader, testloader = loader.getdata()
    idx, data= next(iter(trainloader))
    print(data.shape)



    



    








    

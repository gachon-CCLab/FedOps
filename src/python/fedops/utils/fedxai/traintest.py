import numpy as np 
import torch 
torch.manual_seed(0)
import torch.nn as nn 
import time 
dtype = torch.float32


class TrainTest(nn.Module):
    """
    Class to enable model training
        1. train_one_epoch(): method to carry out the training process for the model for a single epoch
        2. find_accuracy(): method to find the accuracy of the model
        3. train_all_epochs(): method to carry out the training process for all the epochs
        4. test(): takes in an already trained network as input
    """
    
    
    def __init__(self, network, num_epochs, learning_rate):
        super().__init__()
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = network 
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)


    def __str__(self):
        return f" Number of epochs = {self.num_epochs}\n Learning Rate = {self.lr}\n Loss Function = {self.loss}\n Optimizer = {self.optim}"

    
    
    def train_one_epoch(self, loader, epoch):
        self.net.train(True)
        running_loss = 0.0
        
        num_batches = len(loader)
        for idx, data in enumerate(loader):
            image_batch, label_batch = data
            
            self.optim.zero_grad()
            output = self.net(image_batch)
            loss = self.loss(output, label_batch)
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            
            # reporting for one epoch
            if (idx + 1) % num_batches == 0 :
                
                # calculating average loss for all the batches, and reporting it once per epoch.

                avg_loss = running_loss/num_batches
                print(f"Training: Epoch {epoch+1}, Batch = {idx+1}/{num_batches},   Training Loss: {avg_loss}")
        return self.net, avg_loss

        

    def train_all_epochs(self, loader):
        training_loss = [] 
        training_accuracy = []

        st = time.time()

        # training for all epochs
        for e in range(self.num_epochs):
            # training for one epoch 
            trained_net, epoch_loss = self.train_one_epoch(loader, epoch=e)
            training_loss.append(epoch_loss)

        et = time.time()

        print(f"\nTraining Done. Time taken to train the network for {self.num_epochs} epochs = {round((et-st)/60, 5)} minutes.\n")
        return trained_net, training_loss

        

    def test(self, network, loader):
        network.eval()
        batch_loss = 0.0 
        correct = 0
        
        
        num_batches = len(loader)
        with torch.no_grad():
            for idx, data in enumerate(loader):
                image_batch, label_batch = data 
                output = network(image_batch)
                loss = self.loss(output, label_batch)
                batch_loss += loss.item()
            
            avg_loss = batch_loss/num_batches


        torch.save(network.state_dict(), "mnistclassifier.pth")
        print(f"Average Test Loss = {avg_loss}")






if __name__ == "__main__":
    testnetwork = nn.Sequential(
        nn.Linear(100, 50, dtype=dtype),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.LogSoftmax()
    )

    traintest = TrainTest(network=testnetwork, num_epochs=100, learning_rate=0.001)
    
    
    











from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm 

# Define MNIST Model    
class MNISTClassifier(nn.Module):
    # To properly utilize the config file, the output_size variable must be used in __init__().
    def __init__(self, output_size):
        super(MNISTClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)  # Image size is 28x28, reduced to 14x14 and then to 7x7
        self.fc2 = nn.Linear(1000, output_size)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        """
        Train the network on the training set.
        Model must be the return value.
        """
        print("Starting training...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.train()
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    pbar.update()  # Update the progress bar for each batch

        model.to("cpu")
            
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    
    def custom_test_torch(model, test_loader, cfg):
        """
        Validate the network on the entire test set.
        Loss, accuracy values, and dictionary-type metrics variables are fixed as return values.
        """
        print("Starting evalutation...")
        
        criterion = nn.CrossEntropyLoss()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        correct = 0
        total_loss = 0.0
        all_labels = []
        all_predictions = []    
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    
                    pbar.update()  # Update the progress bar for each batch
            
        accuracy = correct / len(test_loader.dataset)
        average_loss = total_loss / len(test_loader)  # Calculate average loss
        
        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        # Add F1 score to metrics
        metrics = {"f1_score": f1}
        # metrics=None    
        
        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy, metrics
    
    return custom_test_torch

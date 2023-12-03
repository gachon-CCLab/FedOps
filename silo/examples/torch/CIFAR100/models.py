from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm 

# Define Cifar100 Model    
class CIFAR100Classifier(nn.Module):
    # To properly utilize the config file, the output_size variable must be used in __init__()
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

# Set the loss function and optimizer
def set_model_hyperparameter(model, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, criterion, optimizer, epochs, device: str = "cpu"):
        """Train the network on the training set."""
        print("Starting training...")

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

        model.to("cpu")
            
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    def custom_test_torch(model, test_loader, criterion, device: str = "cpu"):
        """Validate the network on the entire test set."""
        print("Starting evalutation...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        correct, loss = 0, 0.0
        total_loss=0.0
        # Initialize lists to store all labels and predictions
        all_labels = []
        all_predictions = []    
        
        model.to(device)  # move model to GPU if available
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
        accuracy = correct / len(test_loader.dataset)
        average_loss = total_loss / len(test_loader)  # Calculate average loss
        model.to("cpu")  # move model back to CPU
        
        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        # Add F1 score to metrics
        metrics = {"f1_score": f1}
        # metrics=None    
        
        return average_loss, accuracy, metrics
    
    return custom_test_torch

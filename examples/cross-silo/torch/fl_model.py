import torch
from torch import nn
from torch import optim
from tqdm import tqdm 

# Define the DNN model using PyTorch
class DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, num_classes)  # Change the output units to num_classes

        self.model_name = self.__class__.__name__  # Store the class name


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

# Set the loss function and optimizer
def set_model_parameter(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return criterion, optimizer

# Set the torch train & test
# torch train
def train_torch():
    
    def custom_train_torch(model, train_loader, criterion, optimizer, epochs, device: str = "cpu"):
        """Train the network on the training set."""
        print("Starting training...")

        # CUDA 지원 여부 확인
        if torch.cuda.is_available():
            device = torch.device("cuda")  # CUDA 디바이스를 사용하도록 설정
        else:
            device = torch.device("cpu")   # CPU를 사용하도록 설정
        
        model.to(device)  # move model to GPU if available

        model.train()
        for _ in range(epochs):
            with tqdm(total=len(train_loader), desc=f'Epoch {_+1}/{epochs}', unit='batch') as pbar:
                for batch_x, batch_y in train_loader:
                    inputs, labels = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

        model.to("cpu")  # move model back to CPU
        
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    def custom_test_torch(model, test_loader, criterion, steps: int = None, device: str = "cpu"):
        """Validate the network on the entire test set."""
        print("Starting evalutation...")
        
         # Check CUDA 
        if torch.cuda.is_available():
            device = torch.device("cuda")  # GPU
        else:
            device = torch.device("cpu")   # CPU
        
        model.to(device)  # move model to GPU if available
        correct, loss = 0, 0.0
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                labels_int = torch.argmax(labels, dim=1)
                loss += criterion(outputs, labels).item()
                predicted = torch.argmax(outputs, dim=1)
                # _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels_int).sum().item()
                if steps is not None and batch_idx == steps:
                    break
        accuracy = correct / len(test_loader.dataset)
        model.to("cpu")  # move model back to CPU
        
        return loss, accuracy
    
    return custom_test_torch

# Define the CNN model using Tensorflow
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
class CNN:
    def __init__(self) -> None:
        self.model_name = self.__class__.__name__  # model name
        self.METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
    
    def build_model(self):
        self.model = Sequential()

        # Convolutional Block (Conv-Conv-Pool-Dropout)
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Classifying
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=self.METRICS)

        return self.model, self.model_name
"""


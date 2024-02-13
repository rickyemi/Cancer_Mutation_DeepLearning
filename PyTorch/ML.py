import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from google.colab import drive
drive.mount('/content/drive')

# Import dataset from Google Drive
ds = pd.read_csv("/content/drive/MyDrive/Previous_Projects/mutation_data_4_ml.csv")

# data Prep for Modeling
X = ds.drop(['Target'],axis=1)
y = ds['Target'].apply(lambda x : 1 if x=='Driver' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float64)
X_test = torch.tensor(X_test.values, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training and testing data
train_data = data.TensorDataset(X_train, y_train)
test_data = data.TensorDataset(X_test, y_test)

train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)


class MutationBinaryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MutationBinaryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # activation function

    def forward(self, x):
        out = self.fc1(x) 
        out = self.relu(out) 
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out



def train_binary_model(model, train_loader, test_loader, num_epochs, learning_rate, target_accuracy, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', average_loss, epoch + 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                outputs = model(inputs)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy/test', accuracy, epoch + 1)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.3f}, Accuracy: {accuracy:.3f}%')

        if accuracy >= target_accuracy:
            print(f"Reached target accuracy of {target_accuracy:.3f}%.")
            break

    writer.flush()
    writer.close()

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
input_dim = X_train.shape[1]
hidden_dim = 16
model = LoanBinaryModel(input_dim, hidden_dim)

# Train the model with early stopping criterion
target_accuracy = 90.0  # terminate training when the accuracy reaches 90%
num_epochs = 80
learning_rate = 0.2

train_binary_model(model, train_loader, test_loader, num_epochs, learning_rate, target_accuracy, device)


# Model Performance Evaluation 

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Print classification report with zero_division parameter
    report = classification_report(y_true, y_pred, target_names=['Safe', 'Risky'], zero_division=1)
    print("\nClassification Report:")
    print(report)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(['Passenger', 'Driver']))
    plt.xticks(tick_marks, ['Passenger', 'Driver'], rotation=30)
    plt.yticks(tick_marks, ['Passenger', 'Driver'])

    # Display the values in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "green")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


# Evaluate the trained model and print the confusion matrix
evaluate_model(model, test_loader, device)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class OptimizedModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(OptimizedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def train_optimized_model(model, data_loader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data in data_loader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}')

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    input_size = len(lines[0].strip())  # Determine the input size based on the first row
    data = [list(map(lambda x: 1 if x == '+' else -1, line.strip())) for line in lines]
    
    return torch.from_numpy(np.array(data, dtype=np.float32)), input_size

def main():
    parser = argparse.ArgumentParser(description='Neural Network Training with argparse')
    parser.add_argument('file_path', type=str, help='Path to the input file')
    parser.add_argument('--hidden_size1', type=int, default=64, help='Size of the first hidden layer')
    parser.add_argument('--hidden_size2', type=int, default=32, help='Size of the second hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()

    train_data, input_size = load_data(args.file_path)
    output_size = input_size  # Assuming output size is the same as input size
    train_labels = train_data.clone()  # Use the input as the target for simplicity

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = OptimizedModel(input_size, args.hidden_size1, args.hidden_size2, output_size)
    train_optimized_model(model, train_loader, args.epochs, args.learning_rate)

if __name__ == '__main__':
    main()

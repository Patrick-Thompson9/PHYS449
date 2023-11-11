import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class BoltzmannMachine(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BoltzmannMachine, self).__init__()
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

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Determine the number of spins based on the first line
    num_spins = len(lines[0].strip())

    input_size = num_spins
    hidden_size1 = num_spins
    hidden_size2 = num_spins
    output_size = num_spins

    data = []
    for line in lines:
        # Convert '+' to 1 and '-' to -1
        line_data = [1 if s == '+' else -1 for s in line.strip()]
        data.append(line_data)

    return torch.tensor(data, dtype=torch.float32), input_size, hidden_size1, hidden_size2, output_size

def train_boltzmann_machine(model, data_loader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data in data_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def main():
    parser = argparse.ArgumentParser(description="Train a Boltzmann Machine on Ising model data.")
    parser.add_argument("file_path", type=str, help="Path to the input file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for SGD")

    args = parser.parse_args()

    data, input_size, hidden_size1, hidden_size2, output_size = load_data(args.file_path)
    model = BoltzmannMachine(input_size, hidden_size1, hidden_size2, output_size)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)

    train_boltzmann_machine(model, data_loader, args.epochs, args.learning_rate)

if __name__ == "__main__":
    main()

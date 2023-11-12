import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import os

def load_data(file_path):
    '''
    Load the data from the file and return a tensor of the data and the input size

    Input:
    - file_path: Path to the input data file

    Output:
    - data: Tensor of the data
    - input_size: Size of the input
    '''
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    input_size = len(lines[0])
    data = np.array([[1.0 if c == '+' else -1.0 for c in line] for line in lines], dtype=np.float32)

    return torch.from_numpy(data), input_size

class BoltzmannMachine(nn.Module):
    '''
    Boltzmann Machine with 2 hidden layers
    Relu activation function is used for the hidden layers
    Output layer is a linear layer
    '''
    def __init__(self, input_size):
        super(BoltzmannMachine, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, dtype=torch.float32)
        self.fc2 = nn.Linear(input_size, input_size, dtype=torch.float32)
        self.fc3 = nn.Linear(input_size, input_size, dtype=torch.float32)

    def forward(self, x):
        '''
        Forward propagation of the Boltzmann Machine

        Input:
        - x: Input data

        Output:
        - x: Output data
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_coupler_dict(self, num_spins):
        '''
        Generate the coupler dictionary of the Boltzmann Machine

        Input:
        - num_spins: Number of spins in each line

        Output:
        - coupler_dict: Coupler dictionary of the Boltzmann Machine
        '''
        coupler_dict = {}
        for i, param in enumerate(self.parameters()):
            if len(param.data.shape) == 2:  # Check if the parameter is a 2D weight matrix
                for j in range(param.data.shape[0]):
                    next_spin = (j + 1) % param.data.shape[0]
                    coupler_dict[(j, next_spin)] = param.data[j, 0].item()/abs(param.data[j, 0].item())  # Normalize the coupler to 1 or -1

        # Ensure the dictionary matches the number of spins in each line
        coupler_dict = dict(sorted(coupler_dict.items())[:num_spins])
        return coupler_dict

def train_boltzmann_machine(model, data_loader, epochs, learning_rate, output_folder, verbose):
    '''
    Train the Boltzmann Machine
    
    Input:
    - model: Boltzmann Machine
    - data_loader: DataLoader of the input data
    - epochs: Number of training epochs
    - learning_rate: Learning rate for SGD
    - output_folder: Folder name to store the output files
    - verbose: Enable verbose version of the program
    
    Output:
    - None
    '''
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_values = []  # to store the loss values for each epoch
    if verbose:
        print("Training the Boltzmann Machine...")
    for epoch in range(1, epochs + 1):

        epoch_loss = 0.0  # to store the total loss for the current epoch
        for data in data_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        loss_values.append(epoch_loss)

        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    # Plot the loss versus epochs
    if verbose:
        print("Plotting the loss vs. epochs...")
    plt.plot(range(1, epochs + 1), loss_values, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')

    # Save the plot in the 'output' folder
    if verbose:
        print("Saving the plot...") 
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'loss_vs_epochs.png')
    plt.savefig(output_path)
    plt.show()

    # Save the model in the 'output' folder
    if verbose:
        print("Saving the model...")    
    model_path = os.path.join(output_folder, 'boltzmann_model.pth')
    torch.save(model.state_dict(), model_path)

def main():
    parser = argparse.ArgumentParser(description='Train a Boltzmann Machine on Ising chain data')
    parser.add_argument('file_path', type=str, help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for SGD')
    parser.add_argument('--output_folder', type=str, default='output', help='Folder name to store the output files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose version of the program.')
    args = parser.parse_args()

    # Load data
    if args.verbose:
        print('Loading data...')
    data, input_size = load_data(args.file_path)

    # Initialize the Boltzmann Machine
    if args.verbose:
        print('Initializing the Boltzmann Machine...')
    model = BoltzmannMachine(input_size)

    # Create DataLoader
    if args.verbose:
        print('Creating DataLoader...')
    data_loader = DataLoader(data, batch_size=1, shuffle=True)

    # Train the Boltzmann machine
    train_boltzmann_machine(model, data_loader, args.epochs, args.learning_rate, args.output_folder, args.verbose)

    # Print the coupler dictionary
    if args.verbose:
        print('Creating coupler dictionary...')
    coupler_dict = model.get_coupler_dict(input_size)
    print("Predicted Coupler Dictionary:")
    print(coupler_dict)

    # Save the coupler dictionary as a text file in the 'output' folder
    coupler_path = os.path.join('output', 'coupler_dictionary.txt')
    with open(coupler_path, 'w') as coupler_file:
        coupler_file.write('{0}'.format(coupler_dict))

    if args.verbose:
        print('Done!')


if __name__ == "__main__":
    main()
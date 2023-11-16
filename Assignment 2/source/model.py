import torch
import torch.nn as nn
import torch.nn.functional as F

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
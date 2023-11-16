import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    '''
    Variational Autoencoder class
    '''
    def __init__(self, input_size, hidden_size1, hidden_size2, latent_size):
        '''
        Initialize the variational autoencoder

        Inputs:
        - input_size: The number of input features
        - hidden_size1: The number of nodes in the first hidden layer
        - hidden_size2: The number of nodes in the second hidden layer
        - latent_size: The number of nodes in the latent layer

        Outputs:
        - None
        '''
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encode1 = nn.Linear(input_size, hidden_size1)
        self.encode2 = nn.Linear(hidden_size1, hidden_size2)
        
        # mean and variance
        self.fc_mu = nn.Linear(hidden_size2, latent_size)
        self.fc_var = nn.Linear(hidden_size2, latent_size)
        
        # Decoder
        self.decode1 = nn.Linear(latent_size, hidden_size2)
        self.decode2 = nn.Linear(hidden_size2, hidden_size1)
        self.decode3 = nn.Linear(hidden_size1, input_size)

    def encode(self, x):
        '''
        Encode the input data
        
        Inputs:
        - x: The input data
        
        Outputs:
        - mean: The mean of the latent layer
        - var: The variance of the latent layer
        '''
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        mean = self.fc_mu(x)
        var = self.fc_var(x)
        return mean, var

    def decode(self, x):
        '''
        Decode the latent layer

        Inputs:
        - x: The latent layer

        Outputs:
        - x: The reconstructed data
        '''
        x = F.relu(self.decode1(x))
        x = F.relu(self.decode2(x))
        x = torch.sigmoid(self.decode3(x))
        return x

    def forward(self, x):
        '''
        Forward pass through the network
        
        Inputs:
        - x: The input data

        Outputs:
        - x: The reconstructed data
        - mu: The mean of the latent layer
        - var: The variance of the latent layer
        '''
        mu, log_var = self.encode(x.view(-1, 196))
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps  # Sampling
        return self.decode(z), mu, log_var

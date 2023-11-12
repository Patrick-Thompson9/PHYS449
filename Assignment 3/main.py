import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(reconstructed, original, mu, logvar):
    bce_loss = nn.BCELoss(reduction='sum')

    # Flatten both the reconstructed and original tensors
    reconstructed_flat = reconstructed.view(-1, 784)
    original_flat = original.view(-1, 784)

    BCE = bce_loss(reconstructed_flat, original_flat)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + kld_loss

def train_vae(model, data_loader, optimizer, epochs, output_folder, verbose=False):
    model.train()
    loss_values = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for data, _ in data_loader:
            data = data.view(-1, 784)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_values.append(epoch_loss)

        if verbose:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    # Plot the loss versus epochs
    plt.plot(range(1, epochs + 1), loss_values, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')

    # Save the plot in the 'output' folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'loss.pdf')
    plt.savefig(output_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train a Variational Autoencoder on MNIST even numbers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--output_folder', type=str, default='output', help='Folder name to store the output files')
    parser.add_argument('--verbose', action='store_true', help='Print verbose training information')
    args = parser.parse_args()

    # Load MNIST data
    transform = ToTensor()
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the VAE model
    model = VAE()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the VAE
    train_vae(model, train_loader, optimizer, args.epochs, args.output_folder, verbose=args.verbose)

if __name__ == "__main__":
    main()

import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from source.model import VariationalAutoencoder


def load_data(file_path, test_size, batch_size):
    '''
    Load and preprocess the data

    Inputs:
    - file_path: The path to the data file
    - test_size: The number of testing samples
    - batch_size: The number of samples per batch

    Outputs:
    - train_loader: The training data loader
    - test_loader: The testing data loader
    '''
    data = pd.read_csv(file_path, delimiter=' ')

    # Training data split
    training = data.iloc[:-test_size - 1]
    training_data = torch.tensor(training.iloc[:, :-1].values, dtype=torch.float32) / 255.0
    training_labels = torch.tensor(training.iloc[:, -1].values, dtype=torch.float32)

    # Testing data split
    testing = data.iloc[29492 - test_size - 1:]
    testing_data = torch.tensor(testing.iloc[:, :-1].values, dtype=torch.float32) / 255.0
    testing_labels = torch.tensor(testing.iloc[:, -1].values, dtype=torch.float32)

    # Reshape to the correct dimensions for analysis
    training_data = training_data.reshape((-1, 14, 14))
    testing_data = testing_data.reshape((-1, 14, 14))

    # Convert to tensor with both data and labels, then load the data
    training_tensor = TensorDataset(training_data, training_labels)
    testing_tensor = TensorDataset(testing_data, testing_labels)
    train_loader = DataLoader(training_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def loss_function(recon_x, x, mu, log_var):
    '''
    Calculate the loss function
    
    Inputs:
    - recon_x: The reconstructed data
    - x: The input data
    - mu: The mean of the latent layer
    - log_var: The log variance of the latent layer

    Outputs:
    - BCE + KLD: The loss function
    '''
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 196), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, dataloader, optimizer, epoch, verbosity):
    '''
    Train the model

    Inputs:
    - model: The model to train
    - dataloader: The data loader
    - optimizer: The optimizer
    - epoch: The current epoch
    - verbosity: Whether program is in verbose mode

    Outputs:
    - average_loss: The average loss over the epoch
    '''
    model.train()
    total_loss = 0
    count = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1

        if verbosity and batch_idx % 100 == 0:
            print(f"\tEpoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data):.6f}")

    average_loss = total_loss / len(dataloader.dataset)
    return average_loss

def test(model, dataloader):
    '''
    Test the model

    Inputs:
    - model: The model to test
    - dataloader: The data loader

    Outputs:
    - test_loss: The average loss over the test set
    '''
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _ in dataloader:
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()

    test_loss /= len(dataloader.dataset)
    return test_loss

def create_sample_images(model, result_dir, n_outputs, verbosity):
    '''
    Create sample images

    Inputs:
    - model: The trained model
    - result_dir: The directory to save the images
    - n_outputs: The number of images to create
    - verbosity: Whether program is in verbose mode

    Outputs:
    - None
    '''
    # Load the trained model
    model = torch.load(result_dir+'model.pth')

    # Inform user on creating sample images
    if verbosity:
        print(f'Creating {n_outputs} sample images')

    # run model and create n sample images
    model.eval()
    os.makedirs(result_dir+'sample_images', exist_ok=True)
    for i in range(1, n_outputs + 1):
        with torch.no_grad():
            z_grid = torch.randn(1, 2)
            sample = model.decode(z_grid)
            save_image(sample.view(1, 1, 14, 14), f"{result_dir}sample_images/{i}.pdf")
    print('Done!')

def main():
    parser = argparse.ArgumentParser(description="Variational Autoencoder for even numbers generation.")
    parser.add_argument('-epochs', default=10, help='Number of epochs.', type=int)
    parser.add_argument('-input_file', default='data/even_mnist.csv', help='Relative path to data file.', type=str)
    parser.add_argument('-param_file', default='param/parameters.json', help='Relative path to json parameter file.', type=str)
    parser.add_argument('-n', default=100, help='Number of sample images to generate.', type=int)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose version of the program.')

    args = parser.parse_args()
    num_epochs = args.epochs
    input_file = args.input_file
    param_file = args.param_file
    n_outputs = args.n
    verbosity = args.verbose

    # Get the absolute file path from the relative
    current_dir = os.path.abspath(os.path.dirname(__file__))
    input_path = os.path.join(current_dir, input_file)
    json_path = os.path.join(current_dir, param_file)

    # Load parameters from the JSON file
    with open(json_path, 'r') as file:
        params = json.load(file)
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        test_size = params['testing_size']
        result_dir = params['output_folder']
        
    # Load and preprocess the data
    train_loader, test_loader = load_data(input_path, test_size, batch_size)

    # Initialize the model and optimizer
    model = VariationalAutoencoder(196, 128, 64, 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model and make output folder if necessary
    epoch_train_loss = []
    epoch_test_loss = []
    os.makedirs(result_dir, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, epoch, verbosity)
        test_loss = test(model, test_loader)
        epoch_train_loss.append(train_loss)
        epoch_test_loss.append(test_loss)

        print(f"\tAverage Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}\n")


    # Save the trained model
    torch.save(model, result_dir+'model.pth')

    # Plot the loss over epochs
    if verbosity:
        print('Plotting Loss')
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, num_epochs + 1), epoch_train_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), epoch_test_loss, label='Testing Loss')
    plt.title('Loss over the Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save loss plot
    if verbosity:
        print('Saving Loss Plot')
    plt.savefig(f'{result_dir}loss_plot.pdf')

    # Create sample images
    create_sample_images(model, result_dir, n_outputs, verbosity)

if __name__ == '__main__':
    main()

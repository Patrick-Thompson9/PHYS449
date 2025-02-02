import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from source.model import ModifiedRNNModel


# Custom dataset class
class BinaryMultiplicationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Function to generate dataset
def generate_dataset(seed, train_size, test_size):
    np.random.seed(seed)

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for _ in range(train_size + test_size):
        A = np.random.randint(0, 2, size=(8,), dtype=int)
        B = np.random.randint(0, 2, size=(8,), dtype=int)
        C = np.binary_repr(np.dot(A, B), width=16)

        # Create input sequence: a_0 b_0 a_1 b_1 ... a_n b_n
        input_sequence = np.concatenate([A, B])

        # Create output sequence: c_0 c_1 ... c_{2n-1} c_2n
        output_sequence = [int(bit) for bit in C]

        if len(X_train) < train_size:
            X_train.append(input_sequence)
            Y_train.append(output_sequence)
        else:
            X_test.append(input_sequence)
            Y_test.append(output_sequence)

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

def parse_arguments():
    parser = argparse.ArgumentParser(description="RNN Binary Multiplication Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for dataset generation")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument("--training_size", type=int, default=8000, help="Size of the training dataset")
    parser.add_argument("--test_size", type=int, default=2000, help="Size of the test dataset")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Hyperparameters
    input_size = 16
    hidden_size = 16
    output_size = 16

    # Generate dataset
    X_train, Y_train, X_test, Y_test = generate_dataset(seed=args.random_seed, train_size=args.training_size, test_size=args.test_size)

    # Create DataLoader for training and testing
    train_dataset = BinaryMultiplicationDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = BinaryMultiplicationDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the modified model
    model = ModifiedRNNModel(input_size, hidden_size, output_size)

    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

            val_loss /= len(test_loader)
            print(f"Training Loss: {loss.item()}, Validation Loss: {val_loss}")

        # Append losses for plotting
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        model.train()

    # Create an output folder if it doesn't exist
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
    plt.show()

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.pth'))

if __name__ == "__main__":
    main()

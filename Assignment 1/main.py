import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Define the RNN model with two hidden layers and ReLU activation
class ModifiedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Define the RNN model with two hidden layers and ReLU activation.

        Inputs:
        input_size: Size of the input
        hidden_size: Size of the hidden layer
        output_size: Size of the output

        Outputs:
        out: Output of the model
        '''
        super(ModifiedRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc_hidden1 = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        ''''
        Foward propagation of the model.

        inputs:
        x: Input to the model

        outputs:
        out: Output of the model
        '''
        out, _ = self.rnn(x)
        out = self.fc_hidden1(out)
        out = self.relu(out)
        out = self.fc_hidden2(out)  # Second hidden layer
        out = self.relu(out)
        out = self.fc_output(out)
        out = torch.sigmoid(out)  # Use sigmoid activation for binary classification
        return out

# Custom dataset class
class BinaryMultiplicationDataset(Dataset):
    '''
    Custom dataset class for binary multiplication.
    '''
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Function to generate dataset
def generate_dataset(seed, train_size, test_size):
    '''
    Generate dataset for binary multiplication and split into trainig and test sets.

    Inputs:
    seed: Random seed for reproducibility
    train_size: Size of the training dataset
    test_size: Size of the test dataset

    Outputs:
    X_train: Training inputs
    Y_train: Training labels
    X_test: Test inputs
    Y_test: Test labels
    '''
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
    '''
    Define the arguments and help statements.
    '''
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate

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

    # Plot the loss versus epochs
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()



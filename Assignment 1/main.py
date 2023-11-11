import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

# Define the RNN model with one hidden layer and ReLU activation
class ModifiedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModifiedRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc_hidden(out)
        out = self.relu(out)
        out = self.fc_output(out)
        return out

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

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Hyperparameters
    input_size = 16
    hidden_size = 16
    output_size = 16

    # Generate dataset
    X_train, Y_train, X_test, Y_test = generate_dataset(seed=args.random_seed, train_size=8000, test_size=2000)

    # Create DataLoader for training and testing
    train_dataset = BinaryMultiplicationDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = BinaryMultiplicationDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the modified model
    model = ModifiedRNNModel(input_size, hidden_size, output_size)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        model.train()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn

# Define the RNN model with two hidden layers and ReLU activation
class ModifiedRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModifiedRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc_hidden1 = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc_hidden1(out)
        out = self.relu(out)
        out = self.fc_hidden2(out)  # Second hidden layer
        out = self.relu(out)
        out = self.fc_output(out)
        out = torch.sigmoid(out)  # Use sigmoid activation for binary classification
        return out
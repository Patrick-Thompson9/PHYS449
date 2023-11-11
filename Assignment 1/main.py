import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to generate binary numbers for dataset
def generate_binary_numbers(num_samples, max_digits=8):
    binary_numbers = []
    for _ in range(num_samples):
        A = np.random.randint(2, size=np.random.randint(1, max_digits + 1))
        B = np.random.randint(2, size=np.random.randint(1, max_digits + 1))
        C = np.binary_repr(int(''.join(map(str, A)), 2) * int(''.join(map(str, B)), 2), width=16)
        binary_numbers.append((A, B, C))
    return binary_numbers

# Function to create input and output bit strings
def create_bit_strings(A, B, C):
    input_str = ' '.join([f'{a} {b}' for a, b in zip(A, B)]) + ' 0'
    output_str = ' '.join(C)
    return input_str, output_str

# Function to convert bit strings to tensors
def bit_string_to_tensor(bit_string):
    return torch.tensor([int(bit) for bit in bit_string.split()], dtype=torch.float32)

# RNN Model
class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryMultiplicationRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# Function to train the RNN model
def train_rnn_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Generate dataset
train_data = generate_binary_numbers(num_samples=1000)
test_data = generate_binary_numbers(num_samples=100)

# Convert dataset to bit strings and tensors
train_data = [(bit_string_to_tensor(create_bit_strings(A, B, C)[0]), bit_string_to_tensor(create_bit_strings(A, B, C)[1])) for A, B, C in train_data]
test_data = [(bit_string_to_tensor(create_bit_strings(A, B, C)[0]), bit_string_to_tensor(create_bit_strings(A, B, C)[1])) for A, B, C in test_data]

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize the RNN model
model = BinaryMultiplicationRNN(input_size=1, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_rnn_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Test the model
with torch.no_grad():
    for inputs, targets in test_data:
        outputs = model(inputs.unsqueeze(0))
        predicted = torch.round(outputs.squeeze()).int()
        print(f'Input: {inputs.int()}, Target: {targets.int()}, Predicted: {predicted}')
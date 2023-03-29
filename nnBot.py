
# The author of the following networks is Bing Ai Chat
# target_size = 1
# This sets the `target_size` variable to 1, which means that the network will predict one day ahead.
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests

# Define the network architecture
class DFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)
        return out

# Set the API endpoint and parameters
symbol = 'BTCUSDT'
interval = '1d'
limit = 5000
url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'

# Send the API request and get the response
response = requests.get(url)
data = response.json()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignored'])

# Convert the timestamps to datetime objects
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

# Set the index to the open time
df.set_index('Open time', inplace=True)

# Set the input size, hidden size, and output size
input_size = 300
hidden_size = 100
output_size = 1

# Set the target size
target_size = 1

# Create an instance of the network
net = DFFNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.006)

# Extract the closing prices
prices = df['Close'].values

# Create the input and target data
inputs = np.zeros((prices.shape[0] - input_size - target_size + 1, input_size))
targets = np.zeros((prices.shape[0] - input_size - target_size + 1, target_size))

for i in range(inputs.shape[0]):
    inputs[i] = prices[i:i+input_size]
    targets[i] = prices[i+input_size:i+input_size+target_size]

# Convert the data to PyTorch tensors
inputs = torch.from_numpy(inputs).float()
targets = torch.from_numpy(targets).float()

# Train the network
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests

# Define the network architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Set the API endpoint and parameters
symbol = 'BTCUSDT'
interval = '1d'
limit = 1000
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

# Set the input size, hidden size, number of layers and output size
input_size = 1
hidden_size = 50
num_layers = 5
output_size = 1

# Create an instance of the network
net = LSTM(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=10)

# Extract the closing prices and normalize the data
prices = df['Close'].values.reshape(-1, 1)
# prices = (prices - np.mean(prices)) / np.std(prices)
# prices = (prices - np.mean(prices)) * (1 / np.std(prices))

# Set the sequence length and target size
seq_length = 10
target_size = 1

# Create the input and target data
inputs = np.zeros((prices.shape[0] - seq_length - target_size + 1, seq_length))
targets = np.zeros((prices.shape[0] - seq_length - target_size + 1))

for i in range(inputs.shape[0]):
    inputs[i] = prices[i:i+seq_length].flatten()
    targets[i] = prices[i+seq_length]

# Convert the data to PyTorch tensors and reshape the inputs for LSTM
inputs = torch.from_numpy(inputs).float().unsqueeze(-1)
targets = torch.from_numpy(targets).float()

# Train the network
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs.squeeze(), targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# This script is similar to the previous DFFNN script but uses an LSTM network instead. The LSTM network takes as input a sequence of `seq_length` price data points and outputs a prediction for the next day's price.
# The script also includes some additional data preprocessing steps to normalize the price data before training. Normalizing the data can help improve the performance of the model.
# You can try running this script to see if it improves the performance of your model. You can also experiment with different values of `hidden_size`, `num_layers`, `seq_length`, and other hyperparameters to see if they improve performance.
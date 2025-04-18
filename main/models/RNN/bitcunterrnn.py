import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class BitCounterRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='RNN', dropout=0.0):
        super(BitCounterRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Choose the RNN type (RNN, LSTM, GRU)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Invalid RNN type. Choose 'RNN', 'LSTM', or 'GRU'.")

        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        if isinstance(self.rnn, nn.LSTM):  # For LSTM, initialize both h0 and c0
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x.unsqueeze(-1), (h0, c0))
        else:
            out, _ = self.rnn(x.unsqueeze(-1), h0)

        # Only the last time-step output is used
        out = self.fc(out[:, -1, :])
        return out

# Training and Evaluation
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

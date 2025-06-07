import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context

class PM25Predictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        context = self.attn(out)
        return self.decoder(context)

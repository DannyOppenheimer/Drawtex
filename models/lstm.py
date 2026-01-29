import torch.nn as nn
import torch


class StrokeClassifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StrokeClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )
        logits = self.fc(lstm_out)
        return logits

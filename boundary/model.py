from torch import nn


class NN(nn.Module):
    def __init__(self, hidden_size, sequence_len, num_classes, device):
        super(NN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        outputs, _ = self.lstm(x)
        out = self.linear(outputs)
        return out

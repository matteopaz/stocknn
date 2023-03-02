import torch
import torch.nn as nn

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
       

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).requires_grad_()
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            x = x.cuda()


        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # (batch, seq, hidden) -> (batch, hidden)
        out = self.fc(out)
        return out
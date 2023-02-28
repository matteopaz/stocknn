import helpers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datastruct import *

brkb_tr = helpers.load("./training/brkb_train_week.pkl")
brkb_ts = helpers.load("./training/brkb_test_week.pkl")

trainloader = DataLoader(brkb_tr, batch_size=100, drop_last=True)
testloader = DataLoader(brkb_ts, batch_size=1, drop_last=True)


# hyperparams

input_size = 2 # Daily vectors
sequence_length = 57 # How many days input
hidden_size = 8 # hidden layer features


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

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # (batch, seq, hidden) -> (batch, hidden)
        out = self.fc(out)
        return out
        



shortmodel = LSTM1(input_size, hidden_size, 1, 1)
print(shortmodel(brkb_tr[0][0].reshape(1, 8, 2)))
print(shortmodel(brkb_tr[100][0].reshape(1, 8, 2)))
print(shortmodel(brkb_tr[300][0].reshape(1, 8, 2)))

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(shortmodel.parameters(), lr=0.01)

EPOCHS = 100

helpers.train(trainloader, shortmodel, optimizer, loss, EPOCHS, print_every=1)

print("AFTER")

print(shortmodel(brkb_tr[0][0].reshape(1, 8, 2)))
print(shortmodel(brkb_tr[100][0].reshape(1, 8, 2)))
print(shortmodel(brkb_tr[300][0].reshape(1, 8, 2)))
# torch.save(shortmodel.state_dict(), "./models/brkb_model_short.pt")

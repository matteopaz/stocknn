import helpers
import torch
from torch import nn
# brkb_train_sixmo = helpers.load("./training/brkb_train_sixmo")
# brkb_test_sixmo = helpers.load("./training/brkb_test_sixmo")
brkb_train_week = helpers.load("./training/brkb_train_week_raw")
brkb_test_week = helpers.load("./training/brkb_test_week")

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
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        self.fc = nn.Linear(hidden_size, output_size)
       

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=torch.float32).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, dtype=torch.float32).requires_grad_()
        print()
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :] # (seq, batch, hidden) -> (batch, hidden)
        out = self.fc(out)
        return out
        



rmodel1 = LSTM1(input_size, hidden_size, 1, 1)

print(rmodel1(brkb_train_week[0][0]))

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rmodel1.parameters(), lr=0.01)


for epoch in range(1):

    for i, (inp, actual) in enumerate(brkb_train_week):
        y = actual[0][1].reshape(1,1) # get the close price
        optimizer.zero_grad()
        out = rmodel1(inp)
        loss_value = loss(out, y)
        loss_value.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        print("Epoch: ", epoch, "Loss: ", loss_value.item())


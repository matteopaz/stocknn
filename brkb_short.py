import helpers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datastruct import *
import matplotlib.pyplot as plt

brkb_tr = helpers.load("./training/brkb_train_week.pkl")
brkb_ts = helpers.load("./training/brkb_test_week.pkl")

trainloader = DataLoader(brkb_tr, batch_size=128, drop_last=True)
testloader = DataLoader(brkb_ts, batch_size=1, drop_last=True)


# hyperparams

input_size = 2 # Daily vectors
sequence_length = 57 # How many days input
hidden_size = 16 # hidden layer features


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
        
loss = torch.nn.MSELoss()

model = LSTM1(input_size, hidden_size, 1, 1)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
if torch.cuda.is_available():
    model.cuda()

# helpers.train(trainloader, model, optimizer, loss, 131, 10)

# torch.save(model.state_dict(), "./models/brkb_model_short.pt")

# EPOCHS = 300

# model1x8 = LSTM1(input_size, 8, 1, 1)
# model1x16 = LSTM1(input_size, 16, 1, 1)
# model1x64 = LSTM1(input_size, 64, 1, 1)

# optimizer1 = torch.optim.Adam(params=model1x8.parameters(), lr=0.01)
# optimizer2 = torch.optim.Adam(params=model1x16.parameters(), lr=0.01)
# optimizer3 = torch.optim.Adam(params=model1x64.parameters(), lr=0.01)

# if torch.cuda.is_available():
#     model1x8.cuda()
#     model1x16.cuda()
#     model1x64.cuda()

# def lossmetric(modelfn):
#     total_loss = 0
#     for batch in testloader:
#         inp = batch[0]
#         label = batch[1].reshape(1, 1)
#         out = modelfn(inp)

#         lossval = (label.item() - out.item())**2

#         total_loss += lossval   

#     return total_loss

# x, yl1, ym1 = helpers.train(trainloader, model1x8, optimizer1, loss, EPOCHS, print_every=25, plotting=True, metric=lossmetric)
# _, yl2, ym2 = helpers.train(trainloader, model1x16, optimizer2, loss, EPOCHS, print_every=25, plotting=True, metric=lossmetric)
# _, yl3, ym3 = helpers.train(trainloader, model1x64, optimizer3, loss, EPOCHS, print_every=25, plotting=True, metric=lossmetric)

# torch.save(model1x8.state_dict(), "./models/brkb_model1x8.pt")
# torch.save(model1x16.state_dict(), "./models/brkb_model1x16.pt")
# torch.save(model1x64.state_dict(), "./models/brkb_model1x64.pt")

# plt.plot(x, yl1, label="loss")
# plt.plot(x, ym2, label="metric")
# plt.legend()
# plt.savefig("brkb_loss_metric1x8.png")
# plt.show()

# plt.plot(x, yl2, label="loss")
# plt.plot(x, ym2, label="metric")
# plt.legend()
# plt.savefig("brkb_loss_metric1x16.png")
# plt.show()

# plt.plot(x, yl3, label="loss")
# plt.plot(x, ym3, label="metric")
# plt.legend()
# plt.savefig("brkb_loss_metric1x64.png")
# plt.show()





 


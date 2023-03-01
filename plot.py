from brkb_short import LSTM1, input_size, hidden_size
import torch
from torch.utils.data import Dataset, DataLoader
import helpers
import matplotlib.pyplot as plt
import math

weektrainset = helpers.load("./training/brkb_train_week.pkl")
weektestset = helpers.load("./training/brkb_test_week.pkl")

weektestloader = DataLoader(weektestset, batch_size=1, drop_last=True)
weektrainloader = DataLoader(weektrainset, batch_size=1, drop_last=True)


naive = LSTM1(input_size, hidden_size, 1, 1)


weekPrediction = LSTM1(input_size, hidden_size, 1, 1)

weekPrediction.load_state_dict(torch.load("./models/brkb_model_short.pt", map_location=torch.device('cpu')))
weekPrediction.eval()

if torch.cuda.is_available():
    naive.cuda()
    weekPrediction.cuda()

def loss(modelfn, testloader):
    total_loss = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1].reshape(1, 1)
        out = modelfn(inp)
        lossval = (label.item() - out.item())**2
        print(out.item(), label.item())
        total_loss += lossval   
        # print("Model thinks: ", math.round(out), "Actual: ", math.round(label)s ")
    return total_loss

def positive_trading_profit(modelfn, testloader):
    total_profit = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1]

        purchaseprice = inp[0][-1][1]
        prediction = modelfn(inp)
        # print(prediction, label)
        if prediction > purchaseprice:
            total_profit += label - purchaseprice
    return total_profit

i = 0
x = []
y = []
predy = []
naivey = []
for batch in weektrainloader:
    x.append(i)
    i += 1
    inp = batch[0]
    label = batch[1]
    if torch.cuda.is_available():
        inp = inp.cuda()
        label = label.cuda()
    out = weekPrediction(inp)
    naiveout = naive(inp)
    y.append(label.item())
    predy.append(out.item())
    naivey.append(naiveout.item())

for batch in weektestloader:
    x.append(i)
    i += 1
    inp = batch[0]
    label = batch[1]
    out = weekPrediction(inp)
    naiveout = naive(inp)
    y.append(label.item())
    predy.append(out.item())
    naivey.append(naiveout.item())

plt.plot()
plt.axvline(x = len(weektrainloader), color = 'b', label = 'axvline - full height')
plt.plot(x, y, label="Actual", color="red")
plt.plot(x, predy, label="Predicted", color="blue")
plt.plot(x, naivey, label="Naive", color="green")
plt.legend()
plt.show()

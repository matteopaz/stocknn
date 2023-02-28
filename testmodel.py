from brkb_short import LSTM1, input_size, hidden_size
import torch
from torch.utils.data import Dataset, DataLoader
import helpers
import math

weektestset = helpers.load("./training/brkb_test_week.pkl")
weektestloader = DataLoader(weektestset, batch_size=1, drop_last=True)

naive = LSTM1(input_size, hidden_size, 1, 1)
weekPrediction = LSTM1(input_size, hidden_size, 1, 1)

weekPrediction.load_state_dict(torch.load("./models/brkb_model_short.pt"))
weekPrediction.eval()

def loss(modelfn, testloader):
    total_loss = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1].reshape(1, 1)
        out = modelfn(inp)
        lossval = (label.item() - out.item())**2
        # print(out.item(), label.item())
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

        if prediction < purchaseprice:
            total_profit += purchaseprice - label
    return total_profit


print(loss(weekPrediction, weektestloader))
print(positive_trading_profit(weekPrediction, weektestloader))
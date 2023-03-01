from brkb_short import LSTM1, input_size, hidden_size
import torch
from torch.utils.data import Dataset, DataLoader
import helpers
import numpy as np
from parsedata import revnormp

weektestset = helpers.load("./training/brkb_test_week.pkl")
weektestloader = DataLoader(weektestset, batch_size=1, drop_last=True)

naive = LSTM1(input_size, hidden_size, 1, 1)
if torch.cuda.is_available():
    naive.cuda()
weekPrediction = LSTM1(input_size, hidden_size, 1, 1)

weekPrediction.load_state_dict(torch.load("./models/brkb_model_short.pt"))
weekPrediction.eval()
if torch.cuda.is_available():
    weekPrediction.cuda()

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
    rev = 0
    cost = 0

    for batch in testloader:
        inp = batch[0]
        label = batch[1]

        purchaseprice = inp[0][-1][1]
        prediction = modelfn(inp)
        

        if prediction > purchaseprice:
            cost += revnormp(purchaseprice)
            rev += revnormp(label)
    
    return (cost, rev, rev - cost)



def mean_reversion(testloader):
    total_profit = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1]
        prices = np.array([day[1] for day in inp[0]])
        
        mean = np.mean(prices)
        today = prices[-1]
        if today < mean:
            total_profit += label.item() - today
        
    return revnormp(total_profit)

def momentum(testloader):
    total_profit = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1]
        prices = np.array([day[1] for day in inp[0]])

    
        mean = np.mean(prices)
        today = prices[-1]

        if today > mean:
            total_profit += today - label.item()
    return revnormp(total_profit)

def justbuy(testloader):
    total_profit = 0
    for batch in testloader:
        inp = batch[0]
        label = batch[1]
        today = inp[0][-1][1]
        total_profit += label.item() - today
    return revnormp(total_profit)



firstprice = list(weektestloader)[0][0][:, -1, 1].item()
lastprice = list(weektestloader)[-1][1].item()
chg = revnormp(lastprice - firstprice)
print("change: ", chg)


print(positive_trading_profit(weekPrediction, weektestloader))
# print("naive:", positive_trading_profit(naive, weektestloader))
print("momentum: ", momentum(weektestloader))
print("mean reversion:", mean_reversion(weektestloader))
print("just buy:", justbuy(weektestloader))


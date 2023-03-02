import torch
from torch.utils.data import DataLoader
import helpers
import matplotlib.pyplot as plt
import numpy as np

weektestset = helpers.load("./training/brkb_test_week.pkl")
weektestloader = DataLoader(weektestset, batch_size=1, drop_last=True)
y = []
yprof = []

def justbuy(dataloader):
    total_profit = 0
    i = 0
    for batch in dataloader:
        if i in range(40, 95):
            inp = batch[0]
            label = batch[1].item()
            purchaseprice = inp[0][-1][1]
            y.append(purchaseprice)
            yprof.append(label - purchaseprice)
            print(i)
            total_profit += label - purchaseprice
        i += 1



x = range(40, 95)
justbuy(weektestloader)

yprof = np.cumsum(yprof).tolist()
plt.plot(x, y, label="Actual", color="red")
plt.plot(x, yprof, label="Profit", color="blue")
plt.legend()
plt.show()

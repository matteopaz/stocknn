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

x= []
y = []
i = 0
for batch in weektrainloader:
    x.append(i)
    i += 1
    inp = batch[0]
    label = batch[1]
    y.append(label.item())

for batch in weektestloader:
    x.append(i)
    i += 1
    inp = batch[0]
    label = batch[1]

    y.append(label.item())


plt.plot()
plt.axvline(x = len(weektrainloader), color = 'b', label = 'axvline - full height')
plt.plot(x, y, label="Actual", color="red")
plt.legend()
plt.show()

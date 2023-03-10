import pandas
from helpers import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datastruct import *

# 7 days
# 6 months = 182 days
# 1 year = 365 days
# 3 years = 1095 days


LEAVEOUTFORTEST = 800 # days
brkb = pandas.read_csv("./data/BRK-B.csv")

closes= np.array([float(x) for x in brkb["Close"]])
ranges = np.array([float(x) for x in brkb["High"]]) - np.array([float(x) for x in brkb["Low"]])
maxp = closes.max()
minp = closes.min()
maxr = ranges.max()
minr = ranges.min()

normp = lambda x: (x - minp) / (maxp - minp)
normr = lambda x: (x - minr) / (maxr - minr)

revnormp = lambda x: x * (maxp - minp) + minp
revnormr = lambda x: x * (maxr - minr) + minr

# maybe shuffle data and then split to randomize test data

brkb_train = brkb[:-LEAVEOUTFORTEST]
brkb_test = brkb[-LEAVEOUTFORTEST:]



brkb_train_week = SDataset(brkb_train, 7, 1, normp, normr)
brkb_test_week = SDataset(brkb_test, 7, 1, normp, normr)

brkb_train_4month = SDataset(brkb_train, 120, 1, normp, normr)
brkb_test_4month = SDataset(brkb_test, 120, 1, normp, normr)

brkb_train_1year = SDataset(brkb_train, 365, 1, normp, normr)
brkb_test_1year = SDataset(brkb_test, 365, 1, normp, normr)

save(brkb_train_week, "./training/brkb_train_week.pkl")
save(brkb_test_week, "./training/brkb_test_week.pkl")
save(brkb_train_4month, "./training/brkb_train_4month.pkl")
save(brkb_test_4month, "./training/brkb_test_4month.pkl")
save(brkb_train_1year, "./training/brkb_train_1year.pkl")
save(brkb_test_1year, "./training/brkb_test_1year.pkl")

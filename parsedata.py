import pandas
from helpers import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datastruct import *

# 7 days
# 6 months = 182 days
# 1 year = 365 days
# 3 years = 1095 days


LEAVEOUTFORTEST = 365 # days
brkb = pandas.read_csv("./data/BRK-B.csv")


brkb_train = brkb[:-LEAVEOUTFORTEST]
brkb_test = brkb[-LEAVEOUTFORTEST:]
brkb_train_week = SDataset(brkb_train, 7, 1)
brkb_test_week = SDataset(brkb_test, 7, 1)


save(brkb_train_week, "./training/brkb_train_week.pkl")
save(brkb_test_week, "./training/brkb_test_week.pkl")

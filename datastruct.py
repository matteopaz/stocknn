from helpers import *
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

backmultiple = 1 # ############################# 

class SDataset(Dataset):

    def __init__(self, raw, prediction_distance, every=1):
        self.ts = []
        for i in range(len(raw)):
            high = list(raw["High"])[i]
            low = list(raw["Low"])[i]
            close = list(raw["Close"])[i]
            pricerange = high - low
            self.ts.append([pricerange, close])

        self.input_days = backmultiple * prediction_distance
        self.prediction_distance = prediction_distance

        self.inputs = []
        self.labels = []

        self.days = len(self.ts)
    
        for i in range(self.days - self.input_days - prediction_distance):
            fromend = -(i+1)

            label = torch.tensor(self.ts[fromend][1]).to(torch.float32) # only close
            self.labels.append(label)

            windowstart = fromend - self.prediction_distance - self.input_days
            windowend = fromend - self.prediction_distance + 1

            inp = torch.tensor(self.ts[windowstart:windowend: every]).to(torch.float32)
            self.inputs.append(inp)
    
    
    def __getitem__(self, index):
        return (self.inputs[index], self.labels[index])
    
    def __len__(self):
        return len(self.inputs)
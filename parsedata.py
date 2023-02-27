import pandas
from helpers import *
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

backmultiple = 8

# 7 days
# 6 months = 182 days
# 1 year = 365 days
# 3 years = 1095 days

def generate_dataset(raw, leave_out, shifting, prediction_distance, every=1):
    asset = {}
    asset['raw'] = raw
    asset['timeseries'] = []
    r = len(asset['raw'])
    for i in range(r):
        open = asset['raw']["Open"][i]
        high = asset['raw']["High"][i]
        low = asset['raw']["Low"][i]
        close = asset['raw']["Close"][i]
        pricerange = high - low
        asset['timeseries'].append([[pricerange, close]])
        # flatten brkb.timeseries
        # asset['flat_timeseries'] = torch.tensor([[item for sl in asset['timeseries'] for item in sl]]).transpose(0,1)

    input_days = backmultiple * prediction_distance

    trainingset = []
    testset = []
    days = len(asset['timeseries'])
    if days < input_days + prediction_distance + leave_out:
        raise Exception("Not enough data for this prediction distance and leave out")
    
    for i in range(days - input_days - prediction_distance - leave_out):
        fromend_test = -(i+1)
        fromend_train = fromend_test - leave_out

        actual = torch.tensor(asset['timeseries'][fromend_train])
        actual = actual.to(torch.float32)

        inp = torch.tensor(asset['timeseries'][(fromend_train-prediction_distance-input_days):(fromend_train-prediction_distance + 1): every])
        inp = inp.to(torch.float32)
        
        # inp = torch.tensor([[item for sl in bumpy_inp for item in sl]]).transpose(0,1)
        trainingset.append((inp, actual))

        actual_test = torch.tensor([asset['timeseries'][fromend_test]])
        actual_test = actual_test.to(torch.float32)

        inp_test = torch.tensor(asset['timeseries'][(fromend_test-prediction_distance-input_days):(fromend_test-prediction_distance + 1): every])
        inp_test = inp_test.to(torch.float32)
        # inp_test = torch.tensor([[item for sl in bumpy_inp_test for item in sl]]).transpose(0,1)
        testset.append((inp_test, actual_test))


    return (trainingset, testset)

class SDataset(Dataset):

    def __init__(self, raw, prediction_distance, every=1):
        self.ts = []
        for i in range(len(raw)):
            high = raw["High"][i]
            low = raw["Low"][i]
            close = raw["Close"][i]
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


brkb = pandas.read_csv("./data/BRK-B.csv")

brkb_train_week = SDataset(brkb, 7, 1)
save(brkb_train_week, "./data/brkb_train_week.pkl")

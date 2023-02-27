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

brkb = pandas.read_csv("./data/BRK-B.csv")
# brkb_trainingset_sixmo_raw, brkb_testset_sixmo_raw = generate_dataset(brkb, 365, 1, 182)
brkb_trainingset_week_raw, brkb_testset_week_raw = generate_dataset(brkb, 365, 1, 7)

# brkb_train_sixmo = DataLoader(brkb_trainingset_sixmo_raw, batch_size=1, shuffle=False)
# brkb_test_sixmo = DataLoader(brkb_testset_sixmo_raw, batch_size=1, shuffle=False)


brkb_train_week = DataLoader(brkb_trainingset_week_raw, batch_size=10, shuffle=False)
brkb_test_week = DataLoader(brkb_testset_week_raw, batch_size=10, shuffle=False)

# save(brkb_train_sixmo, "./training/brkb_train_sixmo")
# save(brkb_test_sixmo, "./training/brkb_test_sixmo")
save(brkb_train_week, "./training/brkb_train_week")
save(brkb_test_week, "./training/brkb_test_week")
save(brkb_trainingset_week_raw, "./training/brkb_train_week_raw")
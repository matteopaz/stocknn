import pickle
from datastruct import *
import math
import torch

def save(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def train(dataloader, model, optimizer, loss, epochs, print_every=100, plotting=False, metric=None):
    last_loss = 0
    first_loss =  0
    x = range(epochs)
    y_epochloss = []
    y_metric = []

    for epoch in range(epochs):
        i = 0
        epoch_loss = 0
        for batch in dataloader:
            inp = batch[0]
            label = batch[1].reshape(len(inp), 1)
            if torch.cuda.is_available():
                inp = inp.cuda()
                label = label.cuda()
            i += 1

            optimizer.zero_grad()
            out = model(inp)
            loss_value = loss(out, label)
            epoch_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
        
        if epoch == 0:
            first_loss = epoch_loss

        if epoch % print_every == 0:
            print("Epoch: ", epoch, "Loss: ", epoch_loss, "Chg: ", epoch_loss - last_loss, "Net: ", epoch_loss - first_loss)
        
        if plotting:
            y_epochloss.append(epoch_loss)
            if metric:
                y_metric.append(metric(model))

        last_loss = epoch_loss
    return x, y_epochloss, y_metric


        
import iteration as it
from model import GCN
import torch
from torch_geometric.loader import DataLoader
import torch
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import os
from tqdm import tqdm
import random

def train():
    model.train()

    total_loss = 0
    for data in training_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 

def test(loader):
    model.eval()

    correct = 0
    for data in loader: 
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum()) 
    return correct / len(loader.dataset) 
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

raw_data = loadmat('Data/circuit_data.mat', squeeze_me=True)
data = pd.DataFrame(raw_data['Graphs'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

median_performance = []
n = 5 #number of iterations
epochs = 600
p_known = 0.2 #what percentage of the known data would you like to use for training?
training_split = 0.8 #what percentage of the training data would you like to use for training and validation?
csv_save_path = 'path' #Enter your desired save path for the csv results
data_save_path = 'path' #Enter the desired save path to store the data

for run in range(0,n):
    seed = random.randint(10000,99999)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()
    for iteration in range(0,n):
        model = GCN(hidden_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()


        if iteration == 0:



            all_perm = np.random.permutation(len(data))
            All_index_split = int(len(data)*p_known)



            known_indices = all_perm[:All_index_split]
            unknown_indices = all_perm[All_index_split+1:]
        else:
            n_known_min = 2000
            n_known_ones = len(known_ones_index)
            known_set_sizes = []
            if n_known_ones < n_known_min :
                n_known_needed = n_known_min - n_known_ones
                predicted_ones_index_needed = predicted_ones_index[:n_known_needed]

                known_indices = np.concatenate((known_ones_index, predicted_ones_index_needed))
                known_set_sizes.append(len(known_indices))
                unknown_indices = predicted_ones_index[n_known_needed:]            
                print('Known Set Size: ', known_set_sizes[-1])
            else:
                known_indices = known_ones_index
                unknown_indices = predicted_ones_index
            print('Worked!')

        

        known_graphs = data.loc[known_indices]
        unknown_graphs = data.loc[unknown_indices]

        known_performance = data['Labels'][known_indices]

        known_median = np.median(known_performance)

        try:
            os.remove(f'{data_save_path}/known_data/processed/data.pt')
        except OSError as e:

            print('Error')
        
        
        try:
            os.remove(f'{data_save_path}/unknown_data/processed/data.pt')
        except OSError as e:
            print('Error')
        
        known_torch = it.IterationDataset(root='known_data', data=known_graphs, performance_threshold=known_median, transform=None, pre_transform=None, pre_filter=None)
        unknown_torch = it.IterationDataset(root='unknown_data', data=unknown_graphs, performance_threshold=known_median, transform=None, pre_transform=None, pre_filter=None)

        training = known_torch[:int(len(known_torch)*training_split)]
        validation = known_torch[int(len(known_torch)*training_split)+1:]

        training_loader = DataLoader(training, batch_size=32, shuffle=False)
        validation_loader = DataLoader(validation, batch_size=32, shuffle=False)
        unknown_loader = DataLoader(unknown_torch, batch_size=1, shuffle=False)


        
        for epoch in tqdm(range(1, epochs + 1), total=epochs):
            train()
            train_acc = test(training_loader)       
            val_acc = test(validation_loader)         
        

        

        predictions = []
        unknown_index = []
        
        for test_graph in tqdm(unknown_loader, total=len(unknown_loader)):
            test_graph = test_graph.to(device)
            out = model(test_graph.x, test_graph.edge_index, test_graph.batch)
            pred = out.argmax(dim=1)
            predictions.append(pred.item())
            unknown_index.append(test_graph.orig_index.item())
            
        
        
        predictions = np.array(predictions)
        predicted_ones_index = unknown_indices[np.where(predictions == 1)[0]]
        predicted_zeros_index = unknown_indices[np.where(predictions == 0)[0]]

        

        known_classifications = []
        for i in range(len(known_torch)):
            known_classifications.append(known_torch[i].y.item())

        known_classifications = np.array(known_classifications)
        known_ones_index = known_indices[np.where(known_classifications == 1)[0]]
        known_zeros_index = known_indices[np.where(known_classifications == 0)[0]]
        
        print(f'Run {run}, Iteration {iteration}')
        print('Known Ones: ', len(known_ones_index))
        print('Known Zeros: ', len(known_zeros_index))
        
        saved_known_ones = pd.DataFrame(known_ones_index)
        saved_known_ones.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_known_ones.csv')

        saved_known_zeros = pd.DataFrame(known_zeros_index)
        saved_known_zeros.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_known_zeros.csv')

        saved_predicted_ones = pd.DataFrame(predicted_ones_index)
        saved_predicted_ones.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_predicted_ones.csv')

        saved_known_zeros = pd.DataFrame(predicted_zeros_index)
        saved_known_zeros.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_predicted_zeros.csv')


        median_performance.append(known_median)

    end_time = time.time()

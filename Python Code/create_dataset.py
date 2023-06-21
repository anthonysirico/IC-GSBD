from scipy.io import loadmat
import pandas as pd
from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
import torch_geometric
from tqdm import tqdm
from typing import Union
from torch import Tensor
from collections.abc import Sequence
from torch_geometric.utils import from_networkx
import networkx as nx

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class IterationDataset(InMemoryDataset):
    def __init__(self, root: str, data, performance_threshold, transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.performance_threshold = performance_threshold
        
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return '/workspaces/GDL-for-Engineering-Design/Data/circuit_data.mat'
        

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        
        r'''Create Known Graphs'''
        data_list = []
        for index, cir in tqdm(self.data.iterrows(), total=len(self.data)):
            nxg = nx.Graph(self.data['A'][index])
            nxg = self._get_node_features(nxg, self.data['Ln'][index])
            pt_graph = self._get_graph_object(nxg)
            pt_graph.y = self._get_known_graph_label(self.data['Labels'][index], self.performance_threshold)
            pt_graph.performance = torch.tensor(self.data['Labels'][index], dtype=torch.float)
            pt_graph.complexity = self._get_complexity(self.data['types'][index])
            pt_graph.orig_index = torch.tensor(index, dtype=torch.long)
            data_list.append(pt_graph)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        
        data_known, slices_known = self.collate(data_list)


        torch.save((data_known, slices_known), self.processed_paths[0])


    def _get_complexity(self, types):
        comp = len(types)
        return torch.tensor(comp, dtype=torch.float)

    def _get_node_features(self, nx_graph, node_labels):
        betweenness = list(nx.betweenness_centrality(nx_graph).values())
        eigenvector = list(nx.eigenvector_centrality(nx_graph, max_iter=600).values())
        node_label_dict = dict(enumerate(node_labels))

        mapping_dict = {'C': 0, 'G': 1, 'I': 2, 'O': 3, 'P': 4, 'R': 5}
        component_labels = []

        for value in node_label_dict.values():
            if value in mapping_dict:
                component_labels.append(mapping_dict[value])

        all_features = zip(betweenness, eigenvector, component_labels)
        all_features = dict(enumerate(all_features))

        nx.set_node_attributes(nx_graph, all_features, 'features')

        return nx_graph

    
    def _get_graph_object(self, nx_graph):
        nxg = from_networkx(nx_graph, group_node_attrs=['features'])
        return nxg

    def _get_known_graph_label(self, performance, threshold):
        if performance < threshold:
            return torch.tensor(1, dtype=torch.long)
        else:
            return torch.tensor(0, dtype=torch.long)
    
    @property
    def num_node_features(self) -> int:
        return 3

    @property
    def num_classes(self) -> int:
        return 2
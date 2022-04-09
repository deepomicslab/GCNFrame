import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial

import torch
import torch_geometric.transforms as T
import torch_geometric.utils as ut
from torch_geometric.data import Data
from GCNFrame import encode_seq

class BipartiteData(Data):
    def _add_other_feature(self, other_feature) :
        self.other_feature = other_feature

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)


class GraphDataset():
    
    def __init__(self, pnode_feature, fnode_feature, other_feature, edge, graph_label):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label


    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)
            
            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)
            
            data_list.append(data)

        return data_list

class Biodata:
    def __init__(self, fasta_file, label_file=None, feature_file=None, K=3, d=3):
        self.dna_seq = {}
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            self.dna_seq[seq_record.id] = str(seq_record.seq)

        if feature_file == None:
            self.other_feature = None
        else:
            self.other_feature = np.loadtxt(feature_file)
        
        self.K = K
        self.d = d

        self.edge = []
        for i in range(4**(K*2)):
            a = i // 4**K
            b = i % 4**K
            self.edge.append([a, i])
            self.edge.append([b, i])
        self.edge = np.array(self.edge).T
        
        if label_file:
            self.label = np.loadtxt(label_file)
        else:
            self.label = None
    
    def encode(self, thread=10):
        print("Encoding sequences...")
        seq_list = list(self.dna_seq.values())
        pool = Pool(thread)
        partial_encode_seq = partial(encode_seq.matrix_encoding, K=self.K, d=self.d)
        feature = np.array(pool.map(partial_encode_seq, seq_list))
        pool.close()
        pool.join()
        self.pnode_feature = feature.reshape(-1, self.d, 4**(self.K*2))
        self.pnode_feature = np.moveaxis(self.pnode_feature, 1, 2)
        if self.d == 1:
            zero_layer = feature.reshape(-1, self.d, 4**self.K, 4**self.K)[:, 0, :, :]
        else:
            zero_layer = feature.reshape(-1, self.d, 4**self.K, 4**self.K)[:, 1, :, :]
        self.fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4**self.K, 1)
        del zero_layer
        
        graph = GraphDataset(self.pnode_feature, self.fnode_feature, self.other_feature, self.edge, self.label)
        dataset = graph.process()
        
        return dataset






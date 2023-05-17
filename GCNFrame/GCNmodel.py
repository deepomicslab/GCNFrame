import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

import copy
import random
import heapq
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
from GCNFrame import Biodata

class model(nn.Module):
    def __init__(self, label_num, other_feature_dim, K=3, d=3, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=2, cnn_dim=64, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=100, dropout_rate=0.2, pnode_nn=True, fnode_nn=True):
        super(model, self).__init__()
        self.label_num = label_num
        self.pnode_dim = d
        self.pnode_num = 4 ** (2 * K)
        self.fnode_num = 4 ** K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout_rate
        self.pnode_nn = pnode_nn
        self.fnode_nn = fnode_nn
        self.other_feature_dim = other_feature_dim

        self.pnode_d = nn.Linear(self.pnode_num * self.pnode_dim, self.pnode_num * self.node_hidden_dim)
        self.fnode_d = nn.Linear(self.fnode_num, self.fnode_num * self.node_hidden_dim)
        
        self.gconvs_1 = nn.ModuleList()
        self.gconvs_2 = nn.ModuleList()
        
        if self.pnode_nn:
            pnode_dim_temp = self.node_hidden_dim
        else:
            pnode_dim_temp = self.pnode_dim
        
        if self.fnode_nn:
            fnode_dim_temp = self.node_hidden_dim
        else:
            fnode_dim_temp = 1
        
        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gconvs_1.append(pyg_nn.SAGEConv((fnode_dim_temp, pnode_dim_temp), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, fnode_dim_temp), self.gcn_dim))
            else:                                   
                self.gconvs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
        
        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.convs = nn.ModuleList()
        for l in range(self.cnn_layer_num):
            if l == 0:
                self.convs.append(nn.Conv1d(in_channels=self.gcn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
            else:
                self.convs.append(nn.Conv1d(in_channels=self.cnn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
        
        if self.other_feature_dim:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim + self.other_feature_dim, self.fc_dim + self.other_feature_dim)
            self.d3 = nn.Linear(self.fc_dim + self.other_feature_dim, self.label_num)
        else:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim, self.label_num)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]

        if self.other_feature_dim:
            other_feature = torch.reshape(data.other_feature, (-1, self.other_feature_dim)) 
        
        # transfer primary nodes
        if self.pnode_nn:
            x_p = torch.reshape(x_p, (-1, self.pnode_num * self.pnode_dim))
            x_p = self.pnode_d(x_p)
            x_p = torch.reshape(x_p, (-1, self.node_hidden_dim))
        else:
            x_p = torch.reshape(x_p, (-1, self.pnode_dim))
        
        # transfer feature nodes
        if self.fnode_nn:
            x_f = torch.reshape(x_f, (-1, self.fnode_num))
            x_f = self.fnode_d(x_f)
            x_f = torch.reshape(x_f, (-1, self.node_hidden_dim))
        else:
            x_f = torch.reshape(x_f, (-1, 1))

        for i in range(self.gcn_layer_num):
            x_p = self.gconvs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.gconvs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.gcn_layer_num - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, self.pnode_num))
        
        for i in range(self.cnn_layer_num):
            x = self.convs[i](x)
            x = F.relu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.other_feature_dim:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(torch.cat([x, other_feature], 1))
            x = F.relu(x)
            x = self.d3(x)
            out = F.softmax(x, dim=1)

        else:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(x)
            out = F.softmax(x, dim=1)

        return out

def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=20, random_seed=111, val_split=0.1, weighted_sampling=True, model_name="GCN_model.pt", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in test_list]
    testset = [dataset[i] for i in data_list if i in test_list]
    
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100/label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size,follow_batch=['x_src', 'x_dst'], sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, follow_batch=['x_src', 'x_dst'])
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, follow_batch=['x_src', 'x_dst'])

    # build model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train
    old_test_acc = 0
    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0
        
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            label = batch.y
            
            # forward + backprop + loss
            pred = model(batch)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            
            # update model params
            optimizer.step()
            
            training_running_loss += loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == label).type(torch.float).mean().item()
        
        # test accuracy
        test_acc = evaluation(test_loader, model, device)
        if test_acc > old_test_acc:
            torch.save(model, model_name)
            old_test_acc = test_acc
        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}".format(epoch, training_running_loss/(i+1), train_acc/(i+1), test_acc))

    return model


def evaluation(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()
    
    total = len(loader.dataset)
    acc = correct / total

    return acc

def test(fasta_file, model_name="GCN_model.pt", feature_file=None, output_file="test_output.txt", thread=10, K=3, d=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data = Biodata(fasta_file=fasta_file, feature_file=feature_file, K=K, d=d)
    testset = data.encode(thread=thread)
    model = torch.load(model_name, map_location=device)
    loader = DataLoader(testset, batch_size=len(testset), shuffle=False, follow_batch=['x_src', 'x_dst'])
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
    pred = pred.cpu().numpy()
    f = open(output_file, "w")
    for each in pred:
        f.write(str(each) + "\n")
    f.close()


####################contribution score##############################
def f(n):
    base = ["A", "C", "G", "T"]
    string = ""
    while len(string)<6:
        s = n // 4
        y = n % 4
        string = string + base[y]
        n = s
    string = string[::-1]
    return string


def contribution_score(model, dataset, new_dataset, label, device):
    dataset_loader = DataLoader(dataset, batch_size=100,follow_batch=['x_src', 'x_dst'])
    new_dataset_loader = DataLoader(new_dataset, batch_size=100, follow_batch=['x_src', 'x_dst'])
    model.eval()
    data_pred_list = []
    for data in dataset_loader:
        with torch.no_grad():
            data = data.to(device)
            data_pred = model(data)
            data_pred_list.extend(data_pred[:, label].cpu().numpy())

    new_data_pred_list = []
    for data in new_dataset_loader:
        with torch.no_grad():
            data = data.to(device)
            new_data_pred = model(data)
            new_data_pred_list.extend(new_data_pred[:, label].cpu().numpy())
    score = np.average(np.abs(np.array(data_pred_list) - np.array(new_data_pred_list)))

    return score


def pattern_contribution_score(fasta_file, label_file, target_label=1, model_name="GCN_model.pt", feature_file=None, output_file="pattern_contribution_score.txt", thread=10, K=3, d=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = torch.load(model_name, map_location=device)
    data = Biodata(fasta_file=fasta_file, label_file=label_file, feature_file=feature_file, K=K, d=d)
    dataset = data.encode(thread=thread)
    dataset = [data for data in dataset if data.y==target_label]
    score_list = []
    patterns = [f(i) for i in range(4096)]
    f_out = open(output_file, "w")
    for n in range(4096):
        print("Calculating the contribution score for %s..."%(patterns[n]))
        new_dataset = copy.deepcopy(dataset)
        for i in range(len(new_dataset)):
            new_dataset[i].x_dst[n] = torch.zeros((d))
            new_dataset[i].x_src[n // 64] = torch.zeros((1))
            new_dataset[i].x_src[n % 64] = torch.zeros((1))
        score = contribution_score(model, dataset, new_dataset, target_label, device)
        score_list.append(score)
        f_out.write(patterns[n] + "\t" + str(score) + "\n")
    f_out.close()

    return score_list


def motif_transfer_list(motif):
    base_dic = {"A":["0"], "C":["1"], "G":["2"], "T":["3"], "W":["0", "3"], "S":["1", "2"], "M":["0", "1"], "K":["2", "3"], "R":["0", "2"], "Y":["1", "3"], "B":["1", "2", "3"], "D":["0", "2", "3"], "H":["0", "1", "3"], "V":["0", "1", "2"], "N":["0", "1", "2", "3"]}
    old_seq_list = [""]
    for base in motif:
        seq_list = []
        for num in base_dic[base]:
            seq_list += [s + num for s in old_seq_list]
        old_seq_list = seq_list
    del old_seq_list

    num_list = []
    for seq in seq_list:
        for dis in [0, 1, 2]:
            for i in range(0, len(seq)-6-dis+1):
                num = int(seq[i:i+3] + seq[i+3+dis:i+6+dis], 4)
                if num not in num_list:
                    num_list.append(num)

    return num_list


def motif_contribution_score(fasta_file, label_file, motif, target_label=1, model_name="GCN_model.pt", feature_file=None, thread=10, K=3, d=3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = torch.load(model_name, map_location=device)
    data = Biodata(fasta_file=fasta_file, label_file=label_file, feature_file=feature_file, K=K, d=d)
    dataset = data.encode(thread=thread)
    dataset = [data for data in dataset if data.y==target_label]
    num_list = motif_transfer_list(motif)
    new_dataset = copy.deepcopy(dataset)
    for n in num_list:
        for i in range(len(new_dataset)):
            new_dataset[i].x_dst[n] = torch.zeros((d))
            new_dataset[i].x_src[n // 64] = torch.zeros((1))
            new_dataset[i].x_src[n % 64] = torch.zeros((1))
    score = contribution_score(model, dataset, new_dataset, target_label, device)

    return score


def pattern_group_contribution_score(fasta_file, label_file, score_list, target_label=1, d=3, figure_dic="pattern_group_scores"):
    if not os.path.exists(figure_dic):
        os.mkdir(figure_dic)
    dis_dic = {1:1, 2:2, 3:3, 4:5, 5:9, 6:17, 7:33, 8:65}
    # read patterns and corresponding MAE
    patterns = [f(i) for i in range(4096)]
    pat2score = dict(zip(patterns, score_list))
    fasta_sequences = SeqIO.parse(fasta_file, 'fasta')
    labels = np.loadtxt(label_file)
    seq_types = dict(zip([fasta.id for fasta in fasta_sequences], list(labels)))
    fasta_sequences = SeqIO.parse(fasta_file, 'fasta')
    sequences = [(fasta.id, fasta.seq) for fasta in fasta_sequences]
    neg_seqs = [seq for (sid, seq) in sequences if seq_types[sid]!=target_label]
    pos_seqs = [seq for (sid, seq) in sequences if seq_types[sid]==target_label]
    
    print("Calculating the pattern occurence. It may take a long time...")
    M = np.zeros((len(sequences), len(patterns)))
    for i in range(len(patterns)):
        p = patterns[i]
        ph = p[:3]
        pe = p[-3:]
        for l in range(dis_dic[d]):
            wildcardstr = '.' * l
            for j in range(len(sequences)):
                seq = str(sequences[j][1])
                matches = [m.start() for m in re.finditer('(?='+ ph + wildcardstr + pe +')', seq)]
                M[j,i] += len(matches)
    np.save('%s/seq_pattern_matrix'%figure_dic, M)
    print("Calculation finished...")
    
    M = np.load('%s/seq_pattern_matrix.npy'%figure_dic)

    MM = M.copy()
    for i in range(M.shape[0]):
        threshold = np.quantile(M[i,:], 0.9)
        MM[i,np.where(M[i,:]< threshold)[0]] = 0
        MM[i,np.where(M[i,:]>=threshold)[0]] = 1

    neg_rows = [i for i in range(len(sequences)) if seq_types[sequences[i][0]]!=target_label]
    pos_rows = [i for i in range(len(sequences)) if seq_types[sequences[i][0]]==target_label]

    MM_pos = MM[pos_rows,:]
    MM_neg = MM[neg_rows,:]

    pat2posmatches = dict([(patterns[i], MM_pos[:,i].sum()) for i in range(len(patterns))])
    pat2negmatches = dict([(patterns[i], MM_neg[:,i].sum()) for i in range(len(patterns))])

    scores = [pat2score[p] for p in patterns]
    pos_matches = [pat2posmatches[p] for p in patterns]
    neg_matches = [pat2negmatches[p] for p in patterns]
    ratios = [pat2posmatches[p]/pat2negmatches[p] for p in patterns]

    sc = plt.scatter(pos_matches, neg_matches, marker='.', c=scores, cmap='Greens')
    plt.colorbar(sc)
    plt.xlabel("# of positive matches", {'size': 16})
    plt.ylabel("# of negative matches", {'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("%s/pos_neg_match_single.png"%figure_dic, dpi=600)
    plt.close()

    plt.scatter(scores, ratios)
    plt.xlabel("Contribution score", {'size': 16})
    plt.ylabel("Pos/neg matches", {'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("%s/pos_neg_score_single.png"%figure_dic, dpi=600)
    plt.close()

    import editdistance

    min_distinct_pat_distance = 2 # At least edit distance 2 from other representative pattern
    min_same_group_distance = 1 # At least edit distance 1 from representative pattern to belong to group

    repr_pats = []

    # test from the highest scoring pattern to the lowest so that the representative pattern has the highest score
    sorted_patterns = sorted([(p, s) for (p, s) in zip(patterns, scores)], reverse=True, key=lambda x:x[1])
    rejected = dict((p, False) for p in patterns)

    _patterns = [p for (p, s) in sorted_patterns]
    for i in range(len(_patterns)):
        if rejected[_patterns[i]]:
            continue
        repr_pats.append(_patterns[i])
        for j in range(i+1, len(_patterns)):
            if rejected[_patterns[j]]:
                continue
            if not editdistance.eval(_patterns[i], _patterns[j]) >= min_distinct_pat_distance :
                rejected[_patterns[j]] = True

    pattern_groups = dict()

    for p in repr_pats:
        group = []
        for _p in patterns:
            if editdistance.eval(p, _p) <= min_same_group_distance:
                group.append(_p)
        pattern_groups[p] = group

    print("Organized into", len(repr_pats), "groups")
    for p in repr_pats[:3]:
        print("representative pat =", p, " members =", pattern_groups[p][:3], "...")
    print("...")

    patgrp2posmatches = dict()
    patgrp2negmatches = dict()

    for p in repr_pats:
        patgrp2posmatches[p] = np.array([pat2posmatches[_p] for _p in pattern_groups[p]]).sum()/len(pattern_groups[p])
        patgrp2negmatches[p] = np.array([pat2negmatches[_p] for _p in pattern_groups[p]]).sum()/len(pattern_groups[p])

    scores = [pat2score[p] for p in repr_pats]
    pos_matches = [patgrp2posmatches[p] for p in repr_pats]
    neg_matches = [patgrp2negmatches[p] for p in repr_pats]
    ratios = [patgrp2posmatches[p]/patgrp2negmatches[p] for p in repr_pats]

    sc = plt.scatter(pos_matches, neg_matches, marker='.', c=scores, cmap='Greens')
    plt.colorbar(sc)
    plt.xlabel("# of positive matches", {'size': 16})
    plt.ylabel("# of negative matches", {'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("%s/pos_neg_match_grouped.png"%figure_dic, dpi=600)
    plt.close()

    plt.scatter(scores, ratios)
    plt.xlabel("Contribution score", {'size': 16})
    plt.ylabel("Pos/neg matches", {'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("%s/pos_neg_score_grouped.png"%figure_dic, dpi=600)
    plt.close()


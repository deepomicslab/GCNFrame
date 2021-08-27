from GNNFrame import Biodata, GNNmodel
import torch

data = Biodata.Biodata(fasta_file="/home/ruohawang2/dataset/Phage_nature2017/nature_2017.fasta", label_file="/home/ruohawang2/dataset/Phage_nature2017/kmer_encoding/lifestyle_label.txt", feature_file="/home/ruohawang2/06.GNN_framework/phage_lifestyle/dataset/nature_2017/CDD_feature/CDD_protein_feature.txt")
dataset = data.encode(thread=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNmodel.model(label_num=2, other_feature_dim=206).to(device)
GNNmodel.train(dataset, model, weighted_sampling=False)

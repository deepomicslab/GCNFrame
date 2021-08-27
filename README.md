# a GNN framework for genomics
This is a python package for genomics study with a GNN(Graph Neural Networks) framework.

# Getting started

## Prerequisite
+ cython
+ numpy
+ Biopython
+ pytorch 1.9.0
+ pytorch\_geometric 1.7.0

## Install
```shell
git lfs clone https://github.com/deepomicslab/GNNFrame.git
cd GNNFrame
python setup.py build_ext --inplace
```

## examples
The framework makes it easy to train your customized models with a few lines of codes.
```Python
# This is an example to train a two-classes model.
from GNNFrame import Biodata, GNNmodel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Biodata.Biodata(fasta_file="/home/ruohawang2/dataset/Phage_nature2017/nature_2017.fasta", 
        label_file="/home/ruohawang2/dataset/Phage_nature2017/kmer_encoding/lifestyle_label.txt",
        feature_file="/home/ruohawang2/06.GNN_framework/phage_lifestyle/dataset/nature_2017/CDD_feature/CDD_protein_feature.txt")
dataset = data.encode(thread=20)
model = GNNmodel.model(label_num=2, other_feature_dim=206).to(device)
GNNmodel.train(dataset, model, weighted_sampling=True)
```
The output is shown bellow:
```Output
Encoding sequences...
Epoch 0| Loss: 0.6335| Train accuracy: 0.7480| Validation accuracy: 0.8839
Epoch 1| Loss: 0.5605| Train accuracy: 0.8165| Validation accuracy: 0.7032
Epoch 2| Loss: 0.5042| Train accuracy: 0.8469| Validation accuracy: 0.8065
Epoch 3| Loss: 0.4873| Train accuracy: 0.8344| Validation accuracy: 0.7677
Epoch 4| Loss: 0.4559| Train accuracy: 0.8703| Validation accuracy: 0.8194
Epoch 5| Loss: 0.4533| Train accuracy: 0.8763| Validation accuracy: 0.7806
Epoch 6| Loss: 0.4372| Train accuracy: 0.8931| Validation accuracy: 0.8387
Epoch 7| Loss: 0.4409| Train accuracy: 0.8842| Validation accuracy: 0.8581
Epoch 8| Loss: 0.4357| Train accuracy: 0.8858| Validation accuracy: 0.8516
Epoch 9| Loss: 0.4314| Train accuracy: 0.8987| Validation accuracy: 0.8387
Epoch 10| Loss: 0.4246| Train accuracy: 0.8992| Validation accuracy: 0.8581
Epoch 11| Loss: 0.4085| Train accuracy: 0.9180| Validation accuracy: 0.8839
Epoch 12| Loss: 0.4071| Train accuracy: 0.9290| Validation accuracy: 0.8903
Epoch 13| Loss: 0.4095| Train accuracy: 0.9170| Validation accuracy: 0.8839
Epoch 14| Loss: 0.4019| Train accuracy: 0.9241| Validation accuracy: 0.8839
Epoch 15| Loss: 0.3960| Train accuracy: 0.9342| Validation accuracy: 0.9161
```
The model with best validation accuracy will be saved as GNN\_model.pt


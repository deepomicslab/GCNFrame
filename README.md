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

data = Biodata.Biodata(fasta_file="data/nature_2017.fasta", label_file="data/lifestyle_label.txt", feature_file="data/CDD_protein_feature.txt")
dataset = data.encode(thread=20)
model = GNNmodel.model(label_num=2, other_feature_dim=206).to(device)
GNNmodel.train(dataset, model, weighted_sampling=True)
``` 

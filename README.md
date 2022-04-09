# a GP-GCN framework for genomics
This is a python package for genomics study with a GP-GCN (Gapped Pattern Graph Convolutional Networks) framework.

# Getting started

## Prerequisite
+ cython
+ numpy
+ Biopython
+ pytorch 1.7.1
+ pytorch\_geometric 1.7.0

## Install
```shell
pip install GCNFrame
```
Or
```shell
git clone https://github.com/deepomicslab/GCNFrame.git
cd GCNFrame
python setup.py build_ext --inplace
```

## Examples
The framework makes it easy to train your customized models with a few lines of codes. The example data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1VtNE0Pk0fM9b3N_WfbKc1x2IqDu8Ny02?usp=sharing).
```Python
# This is an example to train a two-classes model.
from GCNFrame import Biodata, GCNmodel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Biodata(fasta_file="example_data/nature_2017.fasta", 
        label_file="example_data/lifestyle_label.txt",
        feature_file="example_data/CDD_protein_feature.txt")
dataset = data.encode(thread=20)
model = GCNmodel.model(label_num=2, other_feature_dim=206).to(device)
GCNmodel.train(dataset, model, weighted_sampling=True)
GCNmodel.test("GCN_model.pt", fasta_file="example_data/nature_2017.fasta", feature_file="example_data/CDD_protein_feature.txt")
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
The model with best validation accuracy will be saved as GCN\_model.pt

## Parameters
###  ```class Biodata.Biodata```
> + ```fasta_file```: The DNA sequences used for training and evaluation in fasta format.
> + ```label_file```: The labels for the DNA sequences for training and evaluation (should have the same order as fasta_file).
> + ```feature_file```: Other features (like gene density) for the DNA sequences for training and evaluation (should have the same order as fasta_file) (default=None).
> + ```K```: The length of K-mer for encoding (default=3). 
> + ```d```: The number of spaced distance used for encoding (default=3).
> + ```thread```: The number of thread used for encoding (default=10).

###  ```class GCNmodel.model```
> + ```label_num```: The number of labels.
> + ```other_feature_dim```: The dimension for other features, 0 if not available.
> + ```K```: The length of K-mer for encoding (default=3). 
> + ```d```: The number of spaced distance used for encoding (default=3).
> + ```node_hidden_dim```: The size for kmer nodes after transformation(default=3).
> + ```gcn_dim```: The size of output of SAGEConv (default=128).
> + ```gcn_layer_num```: The number of SAGEConv layers (default=4).
> + ```cnn_dim```: The size of output of convolutional layers (default=64).
> + ```cnn_layer_num```: The number of convolutional layers (default=3).
> + ```cnn_kernel_size```: The kernel size of convolutional layers (default=8).
> + ```fc_dim```: The number of neurons for the fully connected layers (default=100).
> + ```dropout_rate```: The dropout rate (default=0.2).
> + ```pnode_nn```: Whether transform primary nodes (default=True).
> + ```fnode_nn```: Whether transform target nodes (default=True).

###  ```GCNmodel.train```
> + ```learning_rate```: The learning rate for training (default=1e-4). 
> + ```batch_size```: The batch_size for training (default=64).
> + ```epoch_n```: The number of training epoches (default=20).
> + ```random_seed```: The random seed for train-validation split (default=111).
> + ```val_split```: The validation size (default=0.1).
> + ```weighted_sampling```: Whether use weighted sampling for training (default=True).
> + ```model_name```: The saved model name (default="GCN_model.pt").

###  ```GCNmodel.test```
> + ```fasta_file```: The DNA sequences used for test in fasta format. 
> + ```model_name```: The saved model name (default="GCN_model.pt").
> + ```feature_file```: Other features (like gene density) for the DNA sequences for test (should have the same order as fasta_file) (default=None).
> + ```output_file```: The output file name (default="test_output.txt").
> + ```thread```: The number of thread used for encoding (default=10).
> + ```K```: The length of K-mer for encoding (default=3). 
> + ```d```: The number of spaced distance used for encoding (default=3).

### Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk

# LFEA

Code for the paper "Linear Functionality Equivalence Attack against Deep Neural Network Watermarks and a Defense Method". 

PyTorch>=1.10.2
CUDA required

#### ./data : Where the dataset is stored by default.
#### ./checkpoint : Where the DNN checkpoints are stored by default.
#### Autoencoder.py : The implementation for LFEA and NeuronMap on an autoencoder.
#### LeNet.py : The implementation for LFEA and NeuronMap on an LeNet.
#### ResNet.py : The implementation for LFEA and NeuronMap on an ResNet.

For illustration, the watermarking schemes implemented in respective scripts are their simplified version. 

As an example, to test the original paper's justifications on the autoencoder DNN, run 

`python Autoencoder.py`

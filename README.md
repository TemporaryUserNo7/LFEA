# LFEA

Code for the paper "Linear Functionality Equivalence Attack against Deep Neural Network Watermarks and a Defense Method". 

PyTorch>=1.10.2
CUDA required

## File structure
1. ./data : Where the dataset is stored by default.
2. ./checkpoint : Where the DNN checkpoints are stored by default.
3. Autoencoder.py : The implementation for LFEA and NeuronMap on an autoencoder.
4. LeNet.py : The implementation for LFEA and NeuronMap on an LeNet.
5. ResNet.py : The implementation for LFEA and NeuronMap on an ResNet.

For illustration, the watermarking schemes implemented in respective scripts are their simplified version. 

As an example, to test the original paper's justifications on the autoencoder DNN, run 

`python Autoencoder.py`

The owner's DNN `AE1` is firstly trained.

```python
AE1=MyAE()
device=torch.device("cuda:1")
AE1=AE1.to(device)
for epoch in range(E):
    train(AE1,train_loader,device,True)
```

Then LFEA is applied on its layers where the watermark information would be involved. 

```python
Q51=GenQ(328)
Q52=GenQ(75)
AE5t=MaskAE12(AE1,Q51)
AE5=MaskAE23(AE5t,Q52)
```

In which `GenQ` produces $\phi\in\Phi^{+}$ and `MaskAEXX` conducts LFEA. 

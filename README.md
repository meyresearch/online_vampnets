# Online VAMPnets
This project will aim to train a VAMPnets model in one epoch i.e., online learning. 

The key reference (at the moment at least) is: 

Sahoo, D. et al. (2017) ‘Online Deep Learning: Learning Deep Neural Networks on the Fly’, arXiv:1711.03705 [cs] [Preprint]. Available at: [http://arxiv.org/abs/1711.03705](http://arxiv.org/abs/1711.03705) (Accessed: 14 May 2021).



## Installation

Assuming a linux OS with CUDA drivers 11.3 - see [Pytorch.org](pytorch.org) for details for other distros/CUDA versions.

```
conda create -n onlinevampnets python==3.9 -y
conda install pytorch cudatoolkit=11.3 -c pytorch -y 
conda install deeptime mdshare matplotlib tqdm ipykernel -y
```

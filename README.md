# HotFlip-CNN-pytorch
code for MS thesis "White-Box Adversarial Attacks on classification in NLP"

This code is based on https://github.com/srviest/char-cnn-text-classification-pytorch

**HotFlip** – algorithm for white-box adversarial attacks. In this project it is applied on character level. 
Implementation of two approaches:
1. [Greedy strategy](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/data_loader_hotflip_greedy.py)
2. [Beam search](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/data_loader_hotflip_beam.py)

For **determinantal point processes** modification of HotFlip algorithm I used [Fast Greedy MAP Inference](https://github.com/laming-chen/fast-map-dpp).

For **transferability** comparison [DeepWordBug](https://github.com/QData/deepWordBug) algorithm with replace one strategy was implemented. 
It was improved using [local beam search strategy](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/locality_property_beam.ipynb): [DeepWordBug data_loader](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/data_loader_deepwordbug.py). 

Usage examples are given in [test.ipynb](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/test.ipynb)

Models for experiments can be downloaded from [storage](https://yadi.sk/d/3IyeSiPqk5b8XA):
1. [CharCNN](https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/model.py)
2. [CharCNN small](https://github.com/Fadeich/HotFlip-CNN-pytorch/blob/master/model_small.py)
3. [SWCNN](https://github.com/doxawang/char-cnn-text-classification-pytorch/blob/master/model/SWCNN.py)
4. CharCNN adv – model with adversarial training for HotFlip attacks
5. CharCNN jac – model trained with [jacobian regularization](https://github.com/facebookresearch/jacobian_regularizer)

## Reference
* Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)
* Ebrahimi, Javid, Anyi Rao, Daniel Lowd and Dejing Dou. “HotFlip: White-Box Adversarial Examples for Text Classification.” ACL (2018).
* Chen, Laming, Guoxin Zhang and Eric Zhou. “Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity.” NeurIPS (2018).
* Gao, Ji, Jack Lanchantin, Mary Lou Soffa and Yanjun Qi. “Black-Box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers.” 2018 IEEE Security and Privacy Workshops (SPW) (2018): 50-56.
* Judy Hoffman, Daniel A. Roberts, and Sho Yaida, "Robust Learning with Jacobian Regularization," 2019. [arxiv:1908.02729 [stat.ML]](https://arxiv.org/abs/1908.02729)

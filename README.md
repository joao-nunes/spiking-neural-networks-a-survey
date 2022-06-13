# Spiking Neural Networks: A Survey

## About

Oficial repository of the paper "Spiking Neural Networks: A Survey" by João Nunes, Marcelo Carvalho, Diogo Carneiro, and Jaime S. Cardoso.
If you use this repository, please cite:

J. D. Nunes, M. Carvalho, D. Carneiro and J. S. Cardoso, "Spiking Neural Networks: A Survey," in IEEE Access, [doi:10.1109/ACCESS.2022.3179968](10.1109/ACCESS.2022.3179968).

BibTeX citation:

@ARTICLE{9787485,
  author={Nunes, Jo&#x00E3;o D. and Carvalho, Marcelo and Carneiro, Diogo and Cardoso, Jaime S.},
  journal={IEEE Access}, 
  title={Spiking Neural Networks: A Survey}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3179968}}

## Abstract

The field of Deep Learning (DL) has seen a remarkable series of developments with increasingly accurate and robust algorithms. However, the increase in performance has been accompanied by an increase in the parameters, complexity, and training and inference time of the models, which means that we are rapidly reaching a point where DL may no longer be feasible. On the other hand, some specific applications need to be carefully considered when developing DL models due to hardware limitations or power requirements. In this context, there is a growing interest in efficient DL algorithms, with Spiking Neural Networks (SNNs) being one of the most promising paradigms. Due to the inherent asynchrony and sparseness of spike trains, these types of networks have the potential to reduce power consumption while maintaining relatively good performance. This is attractive for efficient DL and if successful, could replace traditional Artificial Neural Networks (ANNs) in many applications. However, despite significant progress, the performance of SNNs on benchmark datasets is often lower than that of traditional ANNs. Moreover, due to the non-differentiable nature of their activation functions, it is difficult to train SNNs with direct backpropagation, so appropriate training strategies must be found. Nevertheless, significant efforts have been made to develop competitive models. This survey covers the main ideas behind SNNs and reviews recent trends in learning rules and network architectures, with a particular focus on biologically inspired strategies. It also provides some practical considerations of state-of-the-art SNNs and discusses relevant research opportunities.

## Credits and Aknowledgments

This work implements: ["Unsupervised Learning of Digit Recognition using Spike Timing Dependent Plasticity"](https://doi.org/10.3389/fncom.2015.00099) [1] as suggested in [BindsNET examples directory](https://github.com/BindsNET/bindsnet/blob/master/examples/mnist/eth_mnist.py).

An honorable mention to the authors of [Enabling spike-based backpropagation for training deep neural network architectures](https://doi.org/10.3389/fnins.2020.00119) [2] as we built on top of their ideas regarding spiking neural networks trained with surrogate gradients.
 
We thank the authors of the SNNs *Python* packages we used in the experiments:

-- [BindsNET](https://github.com/BindsNET) [3]

-- [snnTorch](https://github.com/jeshraghian/snntorch) [4]

[1]: P. Diehl and M. Cook, “Unsupervised learning of digit recognition using spike-timing-dependent plasticity,” Frontiers in Computational Neuroscience, vol. 9, 2015. [Online]. Available: https://www.frontiersin.org/article/10.3389/fncom.2015.00099

[2]: C. Lee, S. S. Sarwar, P. Panda, G. Srinivasan, and K. Roy, “Enabling spike-based backpropagation for training deep neural network architectures,” Frontiers in Neuroscience, vol. 14, 2020. [Online]. Available: https://www.frontiersin.org/article/10.3389/fnins.2020.00119

[3]: H. Hazan, D. J. Saunders, H. Khan, D. Patel, D. T. Sanghavi, H. T. Siegelmann, and R. Kozma, “Bindsnet: A machine learning oriented spiking neural networks library in python,” Frontiers in Neuroinformatics, vol. 12, p. 89, 2018. [Online]. Available: https://www.frontiersin.org/article/10.3389/fninf.2018.00089

[4]: Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894, September 2021.  [Online]. Available: https://arxiv.org/abs/2109.12894

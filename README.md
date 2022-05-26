# spiking-neural-networks-a-survey

## Abstract

The field of Deep Learning (DL) has seen a remarkable series of developments with increasingly accurate
and robust algorithms. However, the increase in performance has been accompanied by an increase in the
parameters, complexity, and training and inference time of the models, which means that we are rapidly
reaching a point where DL may no longer be feasible. On the other hand, some specific applications need to
be carefully considered when developing DL models due to hardware limitations or power requirements. In
this context, there is a growing interest in efficient DL algorithms, with Spiking Neural Networks (SNNs)
being one of the most promising paradigms. Due to the inherent asynchrony and sparseness of spike trains,
these types of networks have the potential to reduce power consumption while maintaining relatively good
performance. This is attractive for efficient DL and if successful, could replace traditional Artificial Neural
Networks (ANNs) in many applications. However, despite significant progress, the performance of SNNs
on benchmark datasets is often lower than that of traditional ANNs. Moreover, due to the non-differentiable
nature of their activation functions, it is difficult to train SNNs with direct backpropagation, so appropriate
training strategies must be found. Nevertheless, significant efforts have been made to develop competitive
models. This survey covers the main ideas behind SNNs and reviews recent trends in learning rules and
network architectures, with a particular focus on biologically inspired strategies. It also provides some
practical considerations of state-of-the-art SNNs and discusses relevant research opportunities.

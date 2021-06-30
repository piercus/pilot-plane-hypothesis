## The Measurably Trainable Module (MTM)

The MTM is a neural network module built to follow the training progress and compare the learning between top-level components of a network.
MTM is inspired by Wide-and-Deep network and by the idea that "Wide Networks" are trained faster than "Deep network".

We consider a block generator function like
(depth) |-> B(depth)

In practice for our experiments, we use following blocks :
* Multi-layer convolutionnal network 
* Multi Layer Perceptron layer generator of 35

A MTM layer is composed of multiple blocks, called Bi (i from 0 to N).
A block Bi 

A basic MTM is a wide and deep-network with following differences : 
* a merge layer that adds wide part with deep-part
* A dropout layer that alternate the training between the wide part and the deep part
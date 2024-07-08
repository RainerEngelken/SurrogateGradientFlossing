# SurrogateGradientFlossing
This repository contains the implementation code for the manuscript:  <br>
 __Improving Surrogate Gradient Learning in Spiking Neural Networks Using Dynamical Systems Theory__ <br>


## Overview
We analyze and optimize gradients of binary and spiking recurrent neural networks using concepts from dynamical systems theory. Specifically, we show that surrogate gradient training can be improved by pushing surrogate Lyapunov exponents to zero during or before training.

## Installation

#### Prerequisites
- Download [Julia](https://julialang.org/downloads/) 

#### Dependencies
- Julia (1.6)
- Flux, BackwardsLinalg

## Getting started
To install the required packages, run the following in the julia REPL after installing Julia:

```
using Pkg

for pkg in ["Flux", "BackwardsLinalg"]
    Pkg.add(pkg)
end
```

For example, to train a spiking neural network on the delayed XOR task, run:
```
include("SurrogateGradientFlossing_ExampleCode.jl")
# setting parameters:
N, E, Ef, Ei, Ep, Ni, B, S, T, Tp, Ti, sIC, sIn, sNet, sONS, lr, b1, b2, IC, g, gbar, I1, delay, wsS, wsM, wrS, wrM, bS, bM, nLE, task, intype, Lwnt=
80, 3001, 100, 500, 500, 2, 16, 1, 300, 55, 300, 1,1,1,1, 0.001f0, 0.9, 0.999, 1, 1.0, 0.0, 1.0,10, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0.1f0, 0.0f0,75, -1, 3, 0.0

trainSRNNflossing(N, E, Ef, Ei, Ep, Ni, B, S, T, Tp, Ti, sIC, sIn, sNet, sONS, lr, b1, b2, IC, g, gbar, I1, delay, wsS, wsM, wrS, wrM, bS, bM, nLE, task, intype, Lwnt)
```

## Repository Overview
_GradientFlossing_ExampleCode.jl_:\
Example scripts for training networks with gradient flossing before training, with gradient flossing before and during training and without gradient flossing.


_GradientFlossing_XOR.jl_:\
Generates input and target output for copy task and delayed XOR task.

<!---

runOneStimulus.jl trains an RNN on tracking one OU-signal showing that the network becomes more tightly balanced over training epochs.\
runTwoStimuli.jl trains an RNN on two OU-signal stimuli showing that the network becomes more tightly balanced over training epochs and breaks up into two weakly-connected subnetworks.\
runTheeStimuli.jl trains an RNN on two OU-signal stimuli showing that the network becomes more tightly balanced over training epochs and breaks up into three weakly-connected subnetworks.\
![Training RNN on two signals leads to balanced subpopulations](/figures/S=2.svg?raw=true "balanced subnetworks emerge  after runTheeStimuli.jl")
-->


<!---

### Training dynamics of eigenvalues:
Here is a visualization of the recurrent weight matrix and the eigenvalues throughout across training epochs.
![Training dynamics of networks trained on multiple signals shows first tracking of global mean input](eigenvalue_movie_2D_task.gif)
-->


### Implementation details
A full specification of packages used and their versions can be found in _packages.txt_ .\
For learning rates, the default ADAM parameters were used to avoid any impression of fine-tuning.\
All simulations were run on a single CPU and took on the order of minutes to a few hours.



<!---
### figures/
Contains all figures of the main text and the supplement.
-->


<!---
### tex/
Contains the raw text of the main text and the supplement.
-->

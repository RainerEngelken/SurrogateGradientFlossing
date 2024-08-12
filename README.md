# SurrogateGradientFlossing
This repository contains the implementation code for the manuscript:  <br>
 __Using Dynamical Systems Theory to Improve Surrogate Gradient Learning in Spiking Neural Networks__ <br>


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

## Additional results:
We here provide additional results on surrogate gradient flossing in binary RNNs. The following figures shows that we can manipulate one or several surrogate Lyapunov exponets in binary networks:

**Figure 1: Surrogate gradient flossing** regularizes *surrogate Lyapunov exponents* and facilitates gradient signal propagation in binary neural networks.

![**Surrogate gradient flossing** regularizes *surrogate Lyapunov exponents* and facilitates gradient signal propagation in binary neural networks. **A)** The first *surrogate Lyapunov exponent* of a recurrent binary network plotted as a function of training epochs for different surrogate sharpness $g$. The square of the first *surrogate Lyapunov exponent* is minimized using gradient descent. **B)** *Surrogate Lyapunov spectrum* of a recurrent binary network after different numbers of Lyapunov exponents $k$ have been driven towards zero via *surrogate gradient flossing* for $k\in\{1,16,32\}. The gray lines show the *surrogate Lyapunov spectra* before *surrogate gradient flossing*. Parameters: network size $N=80$, $g=1$ for **B**. Input as in Fig. 3. The thin semitransparent lines in **A** and **B** indicate nine network realizations; the full lines are their average.](/figures/bf_fig02a.svg)


**A)** The first *surrogate Lyapunov exponent* of a recurrent binary network plotted as a function of training epochs for different surrogate sharpness $g$. The square of the first *surrogate Lyapunov exponent* is minimized using gradient descent.

**B)** *Surrogate Lyapunov spectrum* of a recurrent binary network after different numbers of Lyapunov exponents $k$ have been driven towards zero via *surrogate gradient flossing* for $k\in\{1,16,32\}$. The gray lines show the *surrogate Lyapunov spectra* before *surrogate gradient flossing*. Parameters: network size $N=80$, $g=1$ for **B**. Input as in Fig. 3. The thin semitransparent lines in **A** and **B** indicate nine network realizations; the full lines are their average.

The following figure shows that surrogate gradient flossing improves training in binary RNNs:

**Figure 2: Surrogate gradient flossing improves binary RNN training.**

![**Gradient flossing** improves binary RNN training. **A)** Test accuracy for binary RNNs trained on the delayed temporal binary XOR task $y_t=x_{t-d/2} \oplus x_{t-d}$ with *adaptive gradient flossing* during training (orange) and without *gradient flossing* (blue) for $d=18$. Solid lines are the median across 9 network realizations, and individual network realizations are shown in transparent fine lines. **B)** Mean final test accuracy as a function of task difficulty (delay $d$) for delayed XOR task. **C)** Gradient norm with respect to initial network state $\mathbf{h}_0$. **D)** Gradient norm with respect to initial network state as a function of temporal task complexity $T$ averaged over training epochs.](/figures/bf_fig03a.svg)


**A)** Test accuracy for binary RNNs trained on the delayed temporal binary XOR task $y_t=x_{t-d/2} \oplus x_{t-d}$ with *adaptive gradient flossing* during training (orange) and without *gradient flossing* (blue) for $d=18$. Solid lines are the median across 9 network realizations, and individual network realizations are shown in transparent fine lines.

**B)** Mean final test accuracy as a function of task difficulty (delay $d$) for delayed XOR task.

**C)** Gradient norm with respect to initial network state $\mathbf{h}_0$.

**D)** Gradient norm with respect to initial network state as a function of temporal task complexity averaged over training epochs.


<!---
### figures/
Contains all figures of the main text and the supplement.
-->


<!---
### tex/
Contains the raw text of the main text and the supplement.
-->

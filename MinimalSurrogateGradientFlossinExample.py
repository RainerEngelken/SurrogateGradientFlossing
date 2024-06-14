import torch
import torch.optim as optim
import numpy as np


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 10.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply



def sgprime(x, g):
    return 1 / (g * torch.abs(x) + 1)**2

def manual_jacobian_clif15NoReset(mem, syn, x, alpha, beta, v1, iext, gj):
    n = len(mem)
    mthr = mem - 1
    out_derivative = sgprime(mthr, g=gj)
    
    eye_n = torch.eye(n, dtype=torch.float32, device=mem.device)
    du_dmem = beta * eye_n + v1 @ torch.diag(out_derivative * (1 - beta))
    J_dmem_dmem = du_dmem
    J_dsyn_dmem = v1 @ torch.diag(out_derivative)
    J_dmem_dsyn = (1 - beta) * alpha * eye_n
    J_dsyn_dsyn = alpha * eye_n
    
    # Constructing full Jacobian using block diagonal matrix
    J_upper = torch.cat([J_dmem_dmem, J_dmem_dsyn], dim=1)
    J_lower = torch.cat([J_dsyn_dmem, J_dsyn_dsyn], dim=1)
    J = torch.cat([J_upper, J_lower], dim=0)
    
    return J

def neuron_update7hardOut(mem, syn, x, gu):
    mthr = mem - 1
    out = spike_fn(mthr)
    rst = out.detach() # like spytorch tutorial: https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial5.ipynb    
    syn = alpha * syn + x + v1 @ out
    
    mem = (beta * mem + (1 - beta) * syn) * (1 - rst)
    
    return mem, syn
    
    

def calculate_lyapunov_spectrum(gForward, gBackward, x_data, nle, Win, v1, g):
    n = nb_hidden
    nx = nb_inputs
    steps = len(x_data)
    ONSstep = 1
    
    torch.manual_seed(seedIC)
    mem = torch.zeros(n, dtype=torch.float32)
    syn = torch.zeros(n, dtype=torch.float32)
    
    torch.manual_seed(seedONS)
    Q, R = torch.linalg.qr(torch.randn(2*n, nle))
    ls = torch.zeros(nle, dtype=torch.float32)
    #lsall = torch.zeros((nle, steps // ONSstep), dtype=torch.float32)  # Preallocate as a matrix
    
    for step in range(steps):
        x = x_data[step]
        D = manual_jacobian_clif15NoReset(mem, syn, Win @ x, alpha, beta, v1, iext, gBackward)        
        #print(f"before D.shape {D.shape}")
        #print(f"before mem.shape{mem.shape}")        
        mem, syn = neuron_update7hardOut(mem, syn, Win @ x, gForward)
        #print(f"D.shape {D.shape}")
        #print(f"mem.shape{mem.shape}")
        Q = D @ Q

        if step % ONSstep == 0 and nle > 0:
            Q, R = torch.linalg.qr(Q)
            ls += torch.log(torch.abs(torch.diag(R))) / ONSstep
            #lsall[:, step // ONSstep] = torch.log(torch.abs(torch.diag(R))) / ONSstep/

    return ls#torch.mean(lsall[:, steps//2//ONSstep:], axis=1)

# Model parameters
tau_mem = 10e-3
tau_syn = 5e-3
time_step = 1e-3
alpha, beta = torch.exp(torch.tensor(-time_step / tau_syn)), torch.exp(torch.tensor(-time_step / tau_mem))

nb_inputs = 100
nb_steps = 1000
nb_hidden = 64
iext = 0

# Initialize parameters to optimize as leaf tensors
Win = torch.randn(nb_hidden, nb_inputs, dtype=torch.float32, requires_grad=True) / torch.sqrt(torch.tensor(nb_inputs, dtype=torch.float32))
Win = Win - torch.mean(Win, dim=1, keepdim=True)
Win = Win.clone().detach().requires_grad_(True)  # Ensure it is a leaf tensor

v1 = torch.randn(nb_hidden, nb_hidden, dtype=torch.float32, requires_grad=True) / torch.sqrt(torch.tensor(nb_hidden, dtype=torch.float32))
v1 = v1 - torch.mean(v1, dim=1, keepdim=True)
v1 = v1.clone().detach().requires_grad_(True)  # Ensure it is a leaf tensor

g = torch.tensor(5.0, dtype=torch.float32, requires_grad=False)

# Generate input data
pSpike = 0.1  # for dt=1e-3 this is 10 Hz input
x_data = [torch.tensor(np.random.rand(nb_inputs) < pSpike, dtype=torch.float32) for _ in range(nb_steps)]

nle = 50#2 * nb_hidden
seedIC = seedONS = 1
subDir = "cLIF_spectrum"

resetSwitch = False

# Optimization setup
#optimizer = optim.SGD([Win, v1, g], lr=1e-3)
optimizer = optim.Adam([Win, v1])#, g])
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
from IPython.display import clear_output

lyapunov_spectrum = calculate_lyapunov_spectrum(1e9, g, x_data[len(x_data)//2:], nle, Win, v1, g)
# Initialize lists to store the loss and Lyapunov spectrum for plotting
losses = []
lyapunov_spectra = []

# Set up the initial plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

lyapunov_line, = ax1.plot([], [], "k", label="Lyapunov Spectrum")
ax1.set_xlabel("Index")
ax1.set_ylabel("Lyapunov Exponent")
ax1.set_title("Lyapunov Spectrum")
ax1.legend()

loss_line, = ax2.plot([], [], "r", label="Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Loss over Epochs")
ax2.legend()

plt.ion()  # Turn on interactive mode
plt.show()
lyapunov_spectrumInitial=lyapunov_spectrum.detach().cpu().numpy()
plt.subplot(121)
plt.plot(lyapunov_spectrumInitial,"r")
0
# Training loop
#Main training loop with profiling

for epoch in range(100):
    torch.manual_seed(epoch)
    x_data = [torch.tensor(np.random.rand(nb_inputs) < pSpike, dtype=torch.float32) for _ in range(nb_steps)]

    optimizer.zero_grad()
    
    lyapunov_spectrum = calculate_lyapunov_spectrum(1e9, g, x_data[len(x_data) // 2:], nle, Win, v1, g)
    
    # Calculate the loss (mean square of the first nle Lyapunov exponents)
    loss = torch.mean(lyapunov_spectrum**2)
    print(f"Epoch {epoch}: Loss = {loss.item()}, Loss dtype = {loss.dtype}")
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Optimization step
    optimizer.step()



    # Store the loss and Lyapunov spectrum
    losses.append(loss.item())
    lyapunov_spectra.append(lyapunov_spectrum.detach().cpu().numpy())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        
        # Update the Lyapunov spectrum plot
        lyapunov_line.set_ydata(lyapunov_spectrum.detach().cpu().numpy())
        lyapunov_line.set_xdata(range(len(lyapunov_spectrum)))
        ax1.relim()
        ax1.autoscale_view()
        
        # Update the loss plot
        loss_line.set_ydata(losses)
        loss_line.set_xdata(range(len(losses)))
        ax2.relim()
        ax2.autoscale_view()
        
        # Draw the updated plots
        fig.canvas.draw()
        fig.canvas.flush_events()

print("Training complete.")

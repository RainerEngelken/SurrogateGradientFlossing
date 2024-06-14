import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class VanillaRNN(torch.nn.Module):
    def __init__(self, Nin, N, Nout):
        super(VanillaRNN, self).__init__()
        self.N = N
        self.W_in = torch.nn.Parameter(torch.randn(N, Nin) / np.sqrt(Nin))
        self.W_h = torch.nn.Parameter(torch.randn(N, N) / np.sqrt(N))
        self.b_h = torch.nn.Parameter(torch.zeros(N))
        self.W_out = torch.nn.Parameter(torch.randn(Nout, N) / np.sqrt(N))
        self.b_out = torch.nn.Parameter(torch.zeros(Nout))

    def forward(self, x, h_prev):
        h = torch.tanh(self.W_in @ x + self.W_h @ h_prev + self.b_h)
        y = self.W_out @ h + self.b_out
        return h, y

def calculate_jacobian_analytical(vanilla_rnn, h):
    tanh_prime = 1 / torch.cosh(h)**2
    jacobian = vanilla_rnn.W_h @ torch.diag(tanh_prime)
    return jacobian

def calculate_lyapunov_spectrum(vanilla_rnn, x_data, nle, seedIC=1, seedONS=1):
    n = vanilla_rnn.N
    steps = len(x_data)
    ONSstep = 1

    torch.manual_seed(seedIC)
    h = torch.zeros(n, dtype=torch.float32, requires_grad=True)

    torch.manual_seed(seedONS)
    Q, R = torch.linalg.qr(torch.randn(n, nle))
    ls = torch.zeros(nle, dtype=torch.float32)

    for step in range(steps):
        x = x_data[step]
        D = calculate_jacobian_analytical(vanilla_rnn, h)
        h, _ = vanilla_rnn(x, h)
        Q = D @ Q

        if step % ONSstep == 0 and nle > 0:
            Q, R = torch.linalg.qr(Q)
            ls += torch.log(torch.abs(torch.diag(R))) / ONSstep

    return ls

# Model parameters
Nin = 1
nb_steps = 1024
nb_hidden = 64
Nout = 1
nle = 16  # number of Lyapunov exponents to floss
Ef = 41  # number of flossing epochs

# Initialize the RNN
vanilla_rnn = VanillaRNN(Nin, nb_hidden, Nout)
optimizer = optim.Adam(vanilla_rnn.parameters())

# Generate input data
pIn = 0.5  # input probability
x_data = [torch.tensor(np.random.rand(Nin) < pIn, dtype=torch.float32) for _ in range(nb_steps)]

# Optimization setup
losses = []
lyapunov_spectra = []

# Set up the initial plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

lyapunov_line, = ax1.plot([], [], "k", label="Lyapunov spectrum after flossing")
ax1.set_xlabel(r"Index $i$")
ax1.set_ylabel(r"Lyapunov Exponent $\lambda_i$ (1/step)")
ax1.legend()

loss_line, = ax2.semilogy([], [], "r", label="Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Loss over Epochs")
ax2.legend()

plt.ion()  # Turn on interactive mode
plt.show()

lyapunov_spectrum_initial = calculate_lyapunov_spectrum(vanilla_rnn, x_data[len(x_data)//2:], nle).detach().cpu().numpy()
ax1.plot(lyapunov_spectrum_initial, "r", label="Lyapunov spectrum before flossing")
ax1.legend()

# Training loop
for epoch in range(Ef):
    torch.manual_seed(epoch)
    x_data = [torch.tensor(np.random.rand(Nin) < pIn, dtype=torch.float32) for _ in range(nb_steps)]

    optimizer.zero_grad()

    lyapunov_spectrum = calculate_lyapunov_spectrum(vanilla_rnn, x_data[len(x_data) // 2:], nle)

    # Calculate the loss (mean square of the first nle Lyapunov exponents)
    loss = torch.mean(lyapunov_spectrum**2)
    print(f"Epoch {epoch}: Loss = {loss.item()}")

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
        ax1.legend()

        # Update the loss plot
        loss_line.set_ydata(losses)
        loss_line.set_xdata(range(len(losses)))
        ax2.relim()
        ax2.autoscale_view()

        # Draw the updated plots
        fig.canvas.draw()
        fig.canvas.flush_events()

print("Flossing complete.")


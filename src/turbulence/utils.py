import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Plot the training and validation losses
def plot_losses(training_loss, validation_loss):
    print("Plotting losses...")
    if len(training_loss) == 0 or len(validation_loss) == 0:
        raise ValueError('No training or validation losses to plot')
    else : 
        plt.figure(figsize=(10, 5))
        plt.title('Training and validation losses')
        plt.plot(training_loss, label='Training loss')
        plt.plot(validation_loss, label='Validation loss')
        plt.xlabel('nb_batch * Epoch')
        plt.ylabel('Log loss')

# Plot the reconstruction 
def plot_reconstruction(model, batch):
    reconstructed_images = []
    with torch.no_grad():
        for image in batch :
            reconstructed_images.append(model(image))

    N = len(reconstructed_images)
    fig, axes = plt.subplots(N, 2, figsize=(20, 15))
    fig.suptitle("Reconstruction - Testing Batch")

    for index in range(0, N):
        original = batch[index]
        reconstructed = reconstructed_images[index]
        axes[index, 0].imshow(original.squeeze(0))
        axes[index, 0].set_title("original")
        axes[index, 1].imshow(reconstructed.squeeze(0))
        axes[index, 1].set_title("reconstructed")

# Plot the spectrums
def plot_spectrums(model, batch):
    reconstructed_images = []
    with torch.no_grad():
        for image in batch :
            reconstructed_images.append(model(image))

    N = len(reconstructed_images)
    fig, axes = plt.subplots(N, 3, figsize=(20, 15))
    fig.suptitle(f"Spectrums - Testing Batch")

    for index in range(0, N):
        original_spectrum = torch.mean(torch.mean(torch.abs(torch.fft.rfft(batch[index],axis=1))**2,axis=0), axis=1).numpy()
        reconstructed_spectrum = torch.mean(torch.mean(torch.abs(torch.fft.rfft(reconstructed_images[index],axis=1))**2,axis=0), axis=1).numpy()
        axes[index, 0].loglog(original_spectrum)
        axes[index, 0].set_title("original")
        axes[index, 1].loglog(reconstructed_spectrum)
        axes[index, 1].loglog([np.argmin(original_spectrum), min(original_spectrum)], color = "white") # Rescaling the reconstructed graph
        axes[index, 1].set_title("reconstructed")
        axes[index, 2].loglog(original_spectrum, label='original')
        axes[index, 2].loglog(reconstructed_spectrum, label='reconstructed')
        axes[index, 2].legend()
        axes[index, 2].set_title("combined")

# Plot the amplitude of the latent space modes
def plot_latent_space_mode_amplitude(model, img):
    model(img) # Forward pass to compute the latent space
    latent = model.latent_space_complex.detach().squeeze(0)
    time = range(latent.shape[0])  # Temporel Axis
    latent_abs = torch.abs(latent)

    _, axes = plt.subplots(3, 1, figsize=(10, 12))
    for i in range(48):  # Tracer chaque mode
        axes[0].plot(time, latent.real[:, i], alpha=0.6)
        axes[1].plot(time, latent.imag[:, i], alpha=0.6)
        axes[2].plot(time, latent_abs[:, i], alpha=0.6)

    axes[0].set_title("Partie réelle des modes de l'espace latent")
    axes[1].set_title("Partie imaginaire des modes de l'espace latent")
    axes[2].set_title("Module des modes de l'espace latent")

    for ax in axes:
        ax.set_xlabel("Temps")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()

# Compute the flatness of the latent space modes modules
def compute_flatness(u_latent):
    """Calcule la flatness pour chaque mode dans l'espace latent."""
    mean_u2 = torch.mean(u_latent**2, dim=0)  # ⟨ u^2 ⟩ sur le temps
    mean_u4 = torch.mean(u_latent**4, dim=0)  # ⟨ u^4 ⟩ sur le temps
    flatness = mean_u4 / (mean_u2**2)  # F = ⟨ u^4 ⟩ / ⟨ u^2 ⟩^2
    return flatness.cpu().numpy()

# Plot the flatness of the latent space modes modules
def plot_flatness(model, img):
    model(img)  # Forward pass to compute the latent space
    latent = model.latent_space_complex.detach().squeeze(0)
    latent_abs = torch.abs(latent)
    flatness = compute_flatness(latent_abs)
    plt.figure(figsize=(8, 5))
    plt.plot(flatness, label="flatness_abs_latent (|u|)")
    plt.xlabel("Mode index")
    plt.ylabel("Flatness")
    plt.legend()
    plt.title("Flatness de l'espace latent")
    plt.grid(True)
    plt.show()

def finite_diff_time_derivative(u, dt):
    times,_ = u.shape
    du_dt = torch.zeros_like(u)

    # Forward difference for the first time step
    du_dt[0, :] = (u[ 1, :] - u[0, :]) / dt
    # Backward difference for the last time step
    du_dt[-1, :] = (u[-1, :] - u[-2, :]) / dt
    # Central differences for the interior time steps
    if times > 2:
        du_dt[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dt)

    return du_dt

N = 48
k_n = torch.tensor([2**(n) for n in range(N)], dtype=torch.float32)

def gledzer_physics_loss_complex(u, dt=2e-3, k_n=k_n, lambd = 2,epsilon=0.5):
    device = u.device
    batch,channel,seq_len, n_shells = u.shape
    u = u.view(batch*seq_len,n_shells*channel)
    seq_len, n_shells = u.shape
    # Coefficients for the nonlinear terms
    a, b, c = 1.0, -epsilon/lambd, (epsilon-1)/lambd

    # Ensure k_n is a tensor on the correct device
    if not torch.is_tensor(k_n):
        k_n = torch.tensor(k_n, dtype=torch.float32, device=device)
    else:
        k_n = k_n.to(device).float()  # shape: (n_shells,)

    # Compute the finite-difference time derivative of u
    du_dt, = torch.gradient(u,spacing=dt,dim=0,edge_order=2)  # shape: (t, n_shells)

    # Precompute conjugate of u for nonlinear terms
    u_conj = u.conj()

    U_plus_1 = torch.zeros(seq_len, n_shells, dtype=torch.complex64, device=device)
    U_plus_1[:,:-1] = u_conj[:,1:]

    U_plus_2 = torch.zeros(seq_len, n_shells, dtype=torch.complex64, device=device)
    U_plus_2[:,:-2] = u_conj[:,2:]

    U_minus_1 = torch.zeros(seq_len, n_shells, dtype=torch.complex64, device=device)
    U_minus_1[:,1:] = u_conj[:,:-1]

    U_minus_2 = torch.zeros(seq_len, n_shells, dtype=torch.complex64, device=device)
    U_minus_2[:,2:] = u_conj[:,:-2]


    #   term1[n] = a * u[:, n+1] * u[:, n+2]
    term1 =  a * U_plus_1 * U_plus_2

    #   term2[n] = b * u[:, n-1] * u[:, n+1]
    term2 = b * U_minus_1 * U_plus_1

    #   term3[n] = c * u[:, n-2] * u[:, n-1]
    term3 = c * U_minus_2 *  U_minus_1

    # Combine the nonlinear terms.
    # Multiply each shell by its corresponding wavenumber k_n (broadcast along time)

    #pdb.set_trace()
    nonlinear = 1j * (term1 + term2 + term3) * k_n[None, :]

    # --- Right-hand side of the GOY equation ---
    rhs = nonlinear 
    #pdb.set_trace()

    # --- Residual and Loss ---
    residual = du_dt - rhs
    loss_real = F.mse_loss(residual.real, torch.zeros_like(residual.real))
    loss_imag = F.mse_loss(residual.imag, torch.zeros_like(residual.imag))
    loss = loss_real + loss_imag

    return loss

def combined_loss(preds,target, model,a=1e-22, b=0.5):
    
    mse_loss = F.mse_loss(preds, target)
    physics_loss = gledzer_physics_loss_complex(model.latent_space_complex)

    # L3 = LPIPS loss
    lpips_loss = torch.tensor(0.0, device=preds.device)

    if hasattr(model, 'lpips_loss_fn') and model.lpips_loss_fn is not None:
        x_lpips = target.to(model.lpips_device)
        x_hat_lpips = preds.to(model.lpips_device)
        lpips_loss = model.lpips_loss_fn(x_lpips, x_hat_lpips).mean()
    
    print(f"MSE: {mse_loss.item():.3e}, Physics: {physics_loss.item():.3e}, LPIPS: {lpips_loss.item():.3e}")

    return a*physics_loss + b*mse_loss + (1-a-b)*lpips_loss

def load_model(model, path):
    if path.endswith('.ckpt'):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
    elif path.endswith('.pt'):
        model.load_state_dict(torch.load(path))
    elif path.endswith('.pth'):
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError('Invalid model path')
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 0. Snapshot folder 생성
# ==========================================================
os.makedirs("snapshots", exist_ok=True)

# ==========================================================
# 1. Parameters
# ==========================================================
np.random.seed(42)

N = 64
L = 1.0
eps = 0.02
sigma = 100.0
m = 0.2
dt = 5e-4
steps = 20000
viz_interval = 200   # 몇 step마다 저장할지

dx = L / N
volume_element = dx**3

# ==========================================================
# 2. Grid and Fourier modes
# ==========================================================
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K4 = K2**2

K2_inv = np.zeros_like(K2)
K2_inv[K2 > 0] = 1.0 / K2[K2 > 0]

# ==========================================================
# 3. Dealiasing mask (2/3 rule)
# ==========================================================
k_cut = (2.0/3.0) * np.max(np.abs(k))
dealias = (np.abs(KX) < k_cut) & \
          (np.abs(KY) < k_cut) & \
          (np.abs(KZ) < k_cut)

# ==========================================================
# 4. Initial condition
# ==========================================================
u = m + 0.05 * (np.random.rand(N, N, N) - 0.5)
u_hat = np.fft.fftn(u)

# mass constraint
u_hat[0,0,0] = m * (N**3)
u = np.real(np.fft.ifftn(u_hat))

# ==========================================================
# 5. Energy functional
# ==========================================================
def compute_energy(u, u_hat):

    grad_energy = 0.5 * eps**2 * volume_element * \
        np.sum(K2 * np.abs(u_hat)**2) / (N**6)

    nonlocal_energy = 0.5 * sigma * volume_element * \
        np.sum(np.abs(u_hat)**2 * K2_inv) / (N**6)

    bulk_energy = volume_element * \
        np.sum(0.25 * (1 - u**2)**2)

    return grad_energy + bulk_energy + nonlocal_energy

# ==========================================================
# 6. Time stepping
# ==========================================================
energies = []
diff_norms = []

for step in range(steps):

    u_old = u.copy()

    # nonlinear term
    nonlinear = u**3 - u
    nonlinear_hat = np.fft.fftn(nonlinear)

    # dealiasing
    nonlinear_hat *= dealias

    # semi-implicit update
    numerator = u_hat - dt * K2 * nonlinear_hat
    denominator = 1 + dt * (eps**2 * K4 + sigma)
    u_hat = numerator / denominator

    # mass conservation
    u_hat[0,0,0] = m * (N**3)

    # inverse FFT
    u = np.real(np.fft.ifftn(u_hat))

    # ======================================================
    # Snapshot 저장
    # ======================================================
    if step % viz_interval == 0:

        # diagnostics
        E = compute_energy(u, u_hat)
        energies.append(E)

        diff = np.sqrt(volume_element * np.sum((u - u_old)**2))
        diff_norms.append(diff)

        print(f"Step {step:5d} | Energy = {E:.8f} | "
              f"L2 diff = {diff:.6e}")

        # PNG 저장
        plt.figure(figsize=(5,5))
        plt.imshow(u[:, :, N//2],
                   cmap='RdBu_r',
                   vmin=-1,
                   vmax=1)
        plt.axis('off')
        plt.tight_layout(pad=0)

        filename = f"snapshots/frame_{step:05d}.png"
        plt.savefig(filename,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

# ==========================================================
# 7. Final diagnostics plot
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].plot(np.arange(len(energies))*viz_interval,
             energies)
axes[0].set_title("Energy Decay")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Energy")
axes[0].grid()

axes[1].plot(np.arange(len(diff_norms))*viz_interval,
             diff_norms)
axes[1].set_yscale("log")
axes[1].set_title("L2 Difference")
axes[1].set_xlabel("Step")
axes[1].grid()

plt.tight_layout()
plt.show()

# ==========================================================
# 8. Save diagnostics
# ==========================================================
np.savetxt("energy_log.txt",
           np.column_stack((np.arange(len(energies))*viz_interval,
                            energies,
                            diff_norms)),
           header="step  energy  L2_difference")

print("Simulation complete.")
print("Snapshots saved in ./snapshots/")
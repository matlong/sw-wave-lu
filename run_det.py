"""Run propagation of a monochromatic inertio-gravity wave driven by the deterministic linearized shallow water model.
Copyright 2023 Long Li.
"""
import numpy as np
import torch
from sw import LSW
torch.backends.cudnn.deterministic = True

# Set param and create object
outdir = './run/det' 
param = {
        'nx': 128, # zonal grid number
        'ny': 128, # meridional grid number
        'n_ens': 1, # ensemble members
        'Lx': 5120.0e3, # zonal length (m)
        'Ly': 5120.0e3, # meridional length (m)
        'H': 100., # ocean depth (m)
        'f0': 1.0e-4, # mean Coriolis parameter (s^-1)
        'g': 9.81, # gravity value (m/s^2)
        'cfl': 5., # CFL number
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
wave = LSW(param)

# Set initial condition
nk = 1 # number of wavenumbers
id0_ky, id0_kx = 0, 3 # smallest wavenumbers
wave.state_vec_hat[-1,:,id0_ky:id0_ky+nk,id0_kx:id0_kx+nk] = 1.
wave.init_wave()
t = 0.
dt = wave.dt

# Set run length and output frequency
n_steps = int(5*365*24*3600/dt)+1 # 5 years
freq_checknan = int(24*3600/dt) # 1 day
freq_plot = int(2*24*3600/dt) 
freq_log = int(24*3600/dt)
freq_save = int(24*3600/dt)
n_steps_save = 0

# Initialize output
if freq_save > 0:
    import os
    os.makedirs(outdir) if not os.path.isdir(outdir) else None
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f'uvh_{n_steps_save}.npz')
    state_vec = torch.fft.ifft2(wave.state_vec_hat, norm=wave.norm_fft).real
    np.savez(filename, t=t/(365*24*3600), \
             u=(state_vec[0]).cpu().numpy().astype('float32'), \
             v=(state_vec[1]).cpu().numpy().astype('float32'), \
             eta=(state_vec[-1]).cpu().numpy().astype('float32'))
    n_steps_save += 1

# Initialize plot
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion()
    x, y = np.meshgrid(np.linspace(1e-3*wave.dx/2, 1e-3*(wave.nx*wave.dx-wave.dx/2), wave.nx, dtype=np.float64), \
                       np.linspace(1e-3*wave.dy/2, 1e-3*(wave.ny*wave.dy-wave.dy/2), wave.ny, dtype=np.float64), \
                       indexing='xy')
    eta = torch.fft.ifft2(wave.state_vec_hat[-1,0], norm=wave.norm_fft).real.cpu().numpy()
    eta_max = abs(eta).max()
    levels = np.linspace(-eta_max, eta_max, 11)  
    fig, ax = plt.subplots(1, figsize=(3.5,3), layout=None, subplot_kw={"projection": "3d"})
    im = ax.plot_surface(x, y, eta, cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_zlim(-eta_max, eta_max)
    ax.set_title(f'Surface elevation (t={t/(365*24*60**2):.3f} yr)')
    ax.zaxis.set_ticklabels([])
    cb = fig.colorbar(im, ticks=np.linspace(-eta_max,eta_max,5), format='%2.1f', extend='both', shrink=0.5, aspect=10)
    cb.set_label('$\eta$ (m)')
    plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1):
    
    wave.step()
    t += dt

    if n % freq_checknan == 0 and torch.isnan(torch.fft.ifft2(wave.state_vec_hat[-1], norm=wave.norm_fft).real).any():
        raise ValueError(f'Stopping, NAN number in eta at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        plt.cla()
        eta = torch.fft.ifft2(wave.state_vec_hat[-1,0], norm=wave.norm_fft).real.cpu().numpy()
        ax.plot_surface(x, y, eta, cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
        ax.set_xlabel('$x$ (km)')
        ax.set_ylabel('$y$ (km)')
        ax.set_zlim(-eta_max, eta_max)
        ax.zaxis.set_ticklabels([])
        ax.set_title(f'Surface elevation (t={t/(365*24*60**2):.3f} yr)')
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        state_vec = torch.fft.ifft2(wave.state_vec_hat, norm=wave.norm_fft).real
        u = (state_vec[0]).cpu().numpy()
        v = (state_vec[1]).cpu().numpy()
        eta = (state_vec[-1]).cpu().numpy()
        energy_mean = wave.H/2 * (np.mean(u, axis=0)**2 + np.mean(v, axis=0)**2).mean() \
                    + wave.g/2 * (np.mean(eta, axis=0)**2).mean()
        mean_energy = wave.H/2 * (u**2 + v**2).mean() + wave.g/2 * (eta**2).mean()           
        log_str = f'n={n:06d}, t={t/(365*24*60**2):.3f} yr, ' \
                  f'(u_mean, u_max): ({u.mean():+.1E}, {np.abs(u).max():.2f}), ' \
                  f'(v_mean, v_max): ({v.mean():+.1E}, {np.abs(v).max():.2f}), ' \
                  f'(eta_mean, eta_max): ({eta.mean():+.1E}, {np.abs(eta).max():.2f}), ' \
                  f'(E_mean, mean_E): ({energy_mean:.2f}, {mean_energy:.2f}).'
        print(log_str)

    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        filename = os.path.join(outdir, f'uvh_{n_steps_save}.npz')
        state_vec = torch.fft.ifft2(wave.state_vec_hat, norm=wave.norm_fft).real
        np.savez(filename, t=t/(365*24*3600), \
                 u=(state_vec[0]).cpu().numpy().astype('float32'), \
                 v=(state_vec[1]).cpu().numpy().astype('float32'), \
                 eta=(state_vec[-1]).cpu().numpy().astype('float32'))
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved uvh to {filename}')

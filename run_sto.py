"""Run propagation of a monochromatic inertio-gravity wave driven by the stochastic shallow water model - so called `Location Uncertainty'. 
See the reference: [Mémin-et-al. 2023]
Copyright 2023 Long Li.
"""

import numpy as np
import torch
from sw import LU

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set param and create object
outdir = f'./run' 
param = {
        'nx': 128, # zonal grid number
        'ny': 128, # meridional grid number
        'n_ens': 100, # ensemble members
        'Lx': 5120.0e3, # zonal length (m)
        'Ly': 5120.0e3, # meridional length (m)
        'H': 100., # ocean depth (m)
        'f0': 1.0e-4, # mean Coriolis parameter (s^-1)
        'g': 9.81, # gravity value (m/s^2)
        'cfl': 5., # CFL number
        'device': device,
        'type_noise': 'homogeneous', # 'constant' or 'homogeneous'
        }
wave = LU(param)

# Set initial condition
nk0 = 1 # number of wavenumbers
id0_ky, id0_kx = 0, 3 # smallest wavenumbers
wave.state_vec_hat[-1,:,id0_ky:id0_ky+nk0,id0_kx:id0_kx+nk0] = 1./nk0 # uniform distribution of amplitudes
wave.init_wave()
t = 0.
dt = wave.dt

# Set noise correlation kernel
nkr = 1 # number of wavenumbers
idr_ky, idr_kx = 6, 4 # smallest wavenumbers
alpha = np.sqrt(dt/param['cfl']) * wave.g / wave.f0 / nkr
wave.sigma_vec_hat[0,0,idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr] = -1j * alpha * wave.ky[idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr]
wave.sigma_vec_hat[1,0,idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr] =  1j * alpha * wave.kx[idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr]

# Set run length and output frequency
n_steps = int(5*365*24*3600/dt)+1 # 5 years
freq_checknan = int(24*3600/dt) # 1 day
freq_plot = int(2*24*3600/dt) 
freq_log = int(24*3600/dt) 
freq_save = int(5*24*3600/dt) 
n_steps_save = 0
diag_energy = True

# Initialize output
if freq_save > 0:
    import os
    outdir = os.path.join(outdir, f'sto_{wave.type_noise}')
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
    eta = torch.fft.ifft2(wave.state_vec_hat[-1], norm=wave.norm_fft).real.cpu().numpy()
    eta_max = abs(eta).max()
    levels = np.linspace(-eta_max, eta_max, 11)  
    fig, ax = plt.subplots(1, 2, figsize=(6,3), layout=None, subplot_kw={"projection": "3d"})
    fig.suptitle(f't={t/(365*24*60**2):.3f} yrs')
    ax[0].plot_surface(x, y, eta[0], cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
    im = ax[1].plot_surface(x, y, np.mean(eta, axis=0), cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
    for i in range(2):
        ax[i].set_xlabel('$x$ (km)')
        ax[i].set_ylabel('$y$ (km)')
        ax[i].set_zlim(-eta_max, eta_max)
        ax[i].zaxis.set_ticklabels([])
        cb = fig.colorbar(im, ax=ax[i], ticks=np.linspace(-eta_max,eta_max,5), format='%2.1f', extend='both', shrink=0.4, aspect=10)
        cb.set_label('$\eta$ (m)')
    ax[0].set_title('Pathwise wave')
    ax[1].set_title('Mean wave')
    plt.pause(0.1)

if diag_energy:
    time, diag_mean_energy, diag_energy_mean, diag_energy_eddy = [], [], [], []

# Time-stepping
for n in range(1, n_steps+1):
    
    wave.step()
    t += dt

    if n % freq_checknan == 0 and torch.isnan(torch.fft.ifft2(wave.state_vec_hat[-1], norm=wave.norm_fft).real).any():
        raise ValueError(f'Stopping, NAN number in eta at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        #plt.cla()
        eta = torch.fft.ifft2(wave.state_vec_hat[-1], norm=wave.norm_fft).real.cpu().numpy()
        fig.suptitle(f't={t/(365*24*60**2):.3f} yrs')
        for i in range(2):
            ax[i].clear()
            ax[i].set_xlabel('$x$ (km)')
            ax[i].set_ylabel('$y$ (km)')
            ax[i].set_zlim(-eta_max, eta_max)
            ax[i].zaxis.set_ticklabels([])
        ax[0].plot_surface(x, y, eta[0], cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
        im = ax[1].plot_surface(x, y, np.mean(eta, axis=0), cmap='RdBu_r', vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
        ax[0].set_title('Pathwise wave')
        ax[1].set_title('Mean wave')
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        state_vec = torch.fft.ifft2(wave.state_vec_hat, norm=wave.norm_fft).real
        u = (state_vec[0]).cpu().numpy()
        v = (state_vec[1]).cpu().numpy()
        eta = (state_vec[-1]).cpu().numpy()
        um, vm, etam = np.mean(u, axis=0), np.mean(v, axis=0), np.mean(eta, axis=0)
        mean_energy = (wave.H * (u**2 + v**2).mean() + wave.g * (eta**2).mean())/2           
        energy_mean = (wave.H * (um**2 + vm**2).mean() + wave.g * (etam**2).mean())/2
        energy_eddy = (wave.H * ((u-um)**2 + (v-vm)**2).mean() + wave.g * ((eta-etam)**2).mean())/2
        log_str = f'n={n:06d}, t={t/(365*24*60**2):.3f} yr, ' \
                  f'(u_mean, u_max): ({u.mean():+.1E}, {np.abs(u).max():.2f}), ' \
                  f'(v_mean, v_max): ({v.mean():+.1E}, {np.abs(v).max():.2f}), ' \
                  f'(eta_mean, eta_max): ({eta.mean():+.1E}, {np.abs(eta).max():.2f}), ' \
                  f'(E_mean, mean_E): ({energy_mean:.2f}, {mean_energy:.2f}).'
        print(log_str)
        if diag_energy:
            time = np.append(time, t/(365*86400))
            diag_mean_energy = np.append(diag_mean_energy, mean_energy)
            diag_energy_mean = np.append(diag_energy_mean, energy_mean)
            diag_energy_eddy = np.append(diag_energy_eddy, energy_eddy)

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


if diag_energy:
    # Plot time series
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(figsize=(5,3))
    plt.plot(time, diag_mean_energy, label='Mean of energy')
    plt.plot(time, diag_energy_mean, label='Energy of mean')
    plt.plot(time, diag_energy_eddy, label='Energy of eddy')
    plt.axis([time[0], time[-1], 0., 1.25*diag_mean_energy[0]])
    plt.legend(loc='best')
    plt.grid(which='both', axis='both')
    plt.xlabel('Time (years)')
    plt.ylabel('Energy (m$^3$/s$^2$)')
    plt.title(f'Under {wave.type_noise} noise', fontsize=8)
    plt.savefig(f'energy_{wave.type_noise}.pdf', bbox_inches='tight', pad_inches=0)

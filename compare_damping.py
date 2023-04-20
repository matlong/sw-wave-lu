"""Compare the dissipation of ensemble mean wave for different noise's scales (wavenumbers).
Copyright 2023 Long Li.
"""

import numpy as np
import torch
from sw import LU

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set the commun param for different objects
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
id0_ky, id0_kx = 0, 3 # wavenumber of init. cond.

# Create a list of objects with diff. noise's scales (from large to small)
waves = []
n_obj = 5
kstep = 2 
for i in range(n_obj):
    waves.append(LU(param))
    waves[i].state_vec_hat[-1,:,id0_ky,id0_kx] = 1.
    waves[i].init_wave()
    idr_ky, idr_kx = id0_ky + i*kstep, id0_kx + i*kstep
    alpha = np.sqrt(waves[i].dt/param['cfl']) * waves[i].g / waves[i].f0
    waves[i].sigma_vec_hat[0,0,idr_ky,idr_kx] = -1j * alpha * waves[i].ky[idr_ky,idr_kx]
    waves[i].sigma_vec_hat[1,0,idr_ky,idr_kx] =  1j * alpha * waves[i].kx[idr_ky,idr_kx]

# Set run length and output frequency
dt = waves[0].dt
n_steps = int(5*365*24*3600/dt)+1 # 5 years
freq_checknan = int(24*3600/dt) # 1 day
freq_log = int(24*3600/dt) 

# Time-stepping
t = 0.
year, diag = [], [] 
for n in range(1, n_steps+1):
    
    for i in range(n_obj):
        waves[i].step()
    t += dt

    if n % freq_checknan == 0:
        for i in range(n_obj):    
            if torch.isnan(torch.fft.ifft2(waves[i].state_vec_hat[-1], norm=waves[i].norm_fft).real).any():
                raise ValueError(f'Stopping, NAN number in eta of wave {i} at iteration {n}.')

    if freq_log > 0 and n % freq_log == 0:
        year.append(t/(365*86400))
        log_str = f'n={n:06d}, t={t/(365*24*60**2):.3f} yr, Emean: ( '
        energy_mean = []
        for i in range(n_obj):    
            state_vec = torch.fft.ifft2(waves[i].state_vec_hat, norm=waves[i].norm_fft).real
            u = torch.mean(state_vec[0], dim=0).cpu().numpy()
            v = torch.mean(state_vec[1], dim=0).cpu().numpy()
            eta = torch.mean(state_vec[-1], dim=0).cpu().numpy()
            energy_mean.append((waves[i].H * (u**2 + v**2).mean() + waves[i].g * (eta**2).mean())/2)
            log_str += f'{energy_mean[i]:.3f} '
        diag.append(energy_mean)
        print(log_str+') m^3/s^2')


import matplotlib.pyplot as plt
plt.ion()
plt.figure(figsize=(5,3))
for i in range(n_obj):
    idr_ky, idr_kx = id0_ky + i*kstep, id0_kx + i*kstep
    energy = [diag[j][i] for j in range(len(diag))]
    plt.plot(year, energy, label=f'$k_\sigma$ = ({idr_kx}$\Delta k_x$, {idr_ky}$\Delta k_y$)')
plt.axis([year[0], year[-1], 0., 1.25*diag[0][0]])
plt.legend(loc='best')
plt.grid(which='both', axis='both')
plt.xlabel('Time (years)')
plt.ylabel('Energy (m$^3$/s$^2$)')
plt.title(f'Under {waves[0].type_noise} noise', fontsize=8)
plt.savefig(f'energy_diff_{waves[0].type_noise}.pdf', bbox_inches='tight', pad_inches=0)

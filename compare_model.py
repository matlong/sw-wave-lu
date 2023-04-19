import numpy as np
import torch
from sw import NSW,LSW
torch.backends.cudnn.deterministic = True

# Set param
outdir = './run/two_models' 
param = {
        'nx': 128, # zonal grid number
        'ny': 128, # meridional grid number
        'n_ens': 1, # ensemble members
        'Lx': 5120.0e3, # zonal length (m)
        'Ly': 5120.0e3, # meridional length (m)
        'H': 100., # ocean depth (m)
        'f0': 1.0e-4, # mean Coriolis parameter (s^-1)
        'g': 9.81, # gravity value (m/s^2)
        'cfl': 1, # CFL number
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }

# Create objets
lin_wave = LSW(param)
nonlin_wave = NSW(param)

# Set initial condition
nk = 8 # number of wavenumbers
id0_ky, id0_kx = 0, 3 # smallest wavenumbers
lin_wave.state_vec_hat[-1,:,id0_ky:id0_ky+nk,id0_kx:id0_kx+nk] = 1./nk
lin_wave.init_wave()
nonlin_wave.state_vec_hat[-1,:,id0_ky:id0_ky+nk,id0_kx:id0_kx+nk] = 1./nk
nonlin_wave.init_wave()
t = 0.
dt = lin_wave.dt

# Set run length and output frequency
n_steps = int(5*365*24*3600/dt)+1 # 5 years
freq_checknan = int(24*3600/dt) # 1 day
freq_plot = int(5*24*3600/dt) 
freq_log = int(24*3600/dt)
freq_save = 0 #int(5*24*3600/dt)
n_steps_save = 0

# Initialize output
if freq_save > 0:
    import os
    os.makedirs(outdir) if not os.path.isdir(outdir) else None
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f'lin_uvh_{n_steps_save}.npz')
    state_vec = torch.fft.ifft2(lin_wave.state_vec_hat, norm=lin_wave.norm_fft).real
    np.savez(filename, t=t/(365*24*3600), \
             u=(state_vec[0]).cpu().numpy().astype('float32'), \
             v=(state_vec[1]).cpu().numpy().astype('float32'), \
             eta=(state_vec[-1]).cpu().numpy().astype('float32'))
    filename = os.path.join(outdir, f'nonlin_uvh_{n_steps_save}.npz')
    state_vec = torch.fft.ifft2(nonlin_wave.state_vec_hat, norm=nonlin_wave.norm_fft).real
    np.savez(filename, t=t/(365*24*3600), \
             u=(state_vec[0]).cpu().numpy().astype('float32'), \
             v=(state_vec[1]).cpu().numpy().astype('float32'), \
             eta=(state_vec[-1]).cpu().numpy().astype('float32'))
    n_steps_save += 1

# Initialize plot
if freq_plot > 0:
    import matplotlib.pyplot as plt
    from cmocean import cm
    plt.ion()
    f,a = plt.subplots(1,3)
    a[0].set_title('Linear wave')
    a[1].set_title('Nonlinear wave')
    a[2].set_title('Difference')
    plt.tight_layout()
    plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1):
    
    lin_wave.step()
    nonlin_wave.step()
    t += dt

    if n % freq_checknan == 0: 
        if torch.isnan(torch.fft.ifft2(lin_wave.state_vec_hat[-1], norm=lin_wave.norm_fft).real).any():
            raise ValueError(f'Stopping, NAN number in eta of linear wave at iteration {n}.')
        if torch.isnan(torch.fft.ifft2(nonlin_wave.state_vec_hat[-1], norm=nonlin_wave.norm_fft).real).any():
            raise ValueError(f'Stopping, NAN number in eta of nonlinear wave at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        eta_lin = torch.fft.ifft2(lin_wave.state_vec_hat[-1,0], norm=lin_wave.norm_fft).real.cpu().numpy()
        eta_nonlin = torch.fft.ifft2(nonlin_wave.state_vec_hat[-1,0], norm=nonlin_wave.norm_fft).real.cpu().numpy()
        eta_max = abs(eta_lin).max()
        a[0].imshow(eta_lin, cmap=cm.deep, origin='lower', vmin=-eta_max, vmax=eta_max, animated=True)
        a[1].imshow(eta_nonlin, cmap=cm.deep, origin='lower', vmin=-eta_max, vmax=eta_max, animated=True)
        a[2].imshow(eta_lin - eta_nonlin, cmap=cm.deep, origin='lower', vmin=-eta_max, vmax=eta_max, animated=True)
        plt.suptitle(f't={t/(365*24*60**2):.2f} yr')
        plt.pause(0.5)
    
    if freq_log > 0 and n % freq_log == 0:
        state_vec = torch.fft.ifft2(nonlin_wave.state_vec_hat, norm=nonlin_wave.norm_fft).real
        u = (state_vec[0]).cpu().numpy()
        v = (state_vec[1]).cpu().numpy()
        eta = (state_vec[-1]).cpu().numpy()
        h = eta + nonlin_wave.H
        um, vm, hm = np.mean(u, axis=0), np.mean(v, axis=0), np.mean(h, axis=0)
        energy_mean = 0.5*(hm*(um**2 + vm**2) + nonlin_wave.g*hm**2).mean()       
        mean_energy = 0.5*(h*(u**2 + v**2) + nonlin_wave.g*h**2).mean()       
        log_str = f'n={n:06d}, t={t/(365*24*60**2):.3f} yr, ' \
                  f'(u_mean, u_max): ({u.mean():+.1E}, {np.abs(u).max():.2f}), ' \
                  f'(v_mean, v_max): ({v.mean():+.1E}, {np.abs(v).max():.2f}), ' \
                  f'(eta_mean, eta_max): ({eta.mean():+.1E}, {np.abs(eta).max():.2f}), ' \
                  f'(E_mean, mean_E): ({1e-4*energy_mean:.2f}, {1e-4*mean_energy:.2f}).'
        print(log_str)
    
    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        filename = os.path.join(outdir, f'lin_uvh_{n_steps_save}.npz')
        state_vec = torch.fft.ifft2(lin_wave.state_vec_hat, norm=lin_wave.norm_fft).real
        np.savez(filename, t=t/(365*24*3600), \
                 u=(state_vec[0]).cpu().numpy().astype('float32'), \
                 v=(state_vec[1]).cpu().numpy().astype('float32'), \
                 eta=(state_vec[-1]).cpu().numpy().astype('float32'))
        filename = os.path.join(outdir, f'nonlin_uvh_{n_steps_save}.npz')
        state_vec = torch.fft.ifft2(nonlin_wave.state_vec_hat, norm=nonlin_wave.norm_fft).real
        np.savez(filename, t=t/(365*24*3600), \
                 u=(state_vec[0]).cpu().numpy().astype('float32'), \
                 v=(state_vec[1]).cpu().numpy().astype('float32'), \
                 eta=(state_vec[-1]).cpu().numpy().astype('float32'))
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved uvh to {filename}')

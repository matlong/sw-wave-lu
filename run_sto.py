import numpy as np
import torch
from sw import LU

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set param and create object
outdir = f'./run/sto_c' 
param = {
        'nx': 128, # zonal grid number
        'ny': 128, # meridional grid number
        'n_ens': 20, # ensemble members
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
nk0 = 4 # number of wavenumbers
id0_ky, id0_kx = 0, 3 # smallest wavenumbers
wave.state_vec_hat[-1,:,id0_ky:id0_ky+nk0,id0_kx:id0_kx+nk0] = 1./nk0 # uniform distribution of amplitudes
wave.init_wave()
t = 0.
dt = wave.dt

# Set noise correlation kernel
nkr = 4
idr_ky, idr_kx = id0_ky+nk0+1, id0_kx+nk0+1
alpha = np.sqrt(dt/param['cfl']) * wave.g / wave.f0 / nkr
wave.sigma_vec_hat[0,0,idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr] = -1j * alpha * wave.ky[idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr]
wave.sigma_vec_hat[1,0,idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr] =  1j * alpha * wave.kx[idr_ky:idr_ky+nkr,idr_kx:idr_kx+nkr]

# Set run length and output frequency
n_steps = int(5*365*24*3600/dt)+1 # 5 years
freq_checknan = int(24*3600/dt) # 1 day
freq_plot = int(5*24*3600/dt) 
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
    from cmocean import cm
    plt.ion()
    x, y = np.meshgrid(np.linspace(1e-3*wave.dx/2, 1e-3*(wave.nx*wave.dx-wave.dx/2), wave.nx, dtype=np.float64), \
                       np.linspace(1e-3*wave.dy/2, 1e-3*(wave.ny*wave.dy-wave.dy/2), wave.ny, dtype=np.float64), \
                       indexing='xy')
    id_ens = np.random.randint(wave.n_ens)
    eta = torch.fft.ifft2(wave.state_vec_hat[-1,id_ens], norm=wave.norm_fft).real.cpu().numpy()
    eta_max = abs(eta).max()
    levels = np.linspace(-eta_max, eta_max, 11)  
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, eta, cmap=cm.deep, vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
    cset = ax.contourf(x, y, eta, zdir='z', offset=-eta_max, cmap=cm.gray, levels=levels, vmin=-eta_max, vmax=eta_max)
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_zlabel('$\eta$ (m)')
    ax.set_zlim(-eta_max, eta_max)
    ax.set_title(f'Surface elevation (t={t/(365*24*60**2):.3f} yr)')
    plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1):
    
    wave.step()
    t += dt

    if n % freq_checknan == 0 and torch.isnan(torch.fft.ifft2(wave.state_vec_hat[-1], norm=wave.norm_fft).real).any():
        raise ValueError(f'Stopping, NAN number in eta at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        plt.cla()
        eta = torch.fft.ifft2(wave.state_vec_hat[-1,id_ens], norm=wave.norm_fft).real.cpu().numpy()
        surf = ax.plot_surface(x, y, eta, cmap=cm.deep, vmin=-eta_max, vmax=eta_max, linewidth=0, antialiased=False)
        cset = ax.contourf(x, y, eta, zdir='z', offset=-eta_max, levels=levels, cmap=cm.gray)
        ax.set_xlabel('$x$ (km)')
        ax.set_ylabel('$y$ (km)')
        ax.set_zlabel('$\eta$ (m)')
        ax.set_zlim(-eta_max, eta_max)
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

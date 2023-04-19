"""Pytorch, shallow water wave model, Long Li, 2023."""
import numpy as np
import torch

########################################################################################################################

class LSW:
    """Concise implementation of linearized shallow-water model."""

    def __init__(self, param):       
        # Input param
        self.nx = param['nx']
        self.ny = param['ny']
        self.n_ens = param['n_ens']
        self.H = param['H']
        self.g = param['g']
        self.f0 = param['f0']
        self.device = param['device']
        self.farr_kwargs = {'dtype': torch.float64, 'device': self.device}
        self.carr_kwargs = {'dtype': torch.complex128, 'device': self.device}
        self.base_shape = (self.n_ens, self.ny, self.nx)
        self.norm_fft = 'forward' # normalized by 1/n for fft

        # Grid (Fourier) and timestep    
        self.dx = param['Lx'] / param['nx']
        self.dy = param['Ly'] / param['ny']
        self.dt = param['cfl'] * min(self.dx, self.dy) / np.sqrt(self.g * self.H)
        kx = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.farr_kwargs)
        ky = torch.fft.fftfreq(self.ny, self.dy/(2*np.pi), **self.farr_kwargs)
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.ksq = self.kx**2 + self.ky**2
        self.iksq = 1./self.ksq
        self.iksq[0,0] = 0.
        maskx = torch.tensor(abs(self.kx) < (2/3)*abs(self.kx).max(), **self.farr_kwargs) # '2/3' rule
        masky = torch.tensor(abs(self.ky) < (2/3)*abs(self.ky).max(), **self.farr_kwargs)
        self.mask_dealias = maskx * masky
        self.mask_dealias[self.ny//2,:] = 0. # remove highest freq.
        self.mask_dealias[:,self.nx//2] = 0. 

        # Model and state vector (in Fourier)
        A = torch.zeros((self.ny,self.nx,3,3), **self.carr_kwargs) # linear operator
        A[...,0,1], A[...,1,0] = self.f0, -self.f0
        A[...,0,-1], A[...,1,-1] = -1j*self.g*self.kx, -1j*self.g*self.ky
        A[...,-1,0], A[...,-1,1] = -1j*self.H*self.kx, -1j*self.H*self.ky
        self.exp_integrator = torch.linalg.matrix_exp(A * self.dt)
        self.state_vec_hat = torch.zeros((3,)+self.base_shape, **self.carr_kwargs) # ['u_hat', 'v_hat', 'eta_hat']

    def init_wave(self):
        """Initialize Poincaré (inertio-gravity) wave"""
        # Dispersion 
        self.omega = torch.sqrt(self.g*self.H*self.ksq + self.f0**2) 
        
        # Polarization 
        self.state_vec_hat[0] = self.iksq * self.state_vec_hat[-1] * (self.omega*self.kx + 1j*self.f0*self.ky) / self.H
        self.state_vec_hat[1] = self.iksq * self.state_vec_hat[-1] * (self.omega*self.ky - 1j*self.f0*self.kx) / self.H

    def step(self):
        self.state_vec_hat = torch.einsum('yxij,jnyx->inyx', self.exp_integrator, self.state_vec_hat) 

########################################################################################################################

class LU(LSW):
    """Stochastic formulation: Linearized shallow-water model ('parent') under location uncertainty ('child')."""

    def __init__(self, param):
        super().__init__(param)
        self.type_noise = param['type_noise']       
        self.sqrt_dt = np.sqrt(self.dt)
        self.sigma_vec_hat = torch.zeros((2,1,)+self.base_shape[1:], **self.carr_kwargs)
        self.sigma_vec_dBt = torch.zeros((2,)+self.base_shape, **self.farr_kwargs) # ['sigma_x dBt', 'sigma_y dBt']
        
    def build_noise(self):
        dBt_hat = self.sqrt_dt * torch.randn(self.base_shape, **self.carr_kwargs) 
        if self.type_noise == 'constant':
            self.sigma_vec_dBt = torch.sum(self.sigma_vec_hat * dBt_hat, dim=(-2,-1), keepdim=True).real
        if self.type_noise == 'homogeneous':
            self.sigma_vec_dBt = torch.fft.ifft2(self.sigma_vec_hat * dBt_hat, norm=self.norm_fft).real

    def step(self):
        self.build_noise()
        # Milstein approximation of Ito integral
        state_vec = torch.fft.ifft2(self.state_vec_hat, norm=self.norm_fft).real
        adv_hat = -1j * self.kx * self.mask_dealias * torch.fft.fft2(state_vec * self.sigma_vec_dBt[0], norm=self.norm_fft) \
                  -1j * self.ky * self.mask_dealias * torch.fft.fft2(state_vec * self.sigma_vec_dBt[1], norm=self.norm_fft)
        adv = torch.fft.ifft2(adv_hat, norm=self.norm_fft).real
        dif_hat = -1j * self.kx * self.mask_dealias * torch.fft.fft2(adv * self.sigma_vec_dBt[0], norm=self.norm_fft) \
                  -1j * self.ky * self.mask_dealias * torch.fft.fft2(adv * self.sigma_vec_dBt[1], norm=self.norm_fft)
        self.state_vec_hat += adv_hat + dif_hat/2 # No Levy area process under homogeneous noise  
        # Apply exponential integrator
        super().step() 

########################################################################################################################

class NSW(LSW):
    """Fully nonlinear shallow-water model."""

    def __init__(self, param):
        super().__init__(param)
        # Redefine exponential linear operator for splitting
        A = torch.zeros((self.ny,self.nx,3,3), **self.carr_kwargs) 
        A[...,0,1], A[...,1,0] = self.f0, -self.f0
        A[...,0,-1], A[...,1,-1] = -1j*self.g*self.kx, -1j*self.g*self.ky
        A[...,-1,0], A[...,-1,1] = -1j*self.H*self.kx, -1j*self.H*self.ky
        self.exp_integrator = torch.linalg.matrix_exp(A * self.dt/2)

    def nonlinear_rhs(self, state_vec_hat):
        """Compute nonlinear RHS terms of mass and momentum equations"""
        dt_state_vec_hat = torch.zeros_like(state_vec_hat)
        state_vec = torch.fft.ifft2(state_vec_hat, norm=self.norm_fft).real
        # Divergence of mass fluxes
        dt_state_vec_hat[-1] = -1j * self.mask_dealias * ( \
                self.kx * torch.fft.fft2(state_vec[-1] * state_vec[0], norm=self.norm_fft) \
              + self.ky * torch.fft.fft2(state_vec[-1] * state_vec[1], norm=self.norm_fft) )
        # Vorticity flux of momentum
        curl = torch.fft.ifft2(1j*(self.kx*state_vec_hat[1] - self.ky*state_vec_hat[0]), norm=self.norm_fft).real
        dt_state_vec_hat[0] =  self.mask_dealias * torch.fft.fft2(curl*state_vec[1], norm=self.norm_fft)
        dt_state_vec_hat[1] = -self.mask_dealias * torch.fft.fft2(curl*state_vec[0], norm=self.norm_fft)
        # Gradient of kinetic energy
        ke_hat = torch.fft.fft2((state_vec[0]**2 + state_vec[1]**2)/2, norm=self.norm_fft)
        dt_state_vec_hat[0] -= 1j*self.kx * self.mask_dealias * ke_hat
        dt_state_vec_hat[1] -= 1j*self.ky * self.mask_dealias * ke_hat 
        return dt_state_vec_hat

    def step(self):
        """Strang splitting expontential integrator [Brachet-et-al. 2020]"""
        # Prediction by exponential integrator
        super().step() 
        # Add nonlinear terms by RK4 scheme  
        k1 = self.nonlinear_rhs(self.state_vec_hat)
        k2 = self.nonlinear_rhs(self.state_vec_hat + self.dt*k1/2)
        k3 = self.nonlinear_rhs(self.state_vec_hat + self.dt*k2/2)
        k4 = self.nonlinear_rhs(self.state_vec_hat + self.dt*k3)
        self.state_vec_hat += self.dt * (k1 + 2*k2 + 2*k3 + k4)/6
        # Correction by exponential integrator
        super().step() 

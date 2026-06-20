import tqdm
from abc import ABC, abstractmethod

import torch
import numpy as np

class BaseEDM(ABC):
    """Base class for EDM sampling. This implements EDM sampling for diffusion models trained
    with any noise schedule and parameterization. Sub-class this for specific noise schedules and parameterizations,
    and implement the following abstract methods:
        - noise_condition,
        - input_scaling,
        - output_scaling,
        - f_theta_estimate.    
    """
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def denoised_estimate(self, xt, sigma):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            xr: (N, C, H, W) tensor, sample without scaling by s(t) corresponding to p(x, sigma) in the EDM paper.
            sigma: noise level float or tensor of shape (N,)

        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((xt.shape[0],), sigma, dtype=xt.dtype, device=self.device)
        
        f_theta = self.f_theta_estimate(xt, sigma)
        
        c_out = self.output_scaling(sigma)
        c_out = self.match_dimensions(c_out, xt.shape)
        return xt + c_out*f_theta


    def reverse_diffusion(self, n_timesteps, xt=None, **kwargs):

        rho = kwargs.pop("rho", 7)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.001)
        shape = kwargs.pop("shape", None)
        assert shape is not None or xt is not None, "Either shape or xt must be provided for reverse diffusion."

        # num_steps = num_steps+1 # to include the first step 
        sigma_values = self.sigma_schedule(n_timesteps+1, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        if xt is None:
            xt = sigma_values[0]*torch.randn(shape, device=self.device)
        else:
            xt = xt.to(self.device)
        for step in tqdm.tqdm(range(n_timesteps), disable=True):
            
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            x_denoised = self.denoised_estimate(xt, sigma_current)
            dx = (xt - x_denoised)/ sigma_current

            # Euler step...
            xt = xt + dx * (t_next - t_current)

        xt = xt.contiguous()
        return xt

    def sigma_inv(self, sigma):
        """Returns the inverse of the function sigma(t). 
        This is NOT the noise condition sigma^{-1}(sigma) i.e. condition input to the model,
        but rather this is the inverse of the specific noise schedule sigma(t) in EDM's sampling ODE.
        Remember there can be different sigma schedules for VE-ODE. EDM used sigma(t)=t, 
        so mostly this will just return sigma itself, but in case we want to use a different noise schedule,
        this function will return the inverse of sigma(t).
        """
        return sigma

    
    @staticmethod
    def sigma_schedule(num_steps, rho=7, sigma_min=0.0064, sigma_max=80):
        """Sequence of noise levels for sampling, as given in EDM paper, Equation 5.
        """
        if num_steps == 1:
            return np.array([sigma_max])
        step_indices = np.arange(num_steps)
        return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    @staticmethod
    def match_dimensions(tensor, shape):
        """Return view of the input tensor to allow broadcasting with the shape."""
        tensor_shape = tensor.shape
        while len(tensor_shape) < len(shape):
            tensor_shape = tensor_shape + (1,) 
        return tensor.view(tensor_shape)


    @abstractmethod
    def noise_condition(self, sigma):
        """Implements c_noise(sigma) in the EDM paper, Table 1. 
        Depending on the model parameterization, this could be the time step t or the noise level sigma itself.
        """
        pass

    @abstractmethod
    def input_scaling(self, sigma):
        """Implements c_in(sigma) in the EDM paper, Table 1. 
        Depends on the specific Markov process e.g. VP-SDE, VE-SDE or DDPM.
        """
        pass

    @abstractmethod
    def output_scaling(self, sigma):
        """Implements c_out(sigma) in the EDM paper, Table 1. 
        Depends on what the DNN learns, e.g. score, epsilon etc.
        """
        pass

    @abstractmethod
    def f_theta_estimate(self, xt, sigma):
        """Depending on specific parameterization, this could be
        epsilon estimate or score estimate. For VP-SDE and DDPM, this will mostly be
        the epsilon estimate, and for VE-SDE this will mostly be the score estimate, 
        but this is flexible depending on how the model is trained.
        
        Args:
            x: (N, C, H, W) tensor
            sigma: (N,) tensor of noise level
        """
        c_noise = self.noise_condition(sigma)
        c_in = self.input_scaling(sigma)
        c_in = self.match_dimensions(c_in, xt.shape)
        
        eps =  self.estimator(c_in*xt, c_noise)
        return eps


class VPScore(BaseEDM):
    """EDM sampling for VP-SDE noise schedule and with score parameterization. 
    """
    def __init__(self, estimator, beta_min=0.05, beta_max=20, M=1, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_d = beta_max - beta_min
        self.M = M
        self.estimator = estimator.to(self.device)

    def get_init_shape(self, batch_size):
        """Model specific method to return the shape of the initial noise sample for sampling."""
        return (batch_size, 80, 256)

    def noise_condition(self, sigma):
        t =  ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d
        if self.M == 1000:
            t = t * (self.M - 1)
        return t

    def input_scaling(self, sigma):
        return 1 / torch.sqrt(sigma**2 + 1)

    def output_scaling(self, sigma):
        return -sigma

    def f_theta_estimate(self, xt, sigma):
        """
        Args:
            x: (N, C, H, W) tensor
            sigma: (N,) tensor of noise level
        """
        c_noise = self.noise_condition(sigma)
        c_in = self.input_scaling(sigma)
        c_in = self.match_dimensions(c_in, xt.shape)
        
        # model parameterization is score, so we need to convert to epsilon estimate
        # score to epsilon conversion based on the training schedule (VP-SDE)
        score =  self.estimator(c_in*xt, c_noise)
        eps_factor = self.match_dimensions(sigma/torch.sqrt(1+sigma**2), xt.shape)
        eps =  -score*eps_factor
        return eps


class VPEpsilon(BaseEDM):
    """EDM sampling for VP-SDE noise schedule and with epsilon parameterization. 
    """
    def __init__(self, estimator, beta_min=0.1, beta_max=20, M=1000, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_d = beta_max - beta_min
        self.M = M
        self.estimator = estimator.to(self.device)

    def get_init_shape(self, batch_size):
        """Model specific method to return the shape of the initial noise sample for sampling."""
        return (batch_size, 3, 256, 256)

    def noise_condition(self, sigma):
        t =  ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d
        if self.M == 1000:
            t = t * (self.M - 1)
        return t

    def input_scaling(self, sigma):
        return 1 / torch.sqrt(sigma**2 + 1)

    def output_scaling(self, sigma):
        return -sigma
    
    def estimate_epsilon(self, x, t):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        # return self.estimator.model.model(x.to(torch.float32), t.to(torch.float32))[:,:3]
        return self.estimator(x.to(torch.float32), t.to(torch.float32))[:,:3]

    def f_theta_estimate(self, xt, sigma):
        """
        Args:
            x: (N, C, H, W) tensor
            sigma: (N,) tensor of noise level
        """
        c_noise = self.noise_condition(sigma)
        c_in = self.input_scaling(sigma)
        c_in = self.match_dimensions(c_in, xt.shape)
        
        # model parameterizes epsilon,
        eps =  self.estimate_epsilon(c_in*xt, c_noise)
        return eps
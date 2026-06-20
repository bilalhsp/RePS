# from ..utils import set_up_logging
# set_up_logging()


import tqdm
import torch
import torchvision
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torchdiffeq import odeint

import logging
logger = logging.getLogger(__name__)

# from .trajectory import Trajectory
# import predictor_corrector.utils as utils

import matplotlib.pyplot as plt

def visualize_output(output, ax=None):
    """ Visualizes the output of the model."""
    with torch.no_grad():
        attr_input = 0.5*(output+1)
        attr_input = attr_input.contiguous()
        attr_input = attr_input.clamp(0,1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
    ax.imshow(attr_input[0].cpu().numpy().transpose([1,2,0]))


SAMPLER_REGISTRY = {}

def register_sampler(name):
    """Register posterior sampler with the given name"""
    def decorater(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError("Sampler already exists!")
        SAMPLER_REGISTRY[name] = cls
        return cls
    return decorater

def get_sampler(name, *args, **kwargs):
    """Returns posterior sampler based on its name"""
    if name in SAMPLER_REGISTRY:
        return SAMPLER_REGISTRY[name](*args, **kwargs)
    raise NotImplemented("Sampler not found!")


class PosteriorSampler(ABC):
    def __init__(self, diff_model, operator, latent=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.diff_model = diff_model
        self.operator = operator  # measurement operator
        # self.noiser = noiser  # noise model, e.g. Gaussian noise

        # freeze the diffusion model parameters
        for param in self.diff_model.estimator.parameters():
            param.requires_grad = False
        self.diff_model.estimator.eval()

        self.latent=latent


    @torch.no_grad()
    def get_measurement_signal(self, x):
        """Get the measurement signal y = Ax + noise"""
        y = self.operator.measure(x)
        return y
    
    @abstractmethod
    def generate_sample(self, measurement, start_x=None, **kwargs):
        """Generate a sample from the posterior distribution p(x|y)
        using the diffusion model. 
        
        Args:
            measurement (torch.Tensor): The measurement signal y.
            start_x (torch.Tensor, optional): Initial value for sample x.
                If None, it will be set using randn.
            kwargs: Additional arguments for the sampling process.
        
        """
        NotImplementedError


    def compute_cond_loss(self, x_0_hat, measurement, **kwargs):

        difference = self.operator(x_0_hat) - measurement  # to update any internal states of the operator
        norm = torch.linalg.norm(difference)
        return difference, norm
    

@register_sampler("reps")
class RePS(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'reps'
        self.dtype = torch.float32

    def generate_sample(self, measurement, **kwargs):
        """Generate a sample from the posterior distribution p(x|y)
        using the diffusion model.
        """
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.1)
        ode_sigma_min=kwargs.pop("ode_sigma_min", 0.01)

        n_restarts = kwargs.pop('n_restarts', 100)
        n_ode_steps = kwargs.pop("n_ode_steps", 10)
        
        rho = kwargs.pop("rho", 15)
        ode_rho = kwargs.pop("ode_rho", 7)
        sigma_restart = kwargs.pop("sigma_restart", sigma_max)
        show_progress = kwargs.pop("show_progress", True)
        

        restart_schedule = self.diff_model.sigma_schedule(n_restarts, rho=rho, sigma_min=sigma_min, sigma_max=sigma_restart)
        if sigma_restart != sigma_max:
            restart_schedule = np.concatenate(([sigma_max], restart_schedule))

        start_x = kwargs.pop('start_x', None)
        B = measurement.shape[0]
        # shape = self.diff_model.estimator.get_in_shape()
        shape = kwargs.pop("shape", (3, 256, 256))
        shape = (B, *shape)

        if start_x is None:
            xt = sigma_max*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            print(f"starting from initlization...")
            xt = start_x.to(self.device, dtype=self.dtype)

        for step in tqdm.tqdm(range(n_restarts), desc="Restarting...", disable=not show_progress):

            sigma_current = restart_schedule[step]
            xt = self.solve_ode(
                n_ode_steps, xt, sigma_max=sigma_current, sigma_min=ode_sigma_min, ode_rho=ode_rho,
                measurement=measurement, **kwargs
                )

            if step != n_restarts-1:
                sigma_next = restart_schedule[step+1]
                xt = xt + torch.randn_like(xt)*sigma_next
                # xt = xt + torch.randn_like(xt)*(sigma_next**2 - ode_sigma_min**2)**0.5

        xt = xt.contiguous().to(torch.float32)
        if self.latent:
            xt = self.diff_model.estimator.decode(xt)
        outputs = (xt, )
        return outputs
    
    def solve_ode(self, num_steps, start_x=None, **kwargs):
        ode_rho = kwargs.pop("ode_rho", 7)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.001)

        sigma_values = self.diff_model.sigma_schedule(num_steps, rho=ode_rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))
        
        x = start_x.to(self.device)
        for step in tqdm.tqdm(range(num_steps), disable=True):
            
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            x = self.ode_step(x, sigma_current, sigma_next, step_ratio=step/num_steps, **kwargs)
        x = x.contiguous().to(torch.float32)
        return x

    def ode_step(self, xt, sigma_current, sigma_next, measurement=None, **kwargs):

        intial_lr = kwargs.get("lr", 1.e-4)
        lr_min_ratio = kwargs.get("lr_min_ratio", None)      # minimum learning rate
        tol = kwargs.get("tol", 1e-10)
        num_iters = kwargs.get("num_iters", 50)
        lam = kwargs.get("lam", self.operator.sigma**2 / sigma_current**2)

        step_ratio = kwargs.get("step_ratio", None)
        if step_ratio is None or lr_min_ratio is None:
            lr = intial_lr
        else:
            lr = self.get_lr(intial_lr, lr_min_ratio, step_ratio)

        # DDIM ODE solver step...
        x0_hat = self.diff_model.denoised_estimate(xt, sigma_current).to(torch.float32)
        if measurement is not None:
            if getattr(self.operator, 'has_svd', False):
                x0_hat = self.closed_form_solution(measurement, x0_hat, sigma_current)
            else:
                x0_hat = self.optimize_image_adam(
                    x0_hat, measurement, x0_hat, lam=lam, lr=lr, num_iters=num_iters, 
                    tol=tol, verbose=False
                    )
        xt = x0_hat + sigma_next*(xt - x0_hat)/ sigma_current

        return xt
    
    def get_lr(self, lr, lr_min_ratio, ratio, rho=1):
        """Calculates the learning rate based on the ratio.
        Args:
            lr (float): Initial learning rate.
            lr_min_ratio (float): Minimum learning rate ratio.
            ratio (float): Ratio to adjust the learning rate. 
                For polynomial decay, ratio is increasing i.e. idx/num_steps.,
                for rate in terms of variances, ratio is the ratio of sigma_t/sigma_max.
        """
        multiplier = ((1-ratio)*1**(1/rho) + ratio*lr_min_ratio*1**(1/rho))**rho
        return lr * multiplier

    def regularized_least_squares_loss(self, x, y, x0, lam= 0.1) -> torch.Tensor:

        if self.latent:
            data_loss = torch.mean((self.operator(self.diff_model.estimator.decode(x)) - y) ** 2)
        else:
            data_loss = torch.mean((self.operator(x) - y) ** 2)
        reg_loss = lam * torch.mean((x - x0) ** 2)
        return data_loss + reg_loss
    
    def closed_form_solution(self, measurement, x0_hat, sigma_t):

        sr = self.operator.H

        alpha_t = 1
        sigma_y = self.operator.sigma
        A_T_y = sr.Ht(measurement).view(x0_hat.shape)
        b = sigma_y**2 * x0_hat + (sigma_t**2 / alpha_t**2) * A_T_y

        s = sr.singulars()  # flattened singular values
        scale = 1.0 / (alpha_t**2 * sigma_y**2 + sigma_t**2 * s**2)

        b_v = sr.Vt(b)
        x = sr.V(sr.add_zeros(scale * b_v[:, : scale.shape[0]]))
        return x.detach().view(x0_hat.shape)

    @torch.enable_grad()
    def optimize_image_adam(
        self,
        x_init,
        y,
        x0_hat,
        lam=0.1,
        lr=0.01,
        num_iters=50,
        tol=None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Optimize an image tensor x to match target y using Adam with regularized least squares loss.

        Args:
            x_init: Initial image tensor of shape (B, C, H, W)
            y: Target image tensor of same shape as x_init
            x0_hat: Reference image tensor for regularization
            lam: Regularization weight
            lr: Learning rate
            num_iters: Number of optimization steps
            device: Device to run optimization
            verbose: Whether to print loss during optimization

        Returns:
            Optimized image tensor (same shape as x_init)
        """
        device = self.device
        x = x_init.clone().detach().to(device)
        y = y.to(device)
        x0_hat = x0_hat.to(device)

        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.regularized_least_squares_loss(x, y, x0_hat, lam=lam)
            loss.backward()
            optimizer.step()

            if tol is not None and loss.item() < tol:
                if verbose:
                    print(f"Converged at iteration {i+1} with loss {loss.item():.6f}")
                break

            if verbose and (i % max(1, num_iters // 10) == 0):
                print(f"Iter {i+1}/{num_iters}, Loss: {loss.item():.6f}")

        return x.detach()
    

    
#########################################
###        Baseline methods
###        
######################################### 

@register_sampler("dps")
class DPS(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'dps'

        for param in self.diff_model.estimator.parameters():
            param.requires_grad = True

    def generate_sample(self, measurement, start_x=None, **kwargs):
        """Generate a sample from the posterior distribution p(x|y)
        using the diffusion model. 
        
        Args:
            measurement (torch.Tensor): The measurement signal y.
            start_x (torch.Tensor, optional): Initial value for sample x.
                If None, it will be set using randn.
            kwargs: Additional arguments for the sampling process.
        """
        scale = kwargs.get('scale', 1.0)
        n_timesteps = kwargs.get('n_timesteps', 100)
        stoc = kwargs.get('stoc', True)

        if start_x is None:
            x = torch.randn(1, 3, 256, 256)
        else:
            x = start_x
        start_t = self.diff_model.T
        x = x.to(self.device)
        B = x.shape[0]
        h = start_t/n_timesteps     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_timesteps)]
        for i, time_step in enumerate(tqdm.tqdm(timesteps)):
            if i < (n_timesteps - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)

            if s is None:
                with torch.no_grad():
                    x = self.diff_model.denoising_step(x, t, s=None, stoc=False)
            else:
                x.requires_grad_(True)  # Now x0 is a leaf tensor
                x_0_hat = self.diff_model.denoising_step(x, t, s=None, stoc=stoc)
                difference, norm = self.compute_cond_loss(x_0_hat, measurement)
                grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                x.detach_()

                with torch.no_grad():
                    alpha_bar_s = self.diff_model.get_alpha_bar(s)
                    alpha_bar_s = self.diff_model.match_dimensions(alpha_bar_s, x.shape)
                    alpha_bar_t = self.diff_model.get_alpha_bar(t)
                    alpha_bar_t = self.diff_model.match_dimensions(alpha_bar_t, x.shape)
                    
                    x = torch.sqrt(alpha_bar_s)*x_0_hat + torch.sqrt(1-alpha_bar_s)*(x - torch.sqrt(alpha_bar_t)*x_0_hat)/(torch.sqrt(1-alpha_bar_t))
                    x = x - scale * grad  # update step
        x = x.contiguous()
        output = (x, )
        return output
    
    
@register_sampler("daps")
class DAPS(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'daps'
        self.dtype = torch.float32

    def generate_sample(self, measurement, **kwargs):
        n_anneal_steps = kwargs.pop('n_anneal_steps', 50)
        n_ode_steps=kwargs.pop("n_ode_steps", 2)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.1)
        ode_sigma_min=kwargs.pop("ode_sigma_min", 0.001)
        rho = kwargs.pop("rho", 7)

        # langevin parameters
        n_lang_steps=kwargs.pop("n_lang_steps", 100)
        lr = kwargs.pop("lr", 1.e-4)
        lr_min_ratio = kwargs.pop("lr_min_ratio", 1.e-2)      # minimum learning rate
        tau = kwargs.pop("tau", 0.01)

        start_x = kwargs.pop('start_x', None)

        B = measurement.shape[0]
        # shape = self.diff_model.estimator.get_in_shape()
        shape = kwargs.pop("shape", (3, 256, 256))
        shape = (B, *shape)
       
        sigma_values = self.diff_model.sigma_schedule(n_anneal_steps + 1, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values[:-1], np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        
        if start_x is None:
            xt = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            xt = start_x.to(self.device, dtype=self.dtype)

        for step in tqdm.tqdm(range(n_anneal_steps), desc="Generating...", disable=not show_progress):
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            # 1. ODE solve to get x0_hat
            x0_hat = self.solve_ode(
                num_steps=n_ode_steps, start_x=xt,
                sigma_max=sigma_current, sigma_min=ode_sigma_min, rho=rho,
                measurement=measurement, ratio=step/n_anneal_steps, #needed only for conditioned ODE
                **kwargs   
            )

            # 2. Langevin updates to get x0y
            eta_t = self.get_lr(lr, lr_min_ratio, step/n_anneal_steps)
            x0y, x0hat_loss, x0y_loss = self.langevin_updates(
                x0_hat, measurement, 
                n_lang_steps, eta_t, sigma_current, tau
                )
            
            # 3. Transition to next xt
            if sigma_next == 0:
                xt = x0y
            else:
                xt = x0y + torch.randn_like(x0y) * sigma_next
      
        xt = xt.contiguous().to(torch.float32)
        outputs = (xt, )
        return outputs
    
    def get_lr(self, lr, lr_min_ratio, ratio, rho=1):
        """Calculates the learning rate based on the ratio.
        Args:
            lr (float): Initial learning rate.
            lr_min_ratio (float): Minimum learning rate ratio.
            ratio (float): Ratio to adjust the learning rate. 
                For polynomial decay, ratio is increasing i.e. idx/num_steps.,
                for rate in terms of variances, ratio is the ratio of sigma_t/sigma_max.
        """
        # ratio is between 0 and 1
        multiplier = ((1-ratio)*1**(1/rho) + ratio*lr_min_ratio*1**(1/rho))**rho
        return lr * multiplier


    def langevin_updates(self, x0_hat, measurement, n_lang_steps, eta_t, sigma, tau=0.01):
        """Performs Langevin updates to condition the sample."""
        x = x0_hat.detach().clone()
        for idx in range(n_lang_steps):
            
            # Langevin score computation
            score, data_loss = self.get_langevin_score(x, x0_hat, measurement, sigma, tau)

            # Langevin update
            x = x + eta_t*score + np.sqrt(2*eta_t)*torch.randn_like(x)

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x), 0, 0 

            if idx ==0:
                x0hat_loss = data_loss
            elif idx == n_lang_steps-1:
                x0y_loss = data_loss
        return x, x0hat_loss, x0y_loss
    
    def get_langevin_score(self, x, x_0_hat, measurement, sigma, tau):
        """Compute the Langevin score for the given parameters."""

        x_tmp = x.clone().detach().requires_grad_(True)
        measurement_loss = self.operator.loss(x_tmp, measurement).sum()
        measurement_grad = torch.autograd.grad(measurement_loss, x_tmp)[0]
        x_tmp.detach_()

        data_term = -measurement_grad/tau**2
        xt_term = (x_0_hat - x)/sigma**2
        return data_term + xt_term, measurement_loss.item()
    
    def solve_ode(self, num_steps, start_x, **kwargs):

        rho = kwargs.pop("rho", 7)
        sigma_max=kwargs.pop("sigma_max", 100)
        sigma_min=kwargs.pop("sigma_min", 0.001)

        # num_steps = num_steps+1 # to include the first step (following DAPS)
        sigma_values = self.diff_model.sigma_schedule(num_steps+1, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)

        x = start_x.to(self.device)
        for step in tqdm.tqdm(range(num_steps), disable=True):
            
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            # get values of time from sigma values
            t_current = self.diff_model.sigma_inv(sigma_current)
            t_next = self.diff_model.sigma_inv(sigma_next)

            x_denoised = self.diff_model.denoised_estimate(x, sigma_current).to(torch.float32)
            dx = (x - x_denoised)/ sigma_current

            # Euler step...
            x = x + dx * (t_next - t_current)

        x = x.contiguous().to(torch.float32)
        return x



@register_sampler("reddiff")
class REDDIFF(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'reddiff'
        self.dtype = diff_model.dtype
        beta_start=1e-4
        beta_end=2e-2
        num_diffusion_timesteps=1000
        betas = np.linspace(
                    beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
        betas = torch.from_numpy(betas)
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        


    def alpha(self, t):
        # print(t.device, self.alphas.device)
        return self.alphas.index_select(0, t+1)

    @staticmethod
    def get_timesteps(cfg):
        skip = (cfg.exp.start_step - cfg.exp.end_step) // cfg.exp.num_steps
        ts = list(range(cfg.exp.end_step, cfg.exp.start_step, skip))
        return ts

    def diffusion_step(self, xt, t, **kwargs):
        alpha_t = self.alpha(t).view(-1, 1, 1, 1)
        et = self.diff_model.estimate_epsilon(xt, t)

        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        return et, x0_pred
    

    def read_cfg(self, cfg):
                # from reddiff implementation
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.cond_awd = cfg.algo.cond_awd
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.obs_weight = cfg.algo.obs_weight
        self.eta = cfg.algo.eta
        self.lr = cfg.algo.lr
        self.denoise_term_weight = cfg.algo.denoise_term_weight
        self.sigma_x0 = cfg.algo.sigma_x0

        logger.info(f'self.lr: {self.lr}')
        logger.info(f'self.sigma_x0: {self.sigma_x0}')

    # def sample(self, x, y, ts, **kwargs):
    def generate_sample(self, measurement, cfg):

        self.read_cfg(cfg)
        
        y_0 = measurement
        sigma_y = self.cfg.algo.sigma_y
        n = y_0.size(0)
        # H = self.H

        ts = self.get_timesteps(self.cfg)
    
        x = self.initialize(n)
        ss = [-1] + list(ts[:-1])
        # xt_s = [x.cpu()]
        # x0_s = []
        
        # mu_s = x.cpu()
        # x0_pred_s = x.cpu()
        # mu_fft_abs_s = torch.fft.fftshift(torch.abs(torch.fft.fft2(mu_s)))
        # mu_fft_ang_s = torch.fft.fftshift(torch.angle(torch.fft.fft2(mu_s)))

        #optimizer
        dtype = torch.FloatTensor
        mu = torch.autograd.Variable(x, requires_grad=True)   #, device=device).type(dtype)
        optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999
        #optimizer = torch.optim.SGD([mu], lr=1e6, momentum=0.9)  #momentum=0.9

        for ti, si in tqdm.tqdm(zip(reversed(ts), reversed(ss))):
            
            
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.alpha(s).view(-1, 1, 1, 1)
            
            sigma_x0 = self.sigma_x0  #0.0001
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)

            x0_pred = mu + sigma_x0*noise_x0
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
            
            #scale = 0.0
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            #xt = xt.clone().to('cuda').requires_grad_(True)
            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0
                        

            with torch.no_grad():
                et, x0_hat = self.diffusion_step(xt, t)
                # et, x0_hat = self.model(xt, y, t, scale=scale)   #et, x0_pred
                
                if not self.awd:
                    et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            et = et.detach()
            
            # e_obs = y_0 - H.H(x0_pred)
            e_obs = y_0 - self.operator(x0_pred)

            loss_obs = (e_obs**2).mean()/2
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
            
            snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()  #1d torch tensor
            
            if self.denoise_term_weight == "linear":
                snr_inv = snr_inv
            elif self.denoise_term_weight == "sqrt":
                snr_inv = torch.sqrt(snr_inv)
            elif self.denoise_term_weight == "square":
                snr_inv = torch.square(snr_inv)
            elif self.denoise_term_weight == "log":
                snr_inv = torch.log(snr_inv + 1.0)
            elif self.denoise_term_weight == "trunc_linear":
                snr_inv = torch.clip(snr_inv, max=1.0)
            elif self.denoise_term_weight == "power2over3":
                snr_inv = torch.pow(snr_inv, 2/3)
            elif self.denoise_term_weight == "const":
                snr_inv = torch.pow(snr_inv, 0.0)
            
            
            w_t = self.grad_term_weight*snr_inv   #0.25
            v_t = self.obs_weight

            loss = w_t*loss_noise + v_t*loss_obs
            
            #adam step
            optimizer.zero_grad()  #initialize
            loss.backward()
            optimizer.step()
            
            # # #save for visualization
            # if self.cfg.exp.save_evolution:
            #     if (ti/((self.cfg.exp.start_step - self.cfg.exp.end_step)//len(ts))) % (len(ts)//10) == 0:
            #         mu_s = torch.cat((mu_s, mu.detach().cpu()), dim=3)
            #         mu_fft_abs_s = torch.cat((mu_fft_abs_s, torch.fft.fftshift(torch.abs(torch.fft.fft2(mu.detach().cpu())))), dim=3)
            #         mu_fft_ang_s = torch.cat((mu_fft_ang_s, torch.fft.fftshift(torch.angle(torch.fft.fft2(mu.detach().cpu())))), dim=3)
            #         x0_pred_s = torch.cat((x0_pred_s, x0_pred.detach().cpu()), dim=3)
                
        # if self.cfg.exp.save_evolution:
        #     return x0_pred, mu, mu_s, x0_pred_s, mu_fft_abs_s, mu_fft_ang_s
        # else:
        # print(f"shape of x0_pred: {x0_pred.shape}")
        return x0_pred.detach(), mu.detach()

        
    def initialize(self, B, **kwargs):

        x = torch.randn(B, 3, 256, 256).to(self.device)
        return x




@register_sampler("resample")
class ReSample(PosteriorSampler):
    def __init__(self, diff_model, operator, latent=False):
        super().__init__(diff_model, operator, latent)
        self.name = 'resample'

        for param in self.diff_model.estimator.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate_sample(self, measurement, start_x=None, **kwargs):
        """Generate a sample from the posterior distribution p(x|y)
        using the diffusion model. 
        
        Args:
            measurement (torch.Tensor): The measurement signal y.
            start_x (torch.Tensor, optional): Initial value for sample x.
                If None, it will be set using randn.
            kwargs: Additional arguments for the sampling process.
        """
        scale = kwargs.get('scale', 0.5)
        n_timesteps = kwargs.get('n_timesteps', 500)
        gamma = kwargs.get('gamma', 40)
        max_iters_1 = kwargs.get('max_iters_1', 2000)
        max_iters_2 = kwargs.get('max_iters_2', 500)
        eta = kwargs.get('ddim_eta', 0.0)
        tol = kwargs.get('tol', 1e-4)

        
        if start_x is None:
            x = torch.randn(measurement.shape[0], 3, 256, 256)
        else:
            x = start_x
        start_t = self.diff_model.T
        x = x.to(self.device)
        B = x.shape[0]
        h = start_t/n_timesteps     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_timesteps)]
        for i, time_step in enumerate(tqdm.tqdm(timesteps)):
            if i < (n_timesteps - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)

            # DDIM sampler....
            alpha_bar_t = self.diff_model.get_alpha_bar(t)
            alpha_bar_t = self.diff_model.match_dimensions(alpha_bar_t, x.shape)

            
            # previous sample x_t_1
            if s is None:
                # predicted x_0
                eps = self.diff_model.estimator.model.model(x, t)[:,:3]
                x_0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
                x = x_0_hat
            else:
                alpha_bar_s = self.diff_model.get_alpha_bar(s)
                alpha_bar_s = self.diff_model.match_dimensions(alpha_bar_s, x.shape)
                
                with torch.enable_grad():
                    x = x.requires_grad_(True)
                    # DDIM sampler with sigma=0 (detertiministic)
                    eps = self.diff_model.estimator.model.model(x, t)[:,:3]
                    x_0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

                    # stochastic DDIM step
                    sigma_t = eta * torch.sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_s))
                    dir_xt = torch.sqrt(1 - alpha_bar_s - sigma_t**2) * eps
                    noise = sigma_t*torch.randn_like(x)
                    x_prime = torch.sqrt(alpha_bar_s) * x_0_hat + dir_xt + noise

                    # dummy step...tweedie's x0 is the same as x_0_hat
                    pseudo_x0 = x_0_hat
                    # pseudo_x0 = (x - (1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
                    # latent-DPS conditioning step
                    x = self.conditioning_step(x_prime, x, pseudo_x0, measurement, scale=alpha_bar_t*scale)
                    x.detach_()
            
            # conditioning
            inter_timesteps = 5     # interval for time travel..
            splits = 3
            index_split = n_timesteps // splits

            # resampling
            if i > index_split and i < n_timesteps -1:
                x_t = x.detach().clone()
                
                # performing every 10 steps
                if (i+1) % 10==0 and i+inter_timesteps < n_timesteps-1:
                    for k in range(i, i+inter_timesteps):
                        tk = torch.full((B,), timesteps[k], dtype=torch.float32, device=self.device)
                        sk = torch.full((B,), timesteps[k+1], dtype=torch.float32, device=self.device)

                        alpha_bar_tk = self.diff_model.get_alpha_bar(tk)
                        alpha_bar_tk = self.diff_model.match_dimensions(alpha_bar_tk, x.shape)
                        alpha_bar_sk = self.diff_model.get_alpha_bar(sk)
                        alpha_bar_sk = self.diff_model.match_dimensions(alpha_bar_sk, x.shape)

                        eps = self.diff_model.estimator.model.model(x, tk)[:,:3]
                        x_0_hat = (x - torch.sqrt(1 - alpha_bar_tk) * eps) / torch.sqrt(alpha_bar_tk)
                        # stochastic DDIM step
                        sigma_t = eta * torch.sqrt((1 - alpha_bar_sk) / (1 - alpha_bar_tk) * (1 - alpha_bar_tk / alpha_bar_sk))
                        dir_xt = torch.sqrt(1 - alpha_bar_sk - sigma_t**2) * eps
                        noise = sigma_t*torch.randn_like(x)
                        x = torch.sqrt(alpha_bar_sk) * x_0_hat + dir_xt + noise

                        # dummy step...tweedie's x0 is the same as x_0_hat
                        pseudo_x0 = x_0_hat
                        # pseudo_x0 = (x - (1 - alpha_bar_tk) * eps) / torch.sqrt(alpha_bar_tk)

                    # Hard conditioning step
                    pseudo_x0 = pseudo_x0.detach()

                    if i > 2*index_split:
                        max_iters = max_iters_2
                    else:
                        max_iters = max_iters_1

                    opt_var = self.pixel_optimization(measurement=measurement, 
                                                        x_prime=pseudo_x0,
                                                        eps=tol,
                                                        max_iters=max_iters
                                                        )
                    
                    sigma = gamma*(1 - alpha_bar_s) / (alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_s)  
                    x = self.stochastic_resample(pseudo_x0=opt_var, x_t=x_t, a_t=alpha_bar_s, sigma=sigma)
                    # x = x.requires_grad_() # Seems to need to require grad here

            
        # x = self.pixel_optimization(measurement=measurement, x_prime=x)
        x = x.detach().clone().contiguous()
        output = (x, )
        return output

    def conditioning_step(self, x_prime, x_prev, x_0_hat, measurement, scale=1.0):
        """ Conditioning step for ReSample sampler """
        # x.requires_grad_(True)  # Now x0 is a leaf tensor
        # x_0_hat = self.diff_model.denoising_step(x, t, s=None, stoc=stoc)
        difference, norm = self.compute_cond_loss(x_0_hat, measurement)
        grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_prime = x_prime - scale * grad  # update step
        return x_prime
    
    @torch.enable_grad()
    def pixel_optimization(self, measurement, x_prime, eps=1e-3, max_iters=2000):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-2) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, self.operator(opt_var)) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                break

        return opt_var

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))












    


    

#########################################
### Implementing guided diffusion solved by ODE
###         Name: guided-ode
######################################### 

@register_sampler("restart-ode")
class rODE(PosteriorSampler):
    def __init__(self, diff_model, operator):
        super().__init__(diff_model, operator)
        self.name = 'restart-ode'
        self.dtype = diff_model.dtype

    @torch.no_grad()
    def generate_sample(self, measurement, **kwargs):
        # standard DAPS..
        n_main_steps = kwargs.pop('n_main_steps', 50)
        sigma_max=kwargs.pop("sigma_max", 80)
        sigma_min=kwargs.pop("sigma_min", 0.002)
        rho = kwargs.pop("rho", 7)

        tau = kwargs.pop("tau", 0.01)

        restarts = kwargs.pop("restarts", None)
        start_x = kwargs.pop('start_x', None)
        show_progress = kwargs.pop("show_progress", True)
        B = measurement.shape[0]
        shape = kwargs.pop("shape", (B, 3, 256, 256))
        
    
        if start_x is None:
            xt = sigma_max*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            print(f"starting from initlization...")
            xt = start_x.to(self.device, dtype=self.dtype)

        sigma_values = self.diff_model.sigma_schedule(n_main_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
    
        restarts = self.match_t_mins(restarts, sigma_values)

        for step in tqdm.tqdm(range(n_main_steps), desc="Generating...", disable=not show_progress):
            sigma_current = sigma_values[step]
            sigma_next = sigma_values[step+1]

            xt = self.ode_step(xt, measurement, sigma_current, sigma_next, sigma_max, **kwargs)

            if restarts is not None:
                t_min_list = [restarts[idx][2] for idx in range(len(restarts))]
                if sigma_next in t_min_list:
                    self.restart_routine(xt, measurement, sigma_next, sigma_max, restarts, **kwargs)

        xt = xt.contiguous().to(torch.float32)
        outputs = (xt, )
        return outputs
    
    def restart_routine(self, xt, measurement, sigma, sigma_max, restarts, **kwargs):

        t_min_list = [restarts[idx][2] for idx in range(len(restarts))]
        idx = t_min_list.index(sigma)
        n_steps_restart, K, t_min, t_max = restarts[idx]
        for _ in range(K):
            eps = torch.randn_like(xt)*(t_max**2 - t_min**2)**0.5
            xt = xt + eps
            restart_sigmas = self.diff_model.sigma_schedule(n_steps_restart, rho=7, sigma_min=t_min, sigma_max=t_max)
            for i in range(n_steps_restart-1):
                sigma_current = restart_sigmas[i]
                sigma_next = restart_sigmas[i+1]
                xt = self.ode_step(xt, measurement, sigma_current, sigma_next, sigma_max, **kwargs)
        return xt

    
    def match_t_mins(self, restarts, sigma_values):
        if restarts is None:
            return restarts
        for idx, r_list in enumerate(restarts):
            _, _, t_min, _ = r_list
            min_idx = np.argmin(np.abs(sigma_values - t_min))
            restarts[idx][2] = float(sigma_values[min_idx])
        return restarts
    
    def ode_step(self, xt, measurement, sigma_current, sigma_next, sigma_max, **kwargs):
        # x0_hat = self.diff_model.denoised_estimate(xt, sigma_current).to(torch.float32)
        heun = kwargs.pop("heun", False)
        lr = kwargs.get("lr", 1.e-4)
        tol = kwargs.get("tol", 1.e-1)
        num_iters = kwargs.get("num_iters", 50)
        lam = kwargs.get("lam", self.operator.sigma**2 / sigma_current**2)
        # # EDM ODE solver step...
        # delta_t = sigma_next - sigma_current
        # score, likelihood_score = self.get_scores(xt, measurement, sigma_current, **kwargs)
        # # dx = (xt - x0_hat)/ sigma_current
        # dx = - sigma_current*score
        # # Euler step...
        # xt_prime = xt + dx * delta_t

        # DDIM ODE solver step...
        x0_hat = self.diff_model.denoised_estimate(xt, sigma_current).to(torch.float32)

        x0_hat = self.optimize_image_adam(
            x0_hat, measurement, x0_hat, lam=lam, lr=lr, num_iters=num_iters, tol=tol, verbose=False
            )
        xt = x0_hat + sigma_next*(xt - x0_hat)/ sigma_current


        # if sigma_next != 0 and heun:
        #     x0_hat_prime = self.diff_model.denoised_estimate(xt_prime, sigma_next).to(torch.float32)
        #     dx_prime = (xt_prime - x0_hat_prime)/ sigma_next
        #     xt = xt + 0.5*(dx + dx_prime)*delta_t
        # else:
        #     xt = xt_prime
        return xt
    
    def get_scores(self, xt, measurement, sigma, **kwargs):

        x_start = xt.clone().detach().requires_grad_(True)
        
        # solve ODE...
        x = x_start
        # likelihood_ode_steps = kwargs.get("likelihood_ode_steps", 1)
        # sigma_values = self.diff_model.sigma_schedule(likelihood_ode_steps, rho=7, sigma_min=0.01, sigma_max=sigma)
        # sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        # for step in range(likelihood_ode_steps):
        #     sigma_current = sigma_values[step]
        #     sigma_next = sigma_values[step+1]

        #     score = self.diff_model.score_estimate(x, sigma_current).to(torch.float32)
        #     dx = - sigma_current*score

        #     x = x + dx * (sigma_next - sigma_current)

        #     if step == 0:
        #         current_score = score.detach().clone()

        # tweedie's formula:
        current_score = self.diff_model.score_estimate(x, sigma).to(torch.float32)
        x0_hat = xt + sigma**2 * current_score

        measurement_loss = self.operator.loss(x0_hat, measurement).sum()
        cond_grad = - torch.autograd.grad(outputs=measurement_loss, inputs=x_start)[0]
        x_start.detach_()
        
        return current_score, cond_grad
    
    def get_lr_for_sigma(self, lr, lr_min_ratio, sigma_ratio, rho=7):
        """Uses rho as reciprocal of what is used in sigma scheduling."""
        multiplier = ((1)**rho*sigma_ratio + (1-sigma_ratio)*lr_min_ratio**rho)**(1/rho)
        return lr*multiplier
    
    def get_lr(self, lr, lr_min_ratio, ratio, rho=1):
        """Calculates the learning rate based on the ratio.
        Args:
            lr (float): Initial learning rate.
            lr_min_ratio (float): Minimum learning rate ratio.
            ratio (float): Ratio to adjust the learning rate. 
                For polynomial decay, ratio is increasing i.e. idx/num_steps.,
                for rate in terms of variances, ratio is the ratio of sigma_t/sigma_max.
        """
        # ratio is between 0 and 1
        multiplier = ((1-ratio)*1**(1/rho) + ratio*lr_min_ratio*1**(1/rho))**rho
        return lr * multiplier



    def regularized_least_squares_loss(self, x, y, x0, lam= 0.1) -> torch.Tensor:

        data_loss = torch.mean((self.operator(x) - y) ** 2)
        reg_loss = lam * torch.mean((x - x0) ** 2)
        return data_loss + reg_loss

    @torch.enable_grad()
    def optimize_image_adam(
        self,
        x_init,
        y,
        x0_hat,
        lam=0.1,
        lr=0.01,
        num_iters=50,
        tol=1.e-3,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Optimize an image tensor x to match target y using Adam with regularized least squares loss.

        Args:
            x_init: Initial image tensor of shape (B, C, H, W)
            y: Target image tensor of same shape as x_init
            x0_hat: Reference image tensor for regularization
            lam: Regularization weight
            lr: Learning rate
            num_iters: Number of optimization steps
            device: Device to run optimization
            verbose: Whether to print loss during optimization

        Returns:
            Optimized image tensor (same shape as x_init)
        """
        device = self.device
        x = x_init.clone().detach().to(device)
        y = y.to(device)
        x0_hat = x0_hat.to(device)

        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.regularized_least_squares_loss(x, y, x0_hat, lam=lam)
            loss.backward()
            optimizer.step()

            if loss.item() < tol:
                if verbose:
                    print(f"Converged at iteration {i+1} with loss {loss.item():.6f}")
                break

            if verbose and (i % max(1, num_iters // 10) == 0):
                print(f"Iter {i+1}/{num_iters}, Loss: {loss.item():.6f}")

        return x.detach()

    
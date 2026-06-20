import re
import yaml
import librosa
import torch
from PIL import Image
from pathlib import Path
import torchvision as tvt
import numpy as np
import pandas as pd

import sys
import logging
logger = logging.getLogger(__name__)

def set_up_logging(level=None):
    # Set up logging configuration

    fmt = '%(levelname)s:%(message)s'
    if level is None or level == 'info':
        level = logging.INFO 
    elif level == 'debug':
        level = logging.DEBUG
        fmt = '%(levelname)s:%(name)s:%(message)s'
    elif level == 'warning':
        level = logging.WARNING
    logging.basicConfig(
        level=level,    
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),  
        ]
    )
    # Suppress DEBUG logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

whisper_spect_config = {
    'sampling_rate': 16000,
    'n_fft': 400,
    'hop_len': 160,
    'win_len': 400,
    'window': 'hann',
    'n_mels': 80,
    'power': 2.0,
    'mel_scale': 'slaney',
}

def whisper_spect_inverse(spect, invert_normalization=True):
    """
    Whisper models compute log mel spectrograms, specified by the following
    config:
        - sampling_rate = 16000
        - n_fft = 400
        - hop_len = 160
        - win_len = 400
        - window = "hann"
        - n_mels = 80
        - power = 2.0
        - mel_scale = "slaney"
    it takes the following steps:
        - compute stft
        - compute power spectrogram (magnitude^2)
        - apply mel filter bank
        - apply log10
    Args:
        spect: (n_mels, n_frames), expecting log mel spectrogram computed by whisper
    """
    if isinstance(spect, torch.Tensor):
        spect = spect.cpu().detach().numpy()
    if invert_normalization:
        spect = spect * 4 - 4
    mel_spectrogram = 10**spect # log-mel to mel
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram,
        sr=whisper_spect_config['sampling_rate'],
        n_fft=whisper_spect_config['n_fft'],
        hop_length=whisper_spect_config['hop_len'],
        win_length=whisper_spect_config['win_len'],
        window=whisper_spect_config['window'],
        )
    return audio



def sanitize_string(s):
    """Sanitize strings for safe filenames."""
    return re.sub(r'[^\w.-]', '_', str(s))

def settings_to_dirname(settings: dict) -> str:
    parts = [f"{k}-{sanitize_string(v)}" for k, v in sorted(settings.items())]
    return "-".join(parts)

def save_image(image, path):
    """Save an image to the specified path."""
    img = Image.fromarray(image)
    img.save(path)


def load_config(config_path: str) -> dict:
    """Load a configuration file."""
    with open(config_path, 'r') as f:
        # config = yaml.safe_load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_image_batch_to_disk(image_batch, output_dir, img_names):
    """Save a batch of images to the specified directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_tensor, img_name in zip(image_batch, img_names):
        img_tensor = 0.5 * (img_tensor + 1)  # Scale to [0, 1]
        img_tensor = img_tensor.clamp(0, 1)  # Ensure values are in [0, 1]
        to_pil = tvt.transforms.ToPILImage()
        img = to_pil(img_tensor)
        img.save(output_dir / img_name)
        logging.info(f"Saved image to {output_dir / img_name}")

def save_image_to_disk(img_tensor, output_dir, filename):
    """Save a image tensor to disk.

    Args:
        img_tensor (torch.Tensor): Image tensor to save, expected shape is (C, H, W) and values in [-1, 1].
        output_dir (str or Path): Directory where the image will be saved.
        filename (str): Name of the file to save the image as.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / filename

    img_tensor = 0.5*(img_tensor + 1)

    # Step 2: Clip to [0, 1] just in case
    img_tensor = img_tensor.clamp(0, 1)

    # Step 3: Convert to PIL image (automatically scales to [0, 255] and converts to uint8)
    to_pil = tvt.transforms.ToPILImage()
    img = to_pil(img_tensor)

    # Step 4: Save the image
    img.save(image_path)
    logging.info(f"\n Saved image to {image_path}")


def write_to_disk(corr_dict, file_path):
    """Takes in the 'corr' dict and stores the results
    at the 'file_path', (concatenates if file already exists)
    
    Args:
        corr_dict (dict): 
        file_path: Pathlib object.
    """
    columns= list(corr_dict.keys())
    df = pd.DataFrame(corr_dict)
    if file_path.is_file():
        data = pd.read_csv(file_path)[columns]
        data = pd.concat([data,df], axis=0, ignore_index=True)
    else:
        data = df
    data.to_csv(file_path, index=False)
    logger.info(f"Data saved to: '{file_path}'")
    return data


def inverse_interpolation(sigma, rho, start_value, end_value):
    """Inverse interpolation to find the time corresponding to a given sigma."""
    return (start_value**(1/rho) - sigma**(1/rho)) / (start_value**(1/rho) - end_value**(1/rho))

def interpolation(t, rho, start_value, end_value):
    """Interpolation to find the sigma corresponding to a given time."""
    return (start_value**(1/rho) - (start_value**(1/rho) - end_value**(1/rho)) * t)**rho

def get_lr_for_sigma(sigma, rho, sigma_start, sigma_end, lr_rho, initial_lr, final_lr):
    """Get the learning rate for a given sigma."""
    t = inverse_interpolation(sigma, rho, sigma_start, sigma_end)
    return initial_lr*interpolation(t, lr_rho, 1, final_lr)



# DiffStateGrad helper method
def compute_rank_for_explained_variance(singular_values, explained_variance_cutoff):
    """
   Computes average rank needed across channels to explain target variance percentage.
   
   Args:
       singular_values: List of arrays containing singular values per channel
       explained_variance_cutoff: Target explained variance ratio (0-1)
   
   Returns:
       int: Average rank needed across RGB channels
   """
    total_rank = 0
    for channel_singular_values in singular_values:
        squared_singular_values = channel_singular_values ** 2
        cumulative_variance = np.cumsum(squared_singular_values) / np.sum(squared_singular_values)
        rank = np.searchsorted(cumulative_variance, explained_variance_cutoff) + 1
        total_rank += rank
    return int(total_rank / 3)

def compute_svd_and_adaptive_rank(z_t, var_cutoff):
    """
    Compute SVD and adaptive rank for the input tensor.
    
    Args:
        z_t: Input tensor (current image representation at time step t)
        var_cutoff: Variance cutoff for rank adaptation
        
    Returns:
        tuple: (U, s, Vh, adaptive_rank) where U, s, Vh are SVD components
               and adaptive_rank is the computed rank
    """
    # Compute SVD of current image representation
    U, s, Vh = torch.linalg.svd(z_t[0], full_matrices=False)
    
    # Compute adaptive rank
    s_numpy = s.detach().cpu().numpy()

    adaptive_rank = compute_rank_for_explained_variance([s_numpy], var_cutoff)
    
    return U, s, Vh, adaptive_rank

def apply_diffstategrad(norm_grad, iteration_count, period, U=None, s=None, Vh=None, adaptive_rank=None):
    """
    Compute projected gradient using DiffStateGrad algorithm.
    
    Args:
        norm_grad: Normalized gradient
        iteration_count: Current iteration count
        period: Period of SVD projection
        U: Left singular vectors from SVD
        s: Singular values from SVD
        Vh: Right singular vectors from SVD
        adaptive_rank: Computed adaptive rank
        
    Returns:
        torch.Tensor: Projected gradient if period condition is met, otherwise original gradient
    """
    if period != 0 and iteration_count % period == 0:
        if any(param is None for param in [U, s, Vh, adaptive_rank]):
            raise ValueError("SVD components and adaptive_rank must be provided when iteration_count % period == 0")
        
        # Project gradient
        A = U[:, :, :adaptive_rank]
        B = Vh[:, :adaptive_rank, :]
        
        low_rank_grad = torch.matmul(A.permute(0, 2, 1), norm_grad[0]) @ B.permute(0, 2, 1)
        projected_grad = torch.matmul(A, low_rank_grad) @ B
        
        # Reshape projected gradient to match original shape
        projected_grad = projected_grad.float().unsqueeze(0)  # Add batch dimension back
        
        return projected_grad
    
    return norm_grad
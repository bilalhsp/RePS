"""
Inspired by the DAPS sampling script: https://github.com/zhangbingliang2019/DAPS.git

This script is used to generate samples from posterior distribution i.e. p(x|y). 
It uses unconditioned diffusion model as prior p(x) and
measurement model p(y|x) to generate samples.

Example usage:
    python sampling.py sampler=reps task_group=pixel data=test-ffhq model=ffhq task=matrix_super_resolution
"""
# sampling.py
# ------------------  set up logging ----------------------
import logging
from utils import set_up_logging
set_up_logging()

import yaml
import time
import logging
import shutil
import hydra
import numpy as np
from omegaconf import OmegaConf

import setproctitle
from pathlib import Path
from PIL import Image
import torch
import torchvision as tvt
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from torchvision.utils import save_image


# local imports
from model import get_model
from diffusion import VPEpsilon, get_sampler
from forward_operator import get_operator
from evals import get_eval_fn, get_eval_fn_cmp, Evaluator, calculate_fid

from dotenv import load_dotenv
load_dotenv()
import os, wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))


@hydra.main(version_base='1.3', config_path="configs", config_name="default")
def main(args):

    setproctitle.setproctitle(args.task.task_config.operator_config.name)
    output_root_dir = Path(args.output_root_dir)
    if args.wandb:
        # initialize a run
        wandb.init(
                project=args.project_name,
                name=args.project_name,
                config=OmegaConf.to_container(args, resolve=True, structured_config_mode=False),
                dir=output_root_dir,
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # measure operator and noise
    task_config = args.task.task_config
    operator = get_operator(**task_config.operator_config)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # dataset and dataloader
    dataset = ImageDataset(**args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    diff_net = get_model(**args.model).to(device)
    diff_model = VPEpsilon(diff_net)

    setting = {**args.task[args.task_group].sampler_config, **args.sampler.sampler_config}
    # sub_dir = utils.settings_to_dirname(args.sampler.sampler_config)
    sub_dir = f"NFE-{int(setting['n_restarts']*setting['n_ode_steps'])}"
    output_dir = Path(output_root_dir, args.data.name, args.task.task_name, args.sampler.name, sub_dir)

    # create the sampler
    full_samples = []
    latent = args.sampler.get("latent", False)

    if latent:
        image_size = args.model.ldm_config.model.params.image_size
        latent_channels = args.model.ldm_config.model.params.first_stage_config.params.embed_dim
        shape = (latent_channels, image_size, image_size)
    else:
        shape = (3, args.data.resolution, args.data.resolution)

    sampler = get_sampler(args.sampler.name, diff_model, operator, latent)
    sampling_start_time = time.time()
    for run_id in range(args.num_runs):
        lpips_run, psnr_run, ssim_run = [], [], []
        gen_samples = []
        gt_images = []
        y_images = []
        for batch in dataloader:
            ref_imgs, img_names = batch
            ref_imgs = ref_imgs.to(device)
            # generate sample
            measurement = sampler.get_measurement_signal(ref_imgs)
            generated_images = sampler.generate_sample(measurement, shape=shape, **setting)[0]

            if args.save_samples:
                pil_image_list = tensor_to_pils(generated_images)
                image_dir = safe_dir(output_dir / 'samples')
                measurements_dir = safe_dir(output_dir / 'measurements')

                resized_measurement = resize(measurement, generated_images, args.task.task_config.operator_config.name)
                pil_measurement_list = tensor_to_pils(resized_measurement)
                
                for idx in range(generated_images.shape[0]):
                    image_path = image_dir / f'{img_names[idx]}_run{run_id:03d}.png'
                    pil_image_list[idx].save(str(image_path))
                    gen_path = measurements_dir / f'{img_names[idx]}_run{run_id:03d}.png'
                    pil_measurement_list[idx].save(str(gen_path))

            gen_samples.append(generated_images)
            if run_id == 0:
                gt_images.append(ref_imgs)
                y_images.append(measurement)

        full_samples.append(torch.cat(gen_samples, dim=0))
        if run_id == 0:
            images = torch.cat(gt_images, dim=0)
            y = torch.cat(y_images, dim=0)

    total_sampling_time = time.time() - sampling_start_time
    full_samples = torch.stack(full_samples, dim=0)
    # evaluate and log metrics
    results = evaluator.report(images, y, full_samples) 
    if args.wandb:
        evaluator.log_wandb(results, images.shape[0])

    markdown_text = evaluator.display(results) 
    

    logging.info(f"{markdown_text}")
    if args.save_samples:
        # log grid results
        resized_y = resize(y, images, args.task.task_config.operator_config.name)
        stack = torch.cat([images, resized_y, full_samples.flatten(0, 1)])
        save_image(stack * 0.5 + 0.5, fp=str(output_dir / 'grid_results.png'), nrow=images.shape[0])
    logging.info(f"\n Total time for generating samples: {(total_sampling_time)/60:.2f} minutes")  
    logging.info(f"Time per sample: {(total_sampling_time)/args.num_runs/len(dataset):.2f} seconds")  
    
    # evaluate FID score
    if args.eval_fid:
        print('Calculating FID...')
        fid_dir = safe_dir(output_dir / 'fid')
        # select the best samples based on the best of the all runs
        full_samples # [num_runs, B, C, H, W]
        eval_fn_cmp = get_eval_fn_cmp(evaluator.main_eval_fn_name)
        eval_values = np.array(results[evaluator.main_eval_fn_name]['sample']) # [B, num_runs]
        if eval_fn_cmp == 'min':
            best_idx = np.argmin(eval_values, axis=1)
        elif eval_fn_cmp == 'max':
            best_idx = np.argmax(eval_values, axis=1)
        best_samples = full_samples[best_idx, np.arange(full_samples.shape[1])]
        # save the best samples
        best_sample_dir = safe_dir(fid_dir / 'best_sample')
        pil_image_list = tensor_to_pils(best_samples)
        for idx in range(len(pil_image_list)):
            image_path = best_sample_dir / '{:05d}.png'.format(idx)
            pil_image_list[idx].save(str(image_path))

        fake_dataset = ImageDataset(
            name=args.data.name, data_dir=str(best_sample_dir), resolution=args.data.resolution, start_idx=0, end_idx=len(best_samples)
            )
        real_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)

        fid_score = calculate_fid(real_loader, fake_loader)
        print(f'FID Score: {fid_score.item():.4f}')
        with open(str(fid_dir / 'fid.txt'), 'w') as file:
            file.write(f'FID Score: {fid_score.item():.4f}')
        if args.wandb:
            wandb.log({'FID': fid_score.item()})

def remove_dir_contents(dir_path):
    """Remove all files in the given directory."""
    
    if not dir_path.exists():
        logging.warning(f"Directory {dir_path} does not exist. Nothing to remove.")
        return
    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)



def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    return Path(dir)


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)

class ImageDataset(Dataset):
    def __init__(self, data_dir, resolution=256, start_idx=None, end_idx=None, **kwargs):
        """Initialize the dataset with the path to the images directory."""
        self.images_dir = Path(data_dir)
        extenssions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        self.all_image_files = [file for ext in extenssions for file in self.images_dir.rglob(ext)]
        self.all_image_files = sorted(self.all_image_files)
        self.all_image_files = self.all_image_files[start_idx:end_idx]

        self.transforms = tvt.transforms.Compose([
            tvt.transforms.ToTensor(),
            tvt.transforms.Resize(resolution),
            tvt.transforms.CenterCrop(resolution),
            ])
        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_available_images(self):
        """Return a list of available image files in the dataset."""
        return [img for img in self.all_image_files if img.is_file()]

    def get_shape(self):
        return (3, self.resolution, self.resolution)
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.all_image_files)
    
    def __getitem__(self, idx):
        """Return the image at the specified index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds for dataset.")

        img_path = self.all_image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img)*2 - 1
        img_tensor = img_tensor.to(self.device)
        # img_tensor = self.transforms(img).unsqueeze(0)
        return img_tensor, img_path.name
    


if __name__ == "__main__":

    START_TIME = time.time()
    main()
    logging.info(f"Total time taken: {(time.time()-START_TIME)/60} minutes")
    


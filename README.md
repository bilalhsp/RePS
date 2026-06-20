# Restart Posterior Sampling (RePS)

This repository accompanies the paper [Solving Diffusion Inverse Problems with Restart Posterior Sampling](https://arxiv.org/abs/2511.20705). It provides Hydra-based sampling pipelines for image inverse problems, including pixel-space and latent-space settings, and implements Restart Posterior Sampling (RePS) for generating posterior samples from diffusion models.

## What is included

- `sampling.py`: the main entrypoint for running posterior sampling experiments.
- `configs/`: Hydra configuration files for datasets, models, samplers, and tasks.
- `diffusion/`: diffusion and sampler implementations.
- `forward_operator/`: measurement operators for inverse problems.
- `model/`: pretrained model loading and model wrappers.
- `evals/`: metrics and evaluation utilities.

## Requirements

This codebase is designed for a Python environment with PyTorch and the usual scientific stack installed. A typical setup includes:

- Python 3.9 or newer
- PyTorch with CUDA support if you plan to run on GPU
- Hydra
- OmegaConf
- NumPy
- Pillow
- torchvision
- wandb
- PyYAML
- python-dotenv
- setproctitle

Create and activate your preferred virtual environment, then install the project dependencies used by your environment.

## Running sampling

The main entrypoint is [sampling.py](sampling.py). You can run it directly with Hydra overrides:

```bash
python sampling.py \
	sampler=reps \
	task_group=pixel \
	data=test-ffhq \
	model=ffhq \
	task=matrix_super_resolution
```

For a latent-space run, switch the task group and model accordingly:

```bash
python sampling.py \
	sampler=latent_reps \
	task_group=ldm \
	data=test-ffhq \
	model=ffhqldm \
	task=inpaint_box
```

You can override sampler settings on the command line as needed, for example:

```bash
python sampling.py \
	sampler=reps \
	task_group=pixel \
	data=test-ffhq \
	model=ffhq \
	task=matrix_super_resolution \
	num_runs=10 \
	batch_size=25 \
	save_samples=True \
	eval_fid=True
```

## Reproducibility notes

- The code seeds NumPy and PyTorch at the start of each run.
- `wandb` logging can be enabled or disabled with the `wandb` config flag.
- The main sampling script prints the resolved Hydra config at runtime, which makes it easier to reproduce a run later.

## Citation

If you use this code in your work, please cite:

```bibtex
@inproceedings{ahmed2026reps,
  title     = {Solving Diffusion Inverse Problems with Restart Posterior Sampling},
  author    = {Ahmed, Bilal and Makin, Joseph G.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url        = {https://arxiv.org/abs/2511.20705}
}
```

## Attribution

This repository was inspired by and is based on the following projects:

- [DAPS](https://github.com/zhangbingliang2019/DAPS)
- [DPS](https://github.com/dps2022/diffusion-posterior-sampling)
- [RED-diff](https://github.com/NVlabs/RED-diff)
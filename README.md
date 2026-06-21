# Restart Posterior Sampling (RePS)

[![Project Page](https://img.shields.io/badge/Project-Page-176b87)](https://bilalhsp.github.io/RePS/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.20705-b31b1b)](https://arxiv.org/abs/2511.20705)

Official code for **Solving Diffusion Inverse Problems with Restart Posterior Sampling**. RePS is a posterior sampler for diffusion inverse problems: it alternates short measurement-conditioned ODE trajectories with restart noise injections, producing diverse posterior samples without backpropagating through the score network. See the [project page](https://bilalhsp.github.io/RePS/) for figures and a method overview.

## Code Overview

- `sampling.py`: Hydra entrypoint for running RePS and baseline posterior sampling experiments.
- `diffusion/reps.py`: Restart Posterior Sampling implementation.
- `configs/sampler/reps.yaml`: pixel-space RePS sampler configuration.
- `configs/sampler/latent_reps.yaml`: latent-space RePS sampler configuration.
- `configs/task/`: inverse-problem definitions, including super-resolution, inpainting, deblurring, HDR, phase retrieval, and nonlinear deblurring.
- `forward_operator/`: measurement operators used by the inverse-problem tasks.
- `evals/`: PSNR, SSIM, LPIPS, and FID evaluation utilities.

## Running RePS

The default configuration runs pixel-space RePS on FFHQ. Override Hydra fields to select the dataset, model, sampler, and inverse problem.

```bash
python sampling.py \
  sampler=reps \
  task_group=pixel \
  data=test-ffhq \
  model=ffhq \
  task=matrix_super_resolution
```

For latent-space RePS, use the latent sampler and LDM task group:

```bash
python sampling.py \
  sampler=latent_reps \
  task_group=ldm \
  data=test-ffhq \
  model=ffhqldm \
  task=inpaint_box
```

Useful experiment overrides:

```bash
python sampling.py \
  sampler=reps \
  task_group=pixel \
  data=test-imagenet \
  model=imagenet \
  task=motion_deblur \
  num_runs=4 \
  batch_size=25 \
  save_samples=True \
  eval_fid=True
```

Outputs are written under `output_root_dir` from `configs/default.yaml`; override it on the command line for your machine:

```bash
python sampling.py output_root_dir=/path/to/outputs
```

## Supported Tasks

Configured tasks include:

- Linear inverse problems: `super_resolution`, `matrix_super_resolution`, `inpaint_box`, `inpaint_random`, `gaussian_deblur`, `motion_deblur`
- Nonlinear inverse problems: `phase_retrieval`, `nonlinear_deblur`, `hdr`

The repository also includes DAPS configuration support via `configs/sampler/daps.yaml` for comparisons.

## Citation

If you use this code in your work, please cite:

```bibtex
@inproceedings{ahmed2026reps,
  title     = {Solving Diffusion Inverse Problems with Restart Posterior Sampling},
  author    = {Ahmed, Bilal and Makin, Joseph G.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.20705}
}
```

## Attribution

This repository builds on code and ideas from:

- [DAPS](https://github.com/zhangbingliang2019/DAPS)
- [DPS](https://github.com/dps2022/diffusion-posterior-sampling)
- [RED-diff](https://github.com/NVlabs/RED-diff)

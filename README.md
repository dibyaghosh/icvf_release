# ICVF (Reinforcement Learning from Passive Data via Latent Intentions)

This repository contains accompanying code for the paper [Reinforcement Learning from Passive Data via Latent Intentions](https://arxiv.org/abs/2304.04782).

The code is built off of jaxrl_m (see [dibyaghosh/jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) for better documentation) -- all new code is in the src/ directory. 

## Installation

Add this directory to your PYTHONPATH. Install the dependencies for jaxrl_m (the usual suspects: jax, flax, optax, distrax, wandb, ml_collections), and additional dependencies depending on which environments you want to try (see requirements.txt).

The XMagical dataset is available on [Google Drive](https://drive.google.com/drive/folders/1qDiOoKrWUybJBB4dIzz6-lWy7Z1MAYro?usp=sharing)


### Examples

To train an ICVF agent on the Antmaze dataset, run:

```
python experiments/antmaze/train_icvf.py --env_name=antmaze-large-diverse-v2
```


To train an ICVF agent on the XMagical dataset, run:

```
python experiments/xmagical/train_icvf.py
```


### Code Structure:

- [jaxrl_m/](jaxrl_m/): A carbon copy of https://github.com/dibyaghosh/jaxrl_m
- [icvf_envs/](icvf_envs/): Environment wrappers and dataset loaders
- [src/](src/): New code for ICVF
    - [icvf_learner.py](src/icvf_learner.py): Core algorithmic logic
    - [icvf_networks.py](src/icvf_networks.py): ICVF network architecture
    - [extra_agents/](src/extra_agents/): Finetuning downstream RL agents from the ICVF representation
- [experiments/](experiments/): Launchers for ICVF experiments


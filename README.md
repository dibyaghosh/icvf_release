# ICVF (Reinforcement Learning from Passive Data via Latent Intentions)

This repository contains accompanying code for the paper [Reinforcement Learning from Passive Data via Latent Intentions](TODO).

The code is built off of jaxrl_m (see [dibyaghosh/jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) for better documentation) -- all new code is in the src/ directory. 

## Installation

Add this directory to your PYTHONPATH. Install the dependencies for jaxrl_m (the usual suspects: jax, flax, optax, distrax, wandb, ml_collections), and additional dependencies depending on which environments you want to try (see requirements.txt).


### Code Structure:

- [jaxrl_m/](jaxrl_m/): A carbon copy of https://github.com/dibyaghosh/jaxrl_m
- [icvf_envs/](icvf_envs/): Environment wrappers and dataset loaders
- [src/](src/): New code for ICVF
    - [icvf_learner.py](src/icvf_learner.py): Core algorithmic logic
    - [icvf_networks.py](src/icvf_networks.py): ICVF network architecture
    - [extra_agents/](src/extra_agents/): Finetuning downstream RL agents from the ICVF representation
- [experiments/](experiments/): Launchers for ICVF experiments

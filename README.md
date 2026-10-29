# MetaSet

MetaSet addresses a key limitation of applying deep reinforcement learning (DRL) to reconfigurable intelligent surface (RIS) aided multi-user multiple-input single-output (MU-MISO) systems: a conventional DRL agent is built on **flattened fixed-length state/action vectors**, so its network shape changes whenever the system size (number of base-station antennas `M`, RIS elements `L`, or users `K`) changes, and the trained parameters cannot be reused. MetaSet replaces the flattened interface with a **permutation-invariant set encoding** and combines it with **Reptile meta-learning across system sizes**, so a single network (with a constant 84,615 parameters) serves any `(M, L, K)` configuration and transfers to unseen sizes with a strong meta-initialization.

## Key Ideas

- **Set-based dimension-agnostic interface.** The cascaded channel observation is organized as structured token sets `D [L, K, M, 2]` (element x user x antenna x real/imag); the system size appears only through the set cardinalities. The actor and critic weights are fully decoupled from `(M, L, K)`.
- **Three-level set encoding.** `(k, m)` antenna pairs are pooled into element-level and user-level tokens, then mapped to the RIS phase-shift and per-user beamforming action dimensions.
- **Decomposed critic.** The action value is decomposed over the set elements, `Q = sum_l q_phi + sum_{k,m} q_G + q_global`, giving every action dimension an O(1) direct gradient.
- **Differentiable constraints.** The unit-modulus RIS phase constraint and the total transmit-power constraint are applied *inside* the network; exploration noise is added before normalization and then projected back onto the feasible set.
- **Cross-size meta-learning.** Reptile treats different `(M, L, K)` physical configurations as the meta-task distribution (rather than different channel realizations of a fixed configuration), yielding a size-generalizable meta-initialization.

## Repository Structure

```
.
├── environment.py             # RIS-MISO channel environment (sum-rate reward; from Huang et al., JSAC 2020)
├── DDPG.py                    # Baseline DDPG agent (flattened fixed-shape network)
├── utils.py                   # Replay buffer and utilities for the baseline
├── main_Benchmark.py          # Train/evaluate the baseline DDPG on T1-T6
├── main_Benchmark_metaset.py  # Train/evaluate the MetaSet agent (meta-init or scratch) on T1-T6
├── train_metaset.py           # Reptile meta-training across sizes + unseen-size evaluation
├── plot_results.ipynb         # Plot training-curve comparisons and reproduce Table I statistics
├── metaset_drl/               # Dimension-agnostic MetaSet package
│   ├── env_wrapper.py         # Cascaded-channel token wrapper; supports M != K; reward normalization
│   ├── set_networks.py        # SetActor / SetCritic (three-level set encoding + decomposed Q)
│   ├── metaset_ddpg.py        # MetaSet DDPG agent, Reptile meta-training, zero-shot evaluation
│   └── README.md              # Detailed package documentation
├── Learning Curves/           # Recorded training curves (42 .npy files, used by the notebook)
└── metaset_results/           # Meta-trained checkpoint (metaset_meta.pt)
```

## Environment Setup

- Python 3.8+
- PyTorch
- NumPy, Matplotlib, Pandas

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

All commands are run from the repository root.

### 1. Baseline DDPG (fixed-shape network)

```bash
# T1 (4x4x4), parameter set P1
python main_Benchmark.py \
    --num_antennas 4 --num_RIS_elements 4 --num_users 4 \
    --seed 300000 --channel_seed 300000
```

### 2. MetaSet: cross-size meta-training

```bash
# Meta-training configuration set is given as MxLxK entries
python train_metaset.py --ris-root . --meta-tasks 2x4x2,4x8x4,8x16x8,4x8x2 --seed 0 --out metaset_inits/s0
```

This produces the meta-parameter checkpoint `metaset_inits/metaset_meta.pt`, numerical results, and adaptation curves.

### 3. MetaSet on T1-T6 (meta-initialization vs. random initialization)

```bash
# Fine-tuning from the meta-initialization (P3 disables exploration noise during adaptation)
python main_Benchmark_metaset.py --ris-root . --mode meta \
    --meta_init metaset_inits/metaset_meta.pt \
    --num_antennas 4 --num_RIS_elements 4 --num_users 4 \
    --seed 300000 --channel_seed 300000 \
    --exploration_noise 0.0 --out outputs

# Training from scratch (random initialization) by parameter set P1
python main_Benchmark_metaset.py --ris-root . --mode scratch \
    --num_antennas 4 --num_RIS_elements 4 --num_users 4 \
    --seed 300000 --channel_seed 300000 --out outputs
```

### 4. Reproduce the figures and Table I

Open `plot_results.ipynb` and run all cells. It loads the recorded curves in `Learning Curves/`, plots the per-scenario training-curve comparison, and reports the zero-shot / final / peak statistics.

## Conventions

- **Transmit power.** The maximum transmit power is `P_t = 30 dBm = 1 W`. In code, `power_t_W = 10 ** (30 / 10) = 1000` (mW), and the beamforming vectors are normalized so that `Tr(G G^H) = 1000` mW.
- **Reward.** The reward is the raw sum rate `(bps/Hz)`. For cross-size training it is normalized by the rate of the identity action `(G = I, Phi = I)` on the same channel realization; the identity action equals 1.0, a fixed per-channel constant that does not change the optimal policy.
- **Imperfect CSI.** A single knob, `--channel_noise_var` (sigma_e^2), controls the cascaded-channel estimation error. `0` (default) means perfect CSI (`Dhat = D`); `> 0` adds `CN(0, sigma_e^2)` estimation noise per user at every step, while the reward is always computed on the true channels. See `metaset_drl/README.md` for details.
- **Other hyper-parameters.** Discount factor `gamma = 0.99`, channel AWGN variance `sigma^2 = 1e-2`, Adam optimizer; the exploration noise (std 0.1) is annealed to 20%. The P1/P2/P3 parameter sets follow the paper's experimental setup.

## Test Scenarios (T1-T6)

The six test configurations `(M, L, K)` are:

| Scenario | (M, L, K) |
|---|---|
| T1 | (4, 4, 4) |
| T2 | (6, 12, 6) |
| T3 | (6, 12, 3) |
| T4 | (16, 16, 8) |
| T5 | (8, 32, 8) |
| T6 | (8, 16, 16) |

## Data

`Learning Curves/` contains 42 `.npy` files (about 32 MB), each of shape `[10 episodes, 10000 steps]`, and `metaset_results/metaset_meta.pt` is the provided meta-trained checkpoint. These are included so that the comparison notebook runs out of the box. They can be regenerated from the training scripts.

## Acknowledgements

This repository builds on the open-source RIS DRL code by Saglam et al., which extends the DDPG-based RIS design of Huang et al. (JSAC 2020):

- B. Saglam, D. Gurgunoglu, and S. S. Kozat, "Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration in RIS-Aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI," IEEE ICC Workshops, 2023 — [code](https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning)

The inherited code is released under the MIT License (see [LICENSE](LICENSE)).

Relative to these repositories, MetaSet:

- **Removes the phase-dependent amplitude (PDA) model and hardware-impairment mismatch** (the `_compute_PDA` map, angle-dependent beta, and `Phi_mismatch`).
- **Unifies the observation on the cascaded channel** `D = diag(h_2) H_1`, with imperfect CSI `Dhat = D + E`, `E ~ CN(0, sigma_e^2)`, resampled at every step (in `metaset_drl/env_wrapper.py`).
- **Removes the `M == K` restriction**, supporting arbitrary antenna/user counts via a truncated-identity init and `M*K` action slicing.
- **Replaces the fixed-shape MLP with a dimension-agnostic set actor-critic** (`SetActor`/`SetCritic`, 84,615 parameters) whose weights are independent of `(M, L, K)`.
- **Adds Reptile meta-learning across system sizes** (different `(M, L, K)` configurations), rather than across channel realizations of one size.
- **Normalizes the reward by the identity-action rate** on the same channel realization, and applies the unit-modulus and total-power constraints differentiably inside the network.

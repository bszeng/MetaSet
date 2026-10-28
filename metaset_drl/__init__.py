"""
MetaSet-DRL: dimension-agnostic deep reinforcement learning for RIS-MISO
(cascaded channel observation).

Adapted from the RIS DDPG code of Huang et al. (JSAC 2020):
- A2 set encoding: the state/action representation is fully decoupled from
  (M, N, K) (set_networks.py)
- B1 cross-size meta-learning: Reptile meta-training over the task distribution
  defined by (M, N, K) (metaset_ddpg.py)

The observation model is the cascaded channel estimation in paper 2
(Saglam et al., ICCW 2023), Eq. (5):
Dhat_k = diag(h_k) H_1 + E_k; channel_noise_var=0 is perfect CSI (D is a
sufficient statistic).
"""

from metaset_drl.env_wrapper import RISSetEnv
from metaset_drl.set_networks import SetActor, SetCritic
from metaset_drl.metaset_ddpg import MetaSetDDPG

__all__ = ["RISSetEnv", "SetActor", "SetCritic", "MetaSetDDPG"]

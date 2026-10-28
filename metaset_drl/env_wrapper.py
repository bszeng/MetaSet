"""
Dimension-agnostic wrapper for the RIS_MISO environment (cascaded channel
observation; single observation system).

The two limitations of the original environment.RIS_MISO are removed here:
1. The original environment asserts M == K (the number of antennas must equal
   the number of users) -- this wrapper bypasses that assertion and implements
   its own step logic (the original step parses the action in M**2 slices, which
   is also correct only for M == K);
2. The original state/action are flattened fixed-length vectors (dimension =
   f(M, L, K)) -- this wrapper replaces them with a structured token dict in
   which all dimensional information appears only through the set cardinalities.

Observation model (paper 2, Saglam et al., ICCW 2023, Eq. (5)): a passive RIS
can only estimate the cascaded channel; the observation is Dhat_k =
diag(h_k) H_1 + E_k, where the entries of E_k are i.i.d. CN(0,
channel_noise_var), independent across users, and resampled at every step.
When channel_noise_var = 0, Dhat = D exact -- since the sum rate depends on
(H_1, H_2) only through the cascaded channel D_k, D is a sufficient statistic
for the reward, so a clean observation is equivalent in information content to
perfect CSI. The reward is always computed on the true channels (the agent
decides from estimates; the environment responds with the true channels).
Note: the channel_est_error flag in the original environment.py is a dead
switch (the field is stored but noise is never added); the estimation error is
implemented in this wrapper.

Token dict convention (numpy; complex numbers split into real/imag channels):
    D:        [L, K, M, 2]  cascaded channel (estimate): element x user x antenna
    phi_prev: [L, 2]        previous RIS phase shift (own action, exactly known)
    power_t:  [K]           per-user transmit power (following the original env)
    power_r:  [K]           per-user receive power (computed from Dhat via paper Eq. (4))

Action convention (network outputs; constraints applied differentiably inside
the network):
    phi: [L, 2]    per-element unit-modulus phase shift
    G:   [K, M, 2] per-user x per-antenna beamforming (total power constraint)

Reward: raw sum rate (consistent with the paper). Note that the original env's
opt_reward must not be used for normalization -- opt_reward depends on the
current action (Phi, G), and any zero-interference action makes the ratio 1
(the metric saturates and the reward can be gamed by "collapsing the channel").
Instead, at reset the rate of the identity action (G = I, Phi = I),
ref_reward, is used as a fixed reference for that channel realization
(independent of later actions); norm_reward = raw / ref is used for training
and evaluation that are comparable across (M, L, K).
"""

import os
import sys

import numpy as np


def _import_ris_env(ris_root):
    """Add the user's RIS DDPG repository to the search path and import its environment module."""
    ris_root = os.path.abspath(ris_root)
    if ris_root not in sys.path:
        sys.path.insert(0, ris_root)
    import environment  # environment.py from the user's repository
    return environment


class RISSetEnv:
    """RIS-MISO environment wrapper with cascaded channel observation (see module docstring).

    channel_noise_var: cascaded channel estimation error variance sigma_e^2.
    0 = perfect CSI (Dhat = D), >0 = imperfect CSI of paper 2, Eq. (5). Even
    when sigma^2 = 0 random numbers are still consumed (normal(0,0) = 0),
    ensuring identical random-stream consumption across noise levels and
    keeping channel realizations paired.
    """

    def __init__(self, ris_root, num_antennas, num_RIS_elements, num_users,
                 power_t_db=30.0, AWGN_var=1e-2, max_steps=1000, seed=None,
                 channel_noise_var=0.0):
        environment = _import_ris_env(ris_root)

        class _RIS(environment.RIS_MISO):
            def __init__(self, **kw):
                # Bypass assert M == K: replicate the field initialization of the original __init__
                self.M = kw["num_antennas"]
                self.L = kw["num_RIS_elements"]
                self.K = kw["num_users"]
                self.awgn_var = kw.get("AWGN_var", 1e-2)
                self.channel_noise_var = kw.get("channel_noise_var", 0.0)
                self.action_dim = 2 * self.M * self.K + 2 * self.L
                self.state_dim = 2 * self.K + 2 * (self.L * self.M + self.L * self.K) + self.action_dim
                self.H_1 = None
                self.H_2 = None
                self.G = np.eye(self.M, self.K, dtype=complex)
                self.Phi = np.eye(self.L, dtype=complex)
                self.state = None
                self.done = None
                self.episode_t = None

        self.env = _RIS(num_antennas=num_antennas, num_RIS_elements=num_RIS_elements,
                        num_users=num_users, AWGN_var=AWGN_var,
                        channel_noise_var=channel_noise_var)
        self.M, self.L, self.K = num_antennas, num_RIS_elements, num_users
        self.power_t_W = 10 ** (power_t_db / 10.0)
        self.max_steps = max_steps
        self._seed = seed

    # ------------------------------------------------------------------ #
    def _obs(self):
        """Paper 2, Eq. (5): Dhat_k = diag(h_k) H_1 + E_k (independent per user,
        resampled every step; when sigma^2 = 0 Dhat = D exact). Returns
        (D_hat [K,L,M] complex, power_r [K])."""
        D = np.stack([np.diag(self.env.H_2[:, k]) @ self.env.H_1
                      for k in range(self.K)])                        # [K,L,M]
        s = np.sqrt(self.env.channel_noise_var / 2)
        D = D + np.random.normal(0, s, D.shape) + 1j * np.random.normal(0, s, D.shape)
        phi = np.diag(self.env.Phi)                                   # own action, exactly known
        V = (phi[None, :, None] * D).sum(axis=1)                      # [K,M]: phi^T Dhat_k (Eq. (4))
        H2_tilde = V @ self.env.G                                     # [K,K]
        power_r = np.linalg.norm(H2_tilde, axis=0) ** 2               # [K]
        return D, power_r

    def _tokens(self):
        """Build the token dict from the environment's internal variables (agent view; the channel part is the estimate Dhat).
        The reward is unaffected -- the environment internally always uses the true channels."""
        phi = np.diag(self.env.Phi)
        phi_prev = np.stack([np.real(phi), np.imag(phi)], axis=-1)              # [L,2]
        power_t = np.real(np.diag(self.env.G.conjugate().T @ self.env.G)) ** 2  # [K]
        D_hat, power_r = self._obs()
        D_tokens = np.stack([D_hat.real, D_hat.imag], axis=-1)          # [K,L,M,2]
        D_tokens = np.transpose(D_tokens, (1, 0, 2, 3))                 # [L,K,M,2]
        # Power terms can reach magnitudes of 1e4~1e6; feeding them directly into a ReLU network causes activation blow-up -- log-compress to O(1)
        return {
            "D": D_tokens.astype(np.float32),
            "phi_prev": phi_prev.astype(np.float32),
            "power_t": np.log10(power_t + 1e-9).astype(np.float32),
            "power_r": np.log10(power_r + 1e-9).astype(np.float32),
        }

    def reset(self):
        if self._seed is not None:
            np.random.seed(self._seed)
        self.env.reset()
        # After the original env reset, G = I (M x K truncated) and Phi = I: the
        # rate at this point is the identity-action rate, a fixed constant for
        # this channel realization (independent of later actions), used as the
        # normalization reference.
        ref, _ = self.env._compute_reward(self.env.Phi)
        self.ref_reward = float(max(ref, 1e-9))
        return self._tokens()

    def step(self, phi, G):
        """
        phi: [L, 2] unit-modulus phase shift; G: [K, M, 2] beamforming (already satisfying the power constraint).
        Returns (tokens, raw_reward, norm_reward, ref_reward, done)
        raw_reward  : raw sum rate (paper reward)
        norm_reward : raw / ref_reward; identity action = 1.0, comparable across configurations
        """
        self.env.episode_t += 1
        self.env.Phi = np.eye(self.L, dtype=complex) * (phi[:, 0] + 1j * phi[:, 1])
        self.env.G = (G[:, :, 0] + 1j * G[:, :, 1]).T.copy()  # [K,M] -> [M,K]

        reward, _ = self.env._compute_reward(self.env.Phi)
        norm = reward / self.ref_reward
        done = bool(self.env.episode_t >= self.max_steps)
        return self._tokens(), float(reward), float(norm), float(self.ref_reward), done

    def identity_action_reward(self):
        """Raw sum rate of the identity initial action (G = I, Phi = I) -- the normalization reference itself."""
        return float(self.ref_reward)

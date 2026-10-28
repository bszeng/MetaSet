"""
MetaSet-DDPG: dimension-agnostic DDPG agent + Reptile meta-training across (M, L, K).
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from .set_networks import SetActor, SetCritic
from .env_wrapper import RISSetEnv


def _soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


class TaskBuffer:
    """Single-task replay buffer. Within a task the token sizes are identical, so batches can be stacked directly."""

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.data = []

    def add(self, tokens, phi, g, reward, next_tokens):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append((tokens, phi, g, reward, next_tokens))

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.choice(len(self.data), size=min(batch_size, len(self.data)),
                         replace=False)
        batch = [self.data[i] for i in idx]
        tokens = {k: torch.as_tensor(np.stack([b[0][k] for b in batch]),
                                     dtype=torch.float32)
                  for k in batch[0][0]}
        next_tokens = {k: torch.as_tensor(np.stack([b[4][k] for b in batch]),
                                            dtype=torch.float32)
                       for k in batch[0][4]}
        phi = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        g = torch.as_tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        r = torch.as_tensor([b[3] for b in batch], dtype=torch.float32)
        return tokens, phi, g, r, next_tokens


class MetaSetDDPG:
    """Dimension-agnostic DDPG agent with Reptile meta-training and cross-configuration transfer."""

    def __init__(self, ris_root: str, hidden: int = 64, lr_actor: float = 1e-3,
                 lr_critic: float = 1e-3, gamma: float = 0.99, tau: float = 0.001,
                 power_t_db: float = 30.0, AWGN_var: float = 1e-2,
                 channel_noise_var: float = 0.0, device: str = 'cpu'):
        self.ris_root = ris_root
        self.gamma, self.tau = gamma, tau
        self.power_t_db = power_t_db
        self.power_t_W = 10 ** (power_t_db / 10.0)
        self.AWGN_var = AWGN_var
        self.channel_noise_var = channel_noise_var
        self.device = torch.device(device)
        self.hidden = hidden
        self.actor = SetActor(hidden).to(self.device)
        self.critic = SetCritic(hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def _make_env(self, M: int, L: int, K: int, max_steps: int,
                  channel_noise_var: float = None) -> RISSetEnv:
        return RISSetEnv(self.ris_root, num_antennas=M, num_RIS_elements=L,
                         num_users=K, power_t_db=self.power_t_db,
                         AWGN_var=self.AWGN_var, max_steps=max_steps,
                         channel_noise_var=(self.channel_noise_var
                                            if channel_noise_var is None
                                            else channel_noise_var))

    def _to_dev(self, tokens: dict) -> dict:
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in tokens.items()}

    # ---------- Interaction ----------
    @torch.no_grad()
    def act(self, tokens: dict, noise_std: float = 0.0,
            torch_rng: torch.Generator = None):
        """Return constraint-satisfying actions (phi [L,2], G [K,M,2]).
        When noise_std > 0, Gaussian noise is added before normalization and then projected back onto the feasible set."""
        t = {k: v.unsqueeze(0) for k, v in self._to_dev(tokens).items()}
        phi, g = self.actor(t['D'], t['phi_prev'], t['power_t'], t['power_r'],
                            noise_std=noise_std, generator=torch_rng)
        g = SetActor.normalize_power(g, self.power_t_W)
        return (phi.squeeze(0).cpu().numpy().astype(np.float32),
                g.squeeze(0).cpu().numpy().astype(np.float32))

    # ---------- Single DDPG update ----------
    def update(self, batch) -> dict:
        tokens, phi, g, r, next_tokens = batch
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        next_tokens = {k: v.to(self.device) for k, v in next_tokens.items()}
        phi, g, r = phi.to(self.device), g.to(self.device), r.to(self.device)

        with torch.no_grad():
            phi_n, g_n = self.actor_target(next_tokens['D'], next_tokens['phi_prev'],
                                           next_tokens['power_t'], next_tokens['power_r'])
            g_n = SetActor.normalize_power(g_n, self.power_t_W)
            q_next = self.critic_target(next_tokens['D'], next_tokens['power_t'],
                                        next_tokens['power_r'], phi_n, g_n)
            y = r.unsqueeze(-1) + self.gamma * q_next  # continuing optimization task; no terminal bootstrap

        q = self.critic(tokens['D'], tokens['power_t'], tokens['power_r'], phi, g)
        critic_loss = nn.functional.mse_loss(q, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_critic.step()

        phi_a, g_a = self.actor(tokens['D'], tokens['phi_prev'],
                                tokens['power_t'], tokens['power_r'])
        g_a = SetActor.normalize_power(g_a, self.power_t_W)  # differentiable power projection
        actor_loss = -self.critic(tokens['D'], tokens['power_t'],
                                  tokens['power_r'], phi_a, g_a).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_actor.step()

        _soft_update(self.actor_target, self.actor, self.tau)
        _soft_update(self.critic_target, self.critic, self.tau)
        return {'critic_loss': float(critic_loss.detach()),
                'actor_loss': float(actor_loss.detach())}

    # ---------- Single-task training (shared by meta-training inner loop / downstream fine-tuning) ----------
    def train_on_task(self, M: int, L: int, K: int, episodes: int,
                      channel_noise_var: float = None,
                      steps_per_episode: int = 50, batch_size: int = 64,
                      warmup: int = 200, noise_std: float = 0.1,
                      seed: int = 0, log_every: int = 0) -> dict:
        """Interact and learn on a single (M, L, K) configuration; return the per-episode relative-reward curve.
        channel_noise_var can be overridden per task (used for meta-training across CSI error levels; 0 = perfect CSI)."""
        env = self._make_env(M, L, K, max_steps=steps_per_episode,
                             channel_noise_var=channel_noise_var)
        rng = np.random.default_rng(seed)
        torch_rng = torch.Generator(device='cpu').manual_seed(seed)
        buf = TaskBuffer()
        curve, noise = [], noise_std
        for ep in range(episodes):
            tokens = env.reset()
            ep_rel, steps = 0.0, 0
            for _ in range(steps_per_episode):
                phi, g = self.act(tokens, noise_std=noise, torch_rng=torch_rng)
                ntok, raw_r, norm_r, _, done = env.step(phi, g)
                buf.add(tokens, phi, g, norm_r, ntok)  # normalized reward for learning (a constant per-channel scaling that does not change the optimal policy)
                tokens = ntok
                ep_rel += norm_r
                steps += 1
                if len(buf) >= warmup:
                    self.update(buf.sample(batch_size, rng))
                if done:
                    break
            curve.append(ep_rel / steps)
            noise = max(noise_std * 0.5, noise * 0.995)  # exploration annealing
            if log_every and (ep + 1) % log_every == 0:
                print(f'    task ({M},{L},{K}) ep {ep+1}/{episodes} '
                      f'rel_reward={np.mean(curve[-log_every:]):.3f}')
        return {'M': M, 'L': L, 'K': K, 'curve': curve}

    # ---------- Reptile meta-training ----------
    def meta_train(self, tasks: list, meta_iterations: int, inner_episodes: int,
                   meta_lr: float = 0.5, seed: int = 0, log_every: int = 1,
                   **task_kwargs) -> list:
        """
        Reptile: at each iteration sample a (M, L, K) task, run the inner-loop
        training starting from the current meta-parameters, and then
        theta_meta <- theta_meta + meta_lr * (theta_task - theta_meta).
        The set encoding makes the parameters of different tasks naturally
        identical in shape, so parameter differences can be taken directly --
        something fixed-shape networks (the original DDPG_Meta) cannot do.
        """
        rng = np.random.default_rng(seed)
        history = []
        for it in range(meta_iterations):
            task = tasks[rng.integers(len(tasks))]
            overrides = task[3] if len(task) == 4 else {}
            M, L, K = task[0], task[1], task[2]
            task_agent = copy.deepcopy(self)
            log = task_agent.train_on_task(
                M, L, K, episodes=inner_episodes, seed=int(rng.integers(1e9)),
                log_every=0, **{**task_kwargs, **overrides})
            with torch.no_grad():
                for mp, tp in zip(self.actor.parameters(),
                                  task_agent.actor.parameters()):
                    mp.add_(tp - mp, alpha=meta_lr)
                for mp, tp in zip(self.critic.parameters(),
                                  task_agent.critic.parameters()):
                    mp.add_(tp - mp, alpha=meta_lr)
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)
            rec = {'iter': it, 'task': [M, L, K], 'overrides': overrides,
                   'inner_final': float(np.mean(log['curve'][-5:])),
                   'inner_first': float(np.mean(log['curve'][:5]))}
            history.append(rec)
            if log_every and (it + 1) % log_every == 0:
                print(f'  meta-iter {it+1}/{meta_iterations} task=(M{M},L{L},K{K}) '
                      f'{overrides or ""} '
                      f'inner: {rec["inner_first"]:.3f} → {rec["inner_final"]:.3f}')
        return history

    # ---------- Zero-shot evaluation ----------
    @torch.no_grad()
    def zero_shot_reward(self, M: int, L: int, K: int, episodes: int = 5,
                         steps_per_episode: int = 50,
                         channel_noise_var: float = None):
        """No exploration and no learning; directly evaluate the current policy's initial performance on a new configuration."""
        env = self._make_env(M, L, K, max_steps=steps_per_episode,
                             channel_noise_var=channel_noise_var)
        rewards, first_step, raws = [], [], []
        for _ in range(episodes):
            tokens = env.reset()
            tot, steps = 0.0, 0
            for _ in range(steps_per_episode):
                phi, g = self.act(tokens, noise_std=0.0)
                tokens, raw_r, norm_r, _, done = env.step(phi, g)
                if steps == 0:
                    first_step.append(norm_r)  # cold start: quality of the first-step action on a new channel realization
                tot += norm_r
                steps += 1
                if done:
                    break
            rewards.append(tot / steps)
            raws.append(env.ref_reward)
        return {'mean': float(np.mean(rewards)), 'std': float(np.std(rewards)),
                'first_step': float(np.mean(first_step)),
                'first_step_std': float(np.std(first_step)),
                'identity_raw_rate': float(np.mean(raws))}

    def identity_baseline(self, M: int, L: int, K: int, episodes: int = 5,
                          channel_noise_var: float = None) -> float:
        """Relative-reward baseline of the identity action (G = I, Phi = I, i.e. "learn nothing")."""
        env = self._make_env(M, L, K, max_steps=1,
                             channel_noise_var=channel_noise_var)
        vals = []
        for _ in range(episodes):
            env.reset()
            vals.append(env.identity_action_reward())
        return float(np.mean(vals))

    # ---------- Save / load ----------
    def save(self, path: str):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'power_t_db': self.power_t_db,
                    'channel_noise_var': self.channel_noise_var}, path)

    def load(self, path: str):
        if torch.__version__ >= "2.0":
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        else:
            ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

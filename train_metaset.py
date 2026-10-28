"""
MetaSet-DRL training and evaluation entry point.

Usage (run next to the original RIS DDPG repository):
    python train_metaset.py --ris-root /path/to/MetaSet-DRL \
        --meta-tasks 2x4x2,4x8x4,2x8x2\
        --meta-iters 30 --inner-episodes 40

Reward = raw sum rate (paper convention); normalization = rate / identity-action
rate on the same channel (a fixed per-channel constant, independent of the
action, comparable across configurations).
"""
import argparse
import os

from metaset_drl import MetaSetDDPG

def parse_tasks(s: str):
    """'2x4x2,4x8x4' -> [(2,4,2), (4,8,4)]; format MxLxK."""
    tasks = []
    for item in s.split(','):
        m, l, k = item.strip().split('x')
        tasks.append((int(m), int(l), int(k)))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ris-root', required=True, help='Path to the original RIS DDPG repository (containing environment.py)')
    ap.add_argument('--meta-tasks', default='2x4x2,4x8x4,8x16x8,4x8x2', help='Meta-training configuration set, format MxLxK, comma-separated')
    ap.add_argument('--meta-iters', type=int, default=30)
    ap.add_argument('--inner-episodes', type=int, default=15)
    ap.add_argument('--steps-per-episode', type=int, default=200, help='Number of interaction steps per channel realization (original paper paradigm: long optimization within a single channel)')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--meta-lr', type=float, default=0.5)
    ap.add_argument('--noise-std', type=float, default=0.1)
    ap.add_argument('--power-t-db', type=float, default=30.0)
    ap.add_argument('--channel-noise-var', type=float, default=0.0,
                    help='Cascaded channel estimation error variance sigma_e^2 (paper 2 Eq. (5), agent-level default): '
                         '0 = perfect CSI (default, Dhat = D exact), >0 = imperfect CSI (recommended 1e-2)')
    ap.add_argument('--noise-vars', default='', help='Meta-training across CSI error levels: e.g. "0,1e-2,5e-2" (0 = perfect CSI); takes the Cartesian product with the --meta-tasks sizes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='metaset_inits')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_tasks = parse_tasks(args.meta_tasks)
    # Meta-training across CSI error levels: expand the noise variance into task coverage (0 = perfect CSI; observation model unchanged)
    if args.noise_vars:
        meta_tasks = [(M, L, K, {'channel_noise_var': v})
                      for (M, L, K) in meta_tasks
                      for v in [float(x) for x in args.noise_vars.split(',')]]
        print(f'Task distribution across CSI error levels ({len(meta_tasks)} tasks): {meta_tasks}')
    common = dict(hidden=args.hidden, power_t_db=args.power_t_db, channel_noise_var=args.channel_noise_var)
    task_kw = dict(steps_per_episode=args.steps_per_episode, noise_std=args.noise_std)

    agent = MetaSetDDPG(args.ris_root, **common)
    n_params = sum(p.numel() for p in agent.actor.parameters())
    print(f'SetActor parameter count: {n_params} (independent of M/L/K, fixed)')

    # ---------------- Meta-training ----------------
    print(f'\nReptile meta-training, task set {meta_tasks}')
    history = agent.meta_train(meta_tasks, meta_iterations=args.meta_iters,
                               inner_episodes=args.inner_episodes,
                               meta_lr=args.meta_lr, seed=args.seed,
                               **task_kw)
    ckpt_path = os.path.join(args.out, 'metaset_meta.pt')
    agent.save(ckpt_path)
    print(f'Meta-parameters saved: {ckpt_path}')    

if __name__ == '__main__':
    main()

import argparse
import os
import numpy as np
import torch
from metaset_drl import MetaSetDDPG
from metaset_drl.metaset_ddpg import TaskBuffer
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--ris-root', required=True, help='Path to the original RIS DDPG repository')
    ap.add_argument('--mode', choices=['scratch', 'meta'], default='scratch')
    ap.add_argument('--meta_init', default='', help='Meta-parameter checkpoint (required for --mode meta)')
    ap.add_argument('--seed', default=300000, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    ap.add_argument("--channel_seed", default=None, type=int, help="Channel seed base; default None = same as --seed (original behavior)")
    
    ap.add_argument('--num_antennas', type=int, default=4)
    ap.add_argument('--num_RIS_elements', type=int, default=4)
    ap.add_argument('--num_users', type=int, default=4)
    ap.add_argument('--power_t', type=float, default=30)
    ap.add_argument('--awgn_var', type=float, default=0.01, help='Variance of the additive white Gaussian noise (default: 0.01)')
    ap.add_argument('--channel_noise_var', type=float, default=0.1,
                    help='Cascaded channel estimation error variance sigma_e^2 (paper 2, Eq. (5)): 0 = perfect CSI, '
                         '>0 = imperfect CSI, recommended 1e-2 (PDA repository default)')        
    ap.add_argument('--num_time_steps_per_eps', type=int, default=10000)
    ap.add_argument('--num_eps', type=int, default=10, help='Number of episodes / channel realizations (default: 10)')
    
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--exploration_noise', type=float, default=0.1, metavar='G',help='Std of Gaussian exploration noise')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--discount', type=float, default=0.99)
    ap.add_argument('--tau', type=float, default=0.001, help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    ap.add_argument("--utd", default=4, type=int, help='Gradient updates per env step (default: 4; original paper code = 1)')
    ap.add_argument("--noise_final_ratio", default=0.2, type=float,
                        help='Linear exploration-noise annealing: noise(t) = exploration_noise * max(noise_final_ratio, 1 - t/T); '
                             'set to 1.0 to fall back to constant noise (original behavior)')
    
    ap.add_argument('--out', default='outputs')

    args = ap.parse_args()

    M, L, K = args.num_antennas, args.num_RIS_elements, args.num_users
    out_dir = f'{args.out}/eps{args.num_eps}_steps{args.num_time_steps_per_eps}'
    os.makedirs(out_dir, exist_ok=True)
    csi_tag = f'_csi{args.channel_noise_var}' if args.channel_noise_var > 0 else ''
    # In meta mode, prefix the file name with the meta-run tag: the checkpoint's
    # parent directory (e.g. s0 from metaset_results/s0/metaset_meta.pt).
    # if args.mode == 'meta':
    #     meta_run = os.path.basename(os.path.normpath(os.path.dirname(args.meta_init)))
    #     meta_prefix = f'{meta_run}_'
    # else:
    #     meta_prefix = ''
    # file_name = (f'{meta_prefix}{M}_{L}_{K}_{args.power_t}_{args.lr}_{args.tau}_{args.exploration_noise}_{args.utd}_{args.seed}_'
    #              f'{args.mode}{csi_tag}')
    file_name = (f'{M}_{L}_{K}_{args.power_t}_{args.lr}_{args.tau}_{args.exploration_noise}_{args.utd}_{args.seed}_'
                 f'{args.mode}{csi_tag}')

    instant_rewards = []
    max_reward = 0.0
    channel_base = args.seed if args.channel_seed is None else args.channel_seed
    for eps in range(args.num_eps):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        agent = MetaSetDDPG(args.ris_root, lr_actor=args.lr, lr_critic=args.lr,
                            gamma=args.discount, tau=args.tau,
                            power_t_db=args.power_t, AWGN_var=args.awgn_var,
                            channel_noise_var=args.channel_noise_var)
        if args.mode == 'meta':
            if not args.meta_init:
                raise ValueError('--mode meta requires --meta_init')
            agent.load(args.meta_init)

        current_seed = channel_base + eps + 1
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        env = agent._make_env(M, L, K, max_steps=args.num_time_steps_per_eps)
        tokens = env.reset()
        eps_rewards, ep_max = [], 0.0
        rng = np.random.default_rng(current_seed)
        torch_rng = torch.Generator(device='cpu').manual_seed(current_seed)
        buf = TaskBuffer()

        for t in range(args.num_time_steps_per_eps):
            old = tokens
            noise_std = args.exploration_noise * max(args.noise_final_ratio, 1.0 - t / args.num_time_steps_per_eps)
            phi, g = agent.act(tokens, noise_std=noise_std, torch_rng=torch_rng)
            tokens, raw, norm, ref, done = env.step(phi, g)
            buf.add(old, phi, g, norm, tokens)
            eps_rewards.append(raw)
            if raw > max_reward:
                max_reward = raw
            if raw > ep_max:
                ep_max = raw
            if len(buf) >= args.batch_size:
                for _ in range(args.utd):
                    agent.update(buf.sample(args.batch_size, rng))

        instant_rewards.append(eps_rewards)
        rr = np.array(eps_rewards)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
              f'{args.mode} {args.num_antennas}_{args.num_RIS_elements}_{args.num_users} '
              f'r{eps} episode {eps+1}/{args.num_eps} (channel={current_seed}): '
              f'first-100 mean={rr[:100].mean():.2f} '
              f'last-100 mean={rr[-100:].mean():.2f} peak={ep_max:.2f}')

    np.save(f'{out_dir}/MetaSet_{file_name}', instant_rewards)

    R = np.array(instant_rewards)
    print(f'\n[{args.mode}] summary over the last realization ({args.num_eps} episodes):')
    print(f'  first-100 mean: {R[:, :100].mean():.2f} +/- {R[:, :100].mean(axis=1).std():.2f}')
    print(f'  last-100 mean: {R[:, -100:].mean():.2f} +/- {R[:, -100:].mean(axis=1).std():.2f}')
    print(f'  overall peak: {max_reward:.2f}')
    print(f'  results saved: {out_dir}/')

if __name__ == '__main__':
    main()

import argparse
import os
import numpy as np
import torch
import DDPG
import utils
import environment
from datetime import datetime

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def build_cascaded_state(env, state):

    s = np.sqrt(env.channel_noise_var / 2)
    M, L, K = env.M, env.L, env.K
    D = np.stack([np.diag(env.H_2[:, k]) @ env.H_1 for k in range(K)]) # [K,L,M]
    D = D + np.random.normal(0, s, D.shape) + 1j * np.random.normal(0, s, D.shape)
    phi = np.diag(env.Phi)                                             # own action, exact
    V = (phi[None, :, None] * D).sum(axis=1)                           # [K,M]: ϕ^T D̂_k
    power_r = np.linalg.norm(V @ env.G, axis=0) ** 2                   # [K]
    state = np.asarray(state).reshape(1, -1)
    a = 2 * M * K + 2 * L
    head = state[:, :a + K]                                            # [action, power_t]
    return np.hstack((head, power_r.reshape(1, -1), np.real(D).reshape(1, -1),
                      np.imag(D).reshape(1, -1))).reshape(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training-specific parameters
    parser.add_argument("--seed", default=300000, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--channel_seed", default=None, type=int, help="Channel seed base; default None = same as --seed (original behavior)")
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=100000, type=int, help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=16, type=int, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users')
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--num_time_steps_per_eps", default=10000, type=int, metavar='N', help='Maximum number of steps per episode (default: 20000)')
    parser.add_argument("--num_eps", default=10, type=int, metavar='N', help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--awgn_var", default=0.01, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_noise_var", default=0.1, type=float,
                        metavar='G', help='Cascaded channel estimation error variance sigma_e^2 (paper 2, Eq. (5)): '
                                          '0 = perfect CSI (Dhat = D exact); >0 = imperfect CSI, recommended 1e-2. The observation is implemented via build_cascaded_state')

    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", type=float, default=0.1, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=0.001, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--lr", default=0.001, type=float, metavar='G', help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')
    parser.add_argument('--aux_lr', type=float, default=1e-3)  # learning rate
    parser.add_argument("--utd", default=4, type=int, help='Gradient updates per env step (default: 4; original paper code = 1)')
    parser.add_argument("--noise_final_ratio", default=0.2, type=float,
                        help='Linear exploration-noise annealing: noise(t) = exploration_noise * max(noise_final_ratio, 1 - t/T); '
                             'set to 1.0 to fall back to constant noise (original behavior)')

    args = parser.parse_args()

    file_dir = f"eps{args.num_eps}_steps{args.num_time_steps_per_eps}"
    csi_tag = f"_csi{args.channel_noise_var}" if args.channel_noise_var > 0 else ""
    file_name = (f"{args.num_antennas}_{args.num_RIS_elements}_{args.num_users}_"
                 f"{args.power_t}_{args.lr}_{args.tau}_{args.exploration_noise}_{args.utd}_{args.seed}{csi_tag}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(f"./outputs/{file_dir}"):
        os.makedirs(f"./outputs/{file_dir}")

    if args.save_model and not os.path.exists("./Models"):
        os.makedirs("./Models")

    env = environment.RIS_MISO(args.num_antennas, args.num_RIS_elements, args.num_users,
                               AWGN_var=args.awgn_var,
                               channel_noise_var=args.channel_noise_var)

    # Cascaded observation state = [action, power_t, power_r, D_real, D_imag] (paper 2 layout)
    state_dim = env.action_dim + 2 * args.num_users + \
        2 * args.num_users * args.num_RIS_elements * args.num_antennas
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": args.power_t,
        "max_action": max_action,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }
 
    # Initialize the instant rewards recording array
    instant_rewards = []
    max_reward = 0 
    channel_base = args.seed if args.channel_seed is None else args.channel_seed
    for eps in range(int(args.num_eps)):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        agent = DDPG.DDPG(**kwargs)  

        current_seed = channel_base + eps + 1
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        env = environment.RIS_MISO(args.num_antennas, args.num_RIS_elements, args.num_users,
                               AWGN_var=args.awgn_var,
                               channel_noise_var=args.channel_noise_var)
        state, done = env.reset(), False
        state = build_cascaded_state(env, state)
        episode_reward = 0
        episode_num = 0
        episode_time_steps = 0
        state = whiten(state)
        eps_rewards = []
        replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

        for t in range(int(args.num_time_steps_per_eps)):
            # Choose action from the policy
            if args.start_time_steps > t:                
                action = np.random.uniform(-1,1,size=action_dim)
            else:
                action = agent.select_action(np.array(state))
                if args.exploration_noise > 0:
                    noise_std = args.exploration_noise * max(args.noise_final_ratio, 1.0 - t / args.num_time_steps_per_eps)                    
                    action += np.random.normal(0, noise_std, size=action_dim)

            # Take the selected action
            next_state, reward, done, _ = env.step(action)
            next_state = build_cascaded_state(env, next_state)
            done = 1.0 if t == args.num_time_steps_per_eps - 1 else float(done)
            # Store data in the experience replay buffer
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward

            state = whiten(state)

            if reward > max_reward:
                max_reward = reward

            if replay_buffer.size > args.batch_size:
                # Train the agent
                for _ in range(args.utd):
                    agent.update_parameters(replay_buffer, args.batch_size)

            eps_rewards.append(reward)
            episode_time_steps += 1
            if done:
                # Reset the environment
                state, done = env.reset(), False
                state = build_cascaded_state(env, state)
                episode_reward = 0
                episode_time_steps = 0
                episode_num += 1
                state = whiten(state)
                instant_rewards.append(eps_rewards)

        rr = np.array(eps_rewards)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
              f'Benchmark {args.num_antennas}_{args.num_RIS_elements}_{args.num_users} '
              f'episode {eps+1}/{args.num_eps} (channel={current_seed}): '
              f'first-100 mean={rr[:100].mean():.2f} '
              f'last-100 mean={rr[-100:].mean():.2f} '
              f'peak={rr.max():.2f}')

    np.save(f"./outputs/{file_dir}/Benchmark_256_{file_name}", instant_rewards)

    R = np.array(instant_rewards)  # [num_eps, T]
    print(f'\n[Benchmark] summary over {args.num_eps} episodes:')
    print(f'  first-100 mean: {R[:, :100].mean():.2f} +/- {R[:, :100].mean(axis=1).std():.2f}')
    print(f'  last-100 mean: {R[:, -100:].mean():.2f} +/- {R[:, -100:].mean(axis=1).std():.2f}')
    print(f'  overall peak: {max_reward:.2f}')
    print(f'  results saved: ./outputs/{file_dir}/')

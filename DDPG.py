import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Feature(nn.Module):
    def __init__(self, state_dim):
        super(Actor_Feature, self).__init__()
        hidden_dim = 256  # fixed width, independent of state_dim
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state):
        x = torch.tanh(self.l1(state.float()))
        x = torch.tanh(self.l2(x))
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 256  # fixed width, independent of state_dim
        self.device = device
        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t
        self.max_action = max_action
        self.hidden_dim = hidden_dim

        self.l3 = nn.Linear(hidden_dim, action_dim)

    def compute_power(self, a):
        # Normalize the power (G segment layout [M*K real, M*K imag]; safe for M != K)
        G_real = a[:, :self.M * self.K].cpu().data.numpy()
        G_imag = a[:, self.M * self.K:2 * self.M * self.K].cpu().data.numpy()
        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)
        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))
        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)
        return current_power_t

    def compute_phase(self, a):
        # Normalize the phase matrix
        Phi_real = a[:, -2 * self.N:-self.N].detach()
        Phi_imag = a[:, -self.N:].detach()
        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def forward(self, x):
        a = torch.tanh(self.l3(x).float())
        # Normalize the transmission power and phase matrix
        current_power_t = self.compute_power(a.detach()).expand(-1, 2 * self.M * self.K) / np.sqrt(self.power_t)
        real_normal, imag_normal = self.compute_phase(a.detach())
        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)

        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)
        action = self.max_action * a / division_term

        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 256  # fixed width, independent of state/action dimensions
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))
        q = torch.tanh(self.l2(torch.cat([q, action], 1)))
        q = self.l3(q)
        return q

class DDPG(object):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.001):
        self.device = device
        self.lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_decay = actor_decay
        self.critic_decay = critic_decay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.M = M
        self.N = N
        self.K = K

        powert_t_W = 10 ** (power_t / 10)

        # Initialize actor networks and optimizer
        self.actor_feature = Actor_Feature(state_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=self.actor_decay)

        self.actor_feature_target = copy.deepcopy(self.actor_feature)
        self.actor_feature_optimizer = torch.optim.Adam(
            self.actor_feature.parameters(), 
            lr=self.actor_lr * 1.0 , 
            weight_decay=self.actor_decay)

        # Initialize critic networks and optimizer
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=self.critic_decay)

        # Initialize the discount and target update rated
        self.discount = discount
        self.tau = tau
        
    def select_action(self, state):
        self.actor_feature.eval()
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            state_feature = self.actor_feature(state)
            action = self.actor(state_feature).cpu().data.numpy().flatten().reshape(1, -1)
        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor_feature.train()
        self.actor.train()

        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q-value
        next_state_feature = self.actor_feature_target(next_state)
        target_Q = self.critic_target(next_state, self.actor_target(next_state_feature))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get the current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        state_feature = self.actor_feature(state)
        actor_loss = -self.critic(state, self.actor(state_feature)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        self.actor_feature_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_feature_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_feature.parameters(), self.actor_feature_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def load(self, file_name):
        map_location = self.device
        self.critic.load_state_dict(torch.load(file_name + "_critic", map_location=map_location))
        self.critic_target.load_state_dict(torch.load(file_name + "_critic_target", map_location=map_location))        
        self.actor_feature.load_state_dict(torch.load(file_name + "_actor_feature", map_location=map_location))
        self.actor_feature_target.load_state_dict(torch.load(file_name + "_actor_feature_target", map_location=map_location))        
        self.actor.load_state_dict(torch.load(file_name + "_actor", map_location=map_location))
        self.actor_target.load_state_dict(torch.load(file_name + "_actor_target", map_location=map_location))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer", map_location=map_location))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer", map_location=map_location))    
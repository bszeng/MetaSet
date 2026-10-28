import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import pickle 

class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(range(0, self.size), size=batch_size, replace=False)
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )

class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)

class Critic_Network(nn.Module):
    def __init__(self, state_dim):
        super(Critic_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim,300)
        self.fc2 = nn.Linear(300,100)
        self.fc3 = nn.Linear(100,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        x = torch.relu(self.fc2(x))
        x = nn.functional.softplus(self.fc3(x))
        return torch.mean(x)

class MultiTaskReplayBuffer(object):
    def __init__(self, tasks_num, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((tasks_num, max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((tasks_num, max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((tasks_num, max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((tasks_num, max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((tasks_num, max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, task_id, state, action, next_state, reward, done):

        self.state[task_id, self.ptr] = state
        self.action[task_id, self.ptr] = action
        self.next_state[task_id, self.ptr] = next_state
        self.reward[task_id, self.ptr] = reward
        self.not_done[task_id, self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = self.ptr

    def sample(self, task_id, batch_size):
        index = np.random.choice(range(0, self.max_size), size=batch_size, replace=False)    #restored
        return (
            torch.FloatTensor(self.state[task_id, index]).to(self.device),
            torch.FloatTensor(self.action[task_id, index]).to(self.device),
            torch.FloatTensor(self.next_state[task_id, index]).to(self.device),
            torch.FloatTensor(self.reward[task_id, index]).to(self.device),
            torch.FloatTensor(self.not_done[task_id, index]).to(self.device)
        )

    def save(self, path):
            """Save the buffer state to a file."""
            tasks_num = self.state.shape[0]
            state_dim = self.state.shape[2]
            action_dim = self.action.shape[2]
            
            data = {
                "tasks_num": tasks_num,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_size": self.max_size,
                "ptr": self.ptr,
                "size": self.size,
                "state": self.state,
                "action": self.action,
                "next_state": self.next_state,
                "reward": self.reward,
                "not_done": self.not_done,
                "device": self.device
            }
            
            with open(path, "wb") as f:
                pickle.dump(data, f)

    # @staticmethod
    def load(path):
        """Restore the buffer state from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        buffer = MultiTaskReplayBuffer(
            tasks_num=data["tasks_num"],
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
            max_size=data["max_size"]
        )
        
        buffer.ptr = data["ptr"]
        buffer.size = data["size"]
        buffer.state = data["state"]
        buffer.action = data["action"]
        buffer.next_state = data["next_state"]
        buffer.reward = data["reward"]
        buffer.not_done = data["not_done"]
        buffer.device = data["device"] 
        
        return buffer
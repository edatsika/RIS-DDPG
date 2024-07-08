import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# Implementation of the Deep Deterministic Policy Gradient algorithm (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        self.device = device

        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t  # in dB

        rho_dim = self.K
        theta_dim = 2 * self.K * self.M * self.N  # Factor of 2 for real and imaginary parts
        a_dim = self.K * self.M  # Binary association variable

        # Define layers for rho_k
        self.fc1_rho = nn.Linear(state_dim, hidden_dim)
        self.fc2_rho = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_rho = nn.Linear(hidden_dim, rho_dim)

        # Define layers for theta_kmn (real and imaginary parts separately)
        self.fc1_theta_real = nn.Linear(state_dim, hidden_dim)
        self.fc2_theta_real = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_theta_real = nn.Linear(hidden_dim, theta_dim // 2)

        self.fc1_theta_imag = nn.Linear(state_dim, hidden_dim)
        self.fc2_theta_imag = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_theta_imag = nn.Linear(hidden_dim, theta_dim // 2)

        # Define layers for a_km (mapped to continuous space)
        self.fc1_a = nn.Linear(state_dim, hidden_dim)
        self.fc2_a = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_a = nn.Linear(hidden_dim, a_dim)

        self.max_action = max_action

    def forward(self, state):
        # Forward pass for rho_k
        x_rho = F.relu(self.fc1_rho(state))
        x_rho = F.relu(self.fc2_rho(x_rho))
        rho_k = torch.sigmoid(self.fc3_rho(x_rho)) * self.max_action

        # Forward pass for theta_kmn (real part)
        x_theta_real = F.relu(self.fc1_theta_real(state))
        x_theta_real = F.relu(self.fc2_theta_real(x_theta_real))
        theta_kmn_real = torch.tanh(self.fc3_theta_real(x_theta_real))

        # Forward pass for theta_kmn (imaginary part)
        x_theta_imag = F.relu(self.fc1_theta_imag(state))
        x_theta_imag = F.relu(self.fc2_theta_imag(x_theta_imag))
        theta_kmn_imag = torch.tanh(self.fc3_theta_imag(x_theta_imag))

        # Forward pass for a_km (mapped to [-1, 1])
        x_a = F.relu(self.fc1_a(state))
        x_a = F.relu(self.fc2_a(x_a))
        a_km_continuous = torch.tanh(self.fc3_a(x_a))  # Outputs in range [-1, 1]

        # Concatenate rho_k, theta_kmn_real, theta_kmn_imag, and a_km_continuous
        action = torch.cat([rho_k, theta_kmn_real, theta_kmn_imag, a_km_continuous], dim=-1)
        
        return action
    



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, 1)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))

        q = self.l2(torch.cat([q, action], 1))

        #Pass through the output layer (no activation function)
        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, bandwidth, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.001):
        self.device = device

        powert_t_W = 10 ** (power_t / 10)

        #self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
        #self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        #self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay)

                # Initialize the discount and target update rated
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        #state = torch.FloatTensor(state).to(self.device)
        #return self.actor(state).cpu().data.numpy()
        self.actor.eval()

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)

        #print("ACTION in select action:", action)
        #print("ACTION SHAPE in select action:", action.shape)
        return action

    def soft_update(self):

        # Monitor gradient norms for actor and critic ---edatsika
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                gradient_norm = param.grad.norm().item()
               #print(f"Actor Layer: {name}, Gradient Norm: {gradient_norm}")

        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                gradient_norm = param.grad.norm().item()
                #print(f"Critic Layer: {name}, Gradient Norm: {gradient_norm}")

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()

        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Compute the target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
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
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update the target networks
        self.soft_update()

 # Save the model parameters
    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

    # Load the model parameters
    def load(self, file_name):
        self.critic.load_state_dict(torch.load(file_name + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(file_name + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        
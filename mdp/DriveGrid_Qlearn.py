"""Implements the Agent"""
import numpy as np
import gymnasium as gym
import torch 
from collections import defaultdict
from gymnasium.spaces import Discrete
import random


    

class DriveQlearning:
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95) -> None:
        self.env = environment
        self.Q= {}
        
        # Initialise the Q
        for i in range(self.env.unwrapped.observation_space[0].n):                        
            for j in [-1,0,1]: 
                for k in [-1,0,1]: 
                    self.Q.update({(i,(j,k)) : 0})                        # initialise q_value table
        
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        
    def policy(self, state: int):
        if np.random.random() < self.epsilon:
            options = [(i,j) for i in [-1,0,1] for j in [-1,0,1]]
            return random.choice(options)
        else:
            filtered_dict = {k: v for k, v in self.Q.items() if k[0] == state}
            max_key = max(filtered_dict, key=filtered_dict.get)
            return max_key[1]


    
    
    def update(self, state, action, reward, terminated: bool, s_prime):
        Q_hat = (not terminated) * max(value for key, value in self.Q.items() if key[0] == s_prime)
        TD = (reward + self.gamma * Q_hat - self.Q[(state,action)])
        self.Q[(state,action)] += self.alpha * TD     # Update Q
        self.training_error.append(TD)
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)

        
        
        
        
class DriveQlearningVFA:
    def __init__(self, environment: gym.Env, learning_rate: float, epsilon: float, 
                epsilon_decay: float,
                final_epsilon: float, 
                gamma: float = .95, initial_w: np.ndarray = None) -> None:
        self.env = environment
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = epsilon_decay
        self.e_final = final_epsilon
        self.training_error = []
        if initial_w is None:
            i_w = np.ones((9,9))
        else:
            i_w = initial_w
        self.w = torch.tensor(i_w, dtype=float, requires_grad=True)
    
    def x(self, state: int) -> np.ndarray:
        return torch.tensor(self.env.unwrapped.available_actions_value[state],dtype=float, requires_grad=False)
    
    def q(self, state: int, action) -> float:
        return self.x(state) @ self.w[:, action]
        
    def policy(self, state: int) -> int:
        available = self.env.unwrapped.available_actions[state]
        options = [i for i, a in enumerate(available) if a != 0]
        if np.random.random() < self.epsilon:
            return available[np.random.choice(options)]
        else:
            available_values = [self.q(state, a).detach().numpy() for a in options]
            return available[options[np.argmax(available_values)]]

    
    def update(self, state: int, action: int, reward: float, s_prime: int):
        available = self.env.unwrapped.available_actions[s_prime]
        options = [i for i, a in enumerate(available) if a != 0]     
        q_target = reward + self.gamma * max(self.q(s_prime, a) for a in options)
        action_index = self.env.unwrapped.coordinates.index(action)
        q_value = self.q(state, action_index)
        delta = q_target - q_value
        q_value.backward()
        with torch.no_grad():
            self.w += self.alpha * delta * self.w.grad 
            self.w.grad.zero_()
        self.training_error.append(delta.detach().numpy())
    
    def decay_epsilon(self):
        self.epsilon = max(self.e_final, self.epsilon - self.e_decay)
        
        
       

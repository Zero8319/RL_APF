"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import environment
import time

# Hyper Parameters
time_start = time.time()
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = environment.environment()
N_ACTIONS = env.num_action
N_STATES = env.num_state


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state.view(1, -1)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net(state)
            action = torch.argmax(actions_value).numpy()
        else:  # random
            action = env.rng.integers(0, N_ACTIONS)
        return action

    def store_transition(self, state, action, reward, state_next):
        transition = np.hstack((np.ravel(state), action, np.array(reward), np.ravel(state_next)))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.ravel(env.rng.choice(range(MEMORY_CAPACITY), size=(1, BATCH_SIZE)))
        b_memory = self.memory[sample_index, :]
        b_state = torch.tensor(b_memory[:, :N_STATES], dtype=torch.float32)
        b_action = torch.tensor(b_memory[:, N_STATES:N_STATES + 1], dtype=torch.int64)
        b_reward = torch.tensor(b_memory[:, N_STATES + 1:N_STATES + 2], dtype=torch.float32)
        b_state_next = torch.tensor(b_memory[:, -N_STATES:], dtype=torch.float32)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_state).gather(1, b_action)  # shape (batch, 1)
        q_next = self.target_net(b_state_next).detach()  # detach from graph, don't backpropagate
        q_target = b_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(6000000):
    state = env.reset()
    episode_return = 0
    while True:
        # env.render()
        action = dqn.choose_action(state)

        # take action
        state_next, reward, done = env.step(action)

        dqn.store_transition(state, action, reward, state_next)

        episode_return += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('episode: ', i_episode,
                      'episode_return: ', round(episode_return, 2))

        if done:
            break
        state = state_next
net = dqn.eval_net
torch.save(net.state_dict(), 'state_dict_carpole1.pt')
time_end = time.time()
time_cost = time_end - time_start
np.savetxt('1.txt', np.array(time_cost, ndmin=1))
os.system('shutdown -s -t 60')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matlab.engine
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import datetime
import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import time
import os.path

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = self.flatten(x)
        # print(x.size())
        # print('forward')
        x = F.relu(self.fc1(x))
        # print('forward1')
        x = self.fc2(x)
        # print('forward3')
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(OSNR, M, policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []

    # Connect to matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/reedvl/Desktop/Nokia/Simulation/BlackBox/QAMBlackBox_Daria', nargout=0)
    
     # Create logs with current timestamp
    current_date_and_time = datetime.datetime.now()
    current_date_and_time_string = str(current_date_and_time)
    file_name =  current_date_and_time_string + '.txt'
    
    save_path = '/home/reedvl/Desktop'
    completeName = os.path.join(save_path, file_name) 
    f = open(completeName, "a")
    
    rs = []
    for i_episode in range(1, n_episodes+1):
        tic = time.perf_counter()
        saved_log_probs = []
        rewards = []
        state, init_mi = eng.init(M, OSNR, nargout=2)
        print('i got the init')
        mis = [0]*max_t
        actions = [0]*max_t
        best_mi = 0
        broken_loop = False

# increase steps, 4*M, 3*M for hidden
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, mi, done = eng.mi(action, state, OSNR, init_mi, nargout = 4)
            mis[t] = mi
            actions[t] = action
            rewards.append(reward)
            
            print('action', action, 't', t, 'ep:', i_episode, 'mi', mi, 'reward', reward, 'cum', sum(rewards))
            if mi > best_mi:
                best_mi = mi
                
            # if actions[t] == actions[t-1] and t > 5:
            #     if actions[t] == actions[t-2]:
            #         if actions[t] == actions[t-3]:
            #             break
            # Plot the new constellation
            # np_state = np.array(state)
            # x = np.zeros(len(np_state))
            # y = np.zeros(len(np_state))
            # for i in range(len(np_state)):
            #     x[i] = np_state[i, 0]
            #     y[i] = np_state[i, 1]
            
            # plt.plot(x, y, 'ro')
            # plt.show()
            
            # print('mi', mi)
            if done:
                broken_loop = True
                break
        
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        probs = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
            probs.append(float(log_prob))
        policy_loss = torch.cat(policy_loss).sum()
        
        rs.append(float(policy_loss))
        
        # Save all the data
        f.write('state: ')
        f.write(str(state))
        f.write('\n')
        
        f.write('rewards: ')
        f.write(str(rewards))
        f.write('\n')
        
        f.write('scores: ')
        f.write(str(scores))
        f.write('\n')
        
        f.write('probs: ')
        f.write(str(probs))
        f.write('\n')
        
        f.write('actions: ')
        f.write(str(actions))
        f.write('\n')
        
        f.write('policy loss: ')
        f.write(str(rs))
        f.write('\n')
        
        f.write('best_mi: ')
        f.write(str(best_mi))
        f.write('\n')
        
        if broken_loop:
            f.write('BROKE THE LOOP')
            f.write('\n')
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        episodes = [i for i in range(i_episode)]
        plt.plot(episodes, rs, '.-')
        plt.xlabel('Episode')
        plt.ylabel('Policy loss')
        plt.show()
        
        window_width = 10
        cumsum_vec = np.cumsum(np.insert(scores, 0, 0)) 
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

        plt.plot(episodes, scores, '.-')
        plt.plot(np.arange(1, len(ma_vec)+1), ma_vec, 'r')
        plt.xlabel('Episode')
        plt.ylabel('Scores')
        plt.show()
        
        plt.hist(actions)
        plt.xticks(actions)
        plt.title('Actions')
        plt.show()
        
        plt.hist(probs)
        plt.xticks(probs)
        plt.title('Log probs')
        plt.show()
        
        # Plot final constellation
        np_state = np.array(state)
        x = np_state[:,0]
        y = np_state[:,1]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.scatter(x, y)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
        toc = time.perf_counter()
        print(f"RL performed in {toc - tic:0.4f} seconds")
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
    f.close()
    print(scores)
    
    fig_name = current_date_and_time_string + 'loss' + '.png'
    
    completeNameFig = os.path.join(save_path, fig_name) 
    plt.plot(np.arange(1, len(rs)+1), rs)
    plt.xlabel('Episode')
    plt.ylabel('Policy loss')
    plt.savefig(completeNameFig)
    plt.show()
    
    fig_name2 = current_date_and_time_string + 'scores' + '.png'
    completeNameFig2 = os.path.join(save_path, fig_name2) 
    
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.xlabel('Episode')
    plt.ylabel('Scores')
    plt.savefig(completeNameFig2)
    plt.show()
    
    model_name = current_date_and_time_string + '_model' + '.pth'
    completeModelName = os.path.join(save_path, model_name) 
    
    torch.save({
        'episode': n_episodes,
        'max_t': max_t,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, completeModelName)
    
    return scores
    
OSNR = 6
M = 16
n_episodes=400
max_t=150
gamma=1.0
print_every=1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = Policy(s_size=2*M,h_size=3*M,a_size=4*M).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
# load the model checkpoint
# checkpoint = torch.load('/home/reedvl/Desktop/2021-08-29 10:31:29.819551_model.pth')
# # load model weights state_dict
# policy.load_state_dict(checkpoint['model_state_dict'])
# print('Previously trained model weights state_dict loaded...')
# # load trained optimizer state_dict
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# print('Previously trained optimizer state_dict loaded...')
# episodes = checkpoint['episode']

# print(f"Previously trained for {episodes} number of epochs...")
# # train for more epochs
# n_episodes = 1
# print(f"Train for {n_episodes} more epochs...")

scores = reinforce(OSNR, M, policy, optimizer, n_episodes, max_t, gamma, print_every)
# print(scores)
   # Connect to matlab
# eng = matlab.engine.start_matlab()
# eng.cd(r'/home/reedvl/Desktop/Nokia/Simulation/BlackBox/QAMBlackBox_Daria', nargout=0)
    
# state, init_mi = eng.init(16, 6, nargout=2)
# print(init_mi)
# M = 16
# policy = Policy(s_size=2*M,h_size=M,a_size=4*M)
# action, log_prob = policy.act(state)
# print(action, log_prob)
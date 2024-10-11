from torch.utils.data import Dataset
import os
from random import shuffle
import numpy as np
import torch


class Dataset(Dataset):
    def __init__(self, states, goals, starts , envs, lazy=True, device=None):
        super(Dataset, self).__init__()
        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        #self.states = torch.tensor(states, dtype=torch.float32, device=self._device)#[n,i,2]
        self.states = states
        #self.goals = goals
        #self.starts = starts
        self.goals = torch.tensor(goals, dtype=torch.float32, device=self._device)#[n,2]
        self.starts = torch.tensor(starts, dtype=torch.float32, device=self._device)#[n,2]
        self.envs = torch.tensor(envs, dtype=torch.float32, device=self._device)#[n,20,20]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        #TODO 坐标是否对应像素坐标

        env  = self.envs[idx]
        h,w = env.shape
        
        state = self.states[idx]
        # 0-19
        state_tmp = np.zeros((h,w))
        for p in state:
            state_tmp[p[0],p[1]] = 1
        state = torch.tensor(state_tmp, dtype=torch.float32, device=self._device)

        goal = self.goals[idx]
        

        start = self.starts[idx]
        
        env[int(goal[0]),int(goal[1])] =1
        env[int(start[0]),int(start[1])] =1

        return env.unsqueeze(0) , goal , start, state 











        

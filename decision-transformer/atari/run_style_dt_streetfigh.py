import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_style_street import GPT, GPTConfig
from mingpt.trainer_style_street import StyleTrainer, StyleTrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset,create_dataset_rwds
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=50)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--game', type=str, default='StreetFigh')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--mask', action='store_true', default=False)
parser.add_argument('--max_size', type=int, default=7734)
args = parser.parse_args()

set_seed(args.seed)

class StyleStateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, rewards):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = np.reshape(data,(-1,32))
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.rewards = rewards
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for index, i in enumerate(self.done_idxs):
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                context_return = np.mean(self.rewards[idx:done_idx])
                break
        idx = done_idx - block_size
        states = torch.tensor(self.data[idx:done_idx], dtype=torch.float32).reshape(block_size, -1) # (block_size, 32)
        #states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        # 0,0.03,0.1,0.2
        if context_return >= 1:
            style = torch.tensor([0], dtype=torch.int64).unsqueeze(1)
        elif context_return >= 0:
            style = torch.tensor([1], dtype=torch.int64).unsqueeze(1)
        elif context_return >= -1:
            style = torch.tensor([2], dtype=torch.int64).unsqueeze(1)
        elif context_return >= -2:
            style = torch.tensor([3], dtype=torch.int64).unsqueeze(1)
        else:
            style = torch.tensor([4], dtype=torch.int64).unsqueeze(1)
        return states, actions, rtgs, timesteps, style

npzfile = np.load('/home/AIVO-StreetFigherReinforcementLearning/'+"All data; trajs: 152; trans: 157637.npz")
obss = npzfile["obss"]
actions = npzfile["actions"]
returns = npzfile["returns"]
rtgs = npzfile["rtg"]
timesteps = npzfile["timesteps"]
rewards = npzfile["rewards"]
done_idxs = npzfile["done_idxs"]


logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

#train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
train_dataset = StyleStateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps, rewards)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps), style_num=5, mask=args.mask)
model = GPT(mconf)
print(train_dataset.vocab_size, train_dataset.block_size, args.model_type, max(timesteps))
# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = StyleTrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=0, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps), style_num=5, num_trajectories=152, mask=args.mask)
trainer = StyleTrainer(model, train_dataset, None, tconf)

trainer.train()

import csv
import os
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig, Args, Env
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--pretrained_epochs', type=str, default="1 5 10 15 20")
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--render', action='store_true')
parser.add_argument("--eval_times", type=int, default=1)
parser.add_argument("--rtg", type=int, default=90)
parser.add_argument("--difficulty", type=int, default=0)
parser.add_argument("--renderrate", type=int, default=1)
args = parser.parse_args()

set_seed(args.seed)


def get_returns(model, args, ret, epoch):
        model.train(False)
        # model.to(args_.device)
        args_ = Args(args.game.lower(), args.seed, args.difficulty, args.renderrate, epoch, ret)
        model.to(args_.device)
        env = Env(args_)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(args.eval_times):
            state = env.reset()
            state = state.type(torch.float32).to(args_.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(args_.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(args_.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                if args.render:
                    #env.update(goal=ret, rtg=rtgs[-1] - reward)
                    env.update(goal=100, rtg=100)
                    env.render()
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(args_.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(args_.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(args_.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, 1788) * torch.ones((1, 1, 1), dtype=torch.int64).to(args_.device)))
        env.close()
        eval_return = sum(T_rewards)/(args.eval_times)*1.0
        print("Pretrained epoches: %d, target return: %d, eval return: %d" % (epoch, ret, eval_return))
        return eval_return


def test_model(model, model_name, args, epoch):
    model_path = "/home/yao_yao/decision-transformer/atari/ckpt_dir/" + model_name
    if not os.path.exists(model_path):
        raise NotImplementedError()
    else:
        model.load_state_dict(torch.load(model_path), strict=False)
    if args.model_type == 'naive':
        eval_return = get_returns(model=model, args=args, ret=0)
    elif args.model_type == 'reward_conditioned':
        if args.game == 'Breakout':
            eval_return = get_returns(model=model, args=args, ret=90, epoch=int(epoch))
        elif args.game == 'Seaquest':
            eval_return = get_returns(model=model, args=args, ret=1150, epoch=int(epoch))
        elif args.game == 'Qbert':
            eval_return = get_returns(model=model, args=args, ret=14000, epoch=int(epoch))
        elif args.game == 'Pong':
            eval_return = get_returns(model=model, args=args, ret=20, epoch=int(epoch))
        elif args.game == 'Boxing':
            eval_return = get_returns(model=model, args=args, ret=args.rtg, epoch=int(epoch))
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# for Boxing
mconf = GPTConfig(18, 90,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=1788)
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.pretrained_epochs.strip().split()
for test_epoch in epochs:
    model_name = args.game+'-'+args.model_type+"-SEED-"+str(args.seed) +'-EPOCH-' + str(test_epoch) +".pth"
    test_model(model, model_name, args, test_epoch)

# Evaluates a saved NEAT genome (best.pkl) against BaselinePolicy over
# multiple episodes, since a single episode's score is a noisy fitness
# signal (see neat/EXPERIMENT_LOG.md).

import argparse
import os
import pickle

import gym
import neat
import numpy as np

import slimevolleygym
from slimevolleygym import BaselinePolicy, multiagent_rollout as rollout
from slimevolleygym.neat_policy import NeatPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--genome", default=os.path.join(os.path.dirname(__file__), "neat", "best.pkl"))
parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "neat_config.txt"))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=612)
args = parser.parse_args()

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    args.config,
)

with open(args.genome, "rb") as f:
  genome = pickle.load(f)

policy = NeatPolicy(genome, config)
baseline = BaselinePolicy()

env = gym.make("SlimeVolley-v0")
env.seed(args.seed)

scores = []
for i in range(args.episodes):
  # baseline plays as policy_right; score is from policy_right's
  # perspective, so negate it to get the candidate's score.
  score, _ = rollout(env, baseline, policy)
  scores.append(-score)

scores = np.array(scores)
wins = np.sum(scores > 0)
draws = np.sum(scores == 0)
losses = np.sum(scores < 0)

print(f"episodes: {args.episodes}")
print(f"mean score: {scores.mean():.3f} (std {scores.std():.3f})")
print(f"wins/draws/losses: {wins}/{draws}/{losses}")
print(f"score distribution: {np.unique(scores, return_counts=True)}")

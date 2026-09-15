"""
Thin wrapper that turns a neat-python genome into a policy with the
predict(obs) -> action interface used throughout this repo (see
slimevolleygym.mlp.Model, and the policies in eval_agents.py).

The genome's phenotype is a feedforward network with 12 inputs (the
SlimeVolley-v0 state observation) and 3 sigmoid outputs, which are
thresholded at 0.5 to produce the MultiBinary(3) action expected by
the environment.
"""

import numpy as np
import neat


class NeatPolicy:
  def __init__(self, genome, config):
    self.net = neat.nn.FeedForwardNetwork.create(genome, config)

  def predict(self, obs):
    output = self.net.activate(obs)
    forward = 1 if output[0] > 0.5 else 0
    backward = 1 if output[1] > 0.5 else 0
    jump = 1 if output[2] > 0.5 else 0
    if forward and backward:
      # Agent.setAction (slimevolley.py) treats forward=1 AND backward=1
      # as "stand still" -- it silently cancels any intended movement.
      # The 3 outputs are thresholded independently, so nothing stops a
      # genome from firing both; break the tie by keeping whichever raw
      # signal is stronger instead of letting them cancel out (see
      # training_scripts/neat/EXPERIMENT_LOG.md, run 9 postmortem).
      if output[0] >= output[1]:
        backward = 0
      else:
        forward = 0
    return np.array([forward, backward, jump])

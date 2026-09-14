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
    return np.array([1 if o > 0.5 else 0 for o in output])

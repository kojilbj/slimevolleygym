# Trains an agent using NEAT (neat-python), evolving both the topology and
# weights of a small feedforward network.
#
# Run 4: self-play curriculum. Runs 1-3 (see neat/EXPERIMENT_LOG.md)
# found that training directly against full-strength BaselinePolicy (with
# or without reward shaping) gives an almost flat fitness landscape --
# nearly every genome loses by the same margin regardless of skill, so
# evolution has nothing to climb. Here each genome instead plays against
# a "champion" opponent that starts as random actions and only gets
# replaced once a generation's best genome clearly beats it (mirrors
# train_ppo_selfplay.py's dethroning logic), so difficulty ramps up
# gradually instead of being maximal from generation 0.

import os
import pickle
import shutil

import gym
import neat

import slimevolleygym
from slimevolleygym import BaselinePolicy, multiagent_rollout as rollout
from slimevolleygym.neat_policy import NeatPolicy

# Settings
random_seed = 612
n_generations = 1000
save_freq = 10
n_rollouts = 3             # per-genome fitness rollouts vs current champion
n_challenge_rollouts = 20  # rollouts used to decide whether to dethrone
dethrone_margin = 0.5      # challenger's mean score must exceed this

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config_selfplay.txt")

logdir = "neat_run4"
if not os.path.exists(logdir):
  os.makedirs(logdir)

env = gym.make("SlimeVolley-v0")
env.seed(random_seed)

# only used for the final external benchmark, not during self-play training
baseline_policy = BaselinePolicy()


class RandomPolicy:
  def __init__(self):
    self.action_space = gym.spaces.MultiBinary(3)

  def predict(self, obs):
    return self.action_space.sample()


class SelfPlayOpponent:
  """ Holds the current self-play champion; starts as random actions. """

  def __init__(self):
    self.policy = RandomPolicy()
    self.generation = 0

  def predict(self, obs):
    return self.policy.predict(obs)

  def maybe_dethrone(self, challenger_genome, config):
    challenger_policy = NeatPolicy(challenger_genome, config)
    scores = []
    for _ in range(n_challenge_rollouts):
      # self (the current champion) plays as policy_right; negate for
      # the challenger's score.
      score, _ = rollout(env, self, challenger_policy)
      scores.append(-score)
    mean_score = sum(scores) / len(scores)
    if mean_score > dethrone_margin:
      self.policy = challenger_policy
      self.generation += 1
      model_filename = os.path.join(logdir, "champion_" + str(self.generation).zfill(4) + ".pkl")
      with open(model_filename, "wb") as out:
        pickle.dump(challenger_genome, out)
      print(f"SELFPLAY: new champion (gen {self.generation}), "
            f"mean score vs previous champion: {mean_score:.2f}")
      return True
    return False


opponent = SelfPlayOpponent()


def eval_genomes(genomes, config):
  best_genome = None
  for genome_id, genome in genomes:
    candidate_policy = NeatPolicy(genome, config)
    scores = []
    for _ in range(n_rollouts):
      # opponent (current champion) plays as policy_right; negate for
      # the candidate's score.
      score, _ = rollout(env, opponent, candidate_policy)
      scores.append(-score)
    genome.fitness = sum(scores) / len(scores)
    if best_genome is None or genome.fitness > best_genome.fitness:
      best_genome = genome

  opponent.maybe_dethrone(best_genome, config)


def run():
  config = neat.Config(
      neat.DefaultGenome,
      neat.DefaultReproduction,
      neat.DefaultSpeciesSet,
      neat.DefaultStagnation,
      config_path,
  )

  population = neat.Population(config)
  population.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  population.add_reporter(stats)
  population.add_reporter(neat.Checkpointer(save_freq, filename_prefix=os.path.join(logdir, "checkpoint-")))

  winner = population.run(eval_genomes, n_generations)

  # save the overall best genome, plus a copy of the config it was
  # produced with, so eval_agents.py can load it without depending on
  # this training script.
  with open(os.path.join(logdir, "best.pkl"), "wb") as out:
    pickle.dump(winner, out)
  shutil.copy(config_path, os.path.join(logdir, "neat_config.txt"))

  print("best fitness (vs final self-play champion):", winner.fitness)
  print("self-play dethronings:", opponent.generation)


if __name__ == "__main__":
  run()
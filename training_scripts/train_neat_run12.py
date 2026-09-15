# Trains an agent using NEAT (neat-python), evolving both the topology and
# weights of a small feedforward network.
#
# Run 12: mixed opponent pool. Run 11 (pure self-play, num_hidden=10,
# complexity_penalty, multi-opponent dethroning, protected bx/x repair,
# forward/backward tie-break) finally produced genuinely competent
# rallying (champion_0035: 16-72 ball touches depending on opponent,
# near-parity with BaselinePolicy at -1). Copied from train_neat.py at
# that point (see neat/EXPERIMENT_LOG.md) to try mixing this repo's own
# pretrained zoo models (zoo/ga_sp, zoo/cmaes, zoo/cmaes_sp) plus
# BaselinePolicy itself into the opponent archive as permanent,
# never-evicted sparring partners alongside the rotating self-play
# lineage -- feasible now that the population is skilled enough for
# these stronger opponents to give a non-flat fitness gradient (see
# "why #3 was set aside" in the run 4/5 section: mixing in BaselinePolicy
# was judged premature back then because every genome lost to it
# identically; that is no longer true). External models have no genome
# and can never be "crowned" -- they only enrich OpponentArchive.sample()
# and the dethroning check's opponent pool; only NEAT genomes (evolved by
# this population) can become a new champion_NNNN.pkl.
#
# Original run 5 self-play curriculum, v2. Run 4's self-play (see
# neat/EXPERIMENT_LOG.md) produced 172 clean "dethronings" but the final
# champion turned out to have NO functional path from its 12 observation
# inputs to its outputs -- it just emitted a fixed action regardless of
# input. Diagnosis: starting self-play from a random-action opponent
# never rewards reacting to the ball at all (any fixed action is just as
# good against another blind opponent), and NEAT's structural mutations
# (conn_delete_prob/node_delete_prob) can prune the input-facing half of
# a network without hurting fitness if nothing needs those inputs yet.
# Three fixes, all applied together:
#   1. Seed self-play with a simple scripted TrackingPolicy (moves/jumps
#      toward the ball) instead of RandomPolicy, so reacting to ball
#      position matters from generation 0.
#   2. neat_config_selfplay_v2.txt lowers conn_delete_prob/node_delete_prob
#      so sensory pathways are less likely to be pruned away before
#      they're useful.
#   3. OpponentArchive keeps a rotating history of past champions (plus
#      the original TrackingPolicy seed) instead of only the single most
#      recent one, so genomes are evaluated against a diversity of
#      opponents rather than overfitting to one narrow lineage.
#
# v5 update: pushing conn_delete_prob/node_delete_prob too low (v4) fixed
# the pruned-pathway problem above but caused a different failure --
# unconstrained node bloat (158 nodes, mostly orphaned, zero working
# input-to-output paths after 5000 generations). v5 splits the
# difference on deletion rates, slows node creation, and adds an
# explicit per-node fitness penalty (`complexity_penalty`) plus a
# connectivity_report() health check printed on every dethroning, so
# bloat/dead-pathway regressions show up in the log instead of requiring
# manual inspection (see neat/EXPERIMENT_LOG.md for the full history).
#
# v6/v7 update: v5 plateaued (2-5/12 working inputs for 650+ generations,
# no bloat but no growth either). Comparison against this repo's own
# pretrained self-play zoo model (zoo/ga_sp/ga.json, a fixed-topology
# 12->10->10->3 MLP) showed self-play itself works fine here given
# enough capacity -- it beats BaselinePolicy. So v6 starts NEAT with
# num_hidden=10 (a deliberate departure from NEAT's usual "start
# minimal" practice) instead of 0, with compatibility_threshold raised
# to 6.0 (a bigger initial network means bigger genetic distances, and
# the old 3.0 threshold fractured generation 0 into ~1 species per
# genome). That kept 12/12 inputs working for the first ~13
# dethronings, but still collapsed back to 0/12 by generation 100 --
# num_hidden alone wasn't enough. Root cause: `maybe_add_champion` only
# required beating the single most recent champion, so a genome could
# join the archive by exploiting one specific opponent's blind spot
# without being generally competent (the same failure mode GA self-play
# avoids by pitting random pairs from a whole population against each
# other continuously, not gating entry into a single lineage). v7 fixes
# this directly: `maybe_add_champion` now tests the challenger against
# several opponents sampled from across the archive (not only the
# latest) and requires beating that broader sample on average, at the
# same total rollout budget.
#
# v8 update: v7's broader dethroning check worked (no more narrow
# single-opponent exploits -- score margins stayed healthy throughout),
# but connectivity still collapsed around generation ~20 and stayed at
# 0-2/12 through generation 68: node count climbed to 23-31 while
# enabled connections fell to 6-20. Cause: v6 removed
# `complexity_penalty` on the assumption that num_hidden=10 baseline
# capacity wasn't "bloat", but nothing was reintroduced to stop further
# *unbounded* node growth on top of that baseline -- the same failure
# mode as the original v4 run, just starting from a higher floor. v8
# combines all three fixes at once instead of one at a time:
# num_hidden=10 (v6) + broader archive-sampled dethroning (v7) +
# complexity_penalty (v5) reinstated.
#
# v9 update: v8's 5000-gen attempt avoided bloat (nodes settled to a
# tiny 3-7) but the parsimony penalty pruned away BOTH bx and the
# agent's own x entirely by generation ~190 -- the network could no
# longer sense horizontal position at all (user, watching it play:
# "ずっと真ん中の壁に張り付いてる"). Rather than keep re-tuning
# delete/penalty rates and hoping these specific connections survive by
# luck, `repair_protected_inputs()` now directly guarantees bx and x
# always have at least one enabled path to an output, applied fresh
# every generation in eval_genomes.

import copy
import os
import pickle
import random
import shutil

import gym
import neat
import numpy as np

import slimevolleygym
from slimevolleygym import BaselinePolicy, multiagent_rollout as rollout
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite
from slimevolleygym.neat_policy import NeatPolicy

# Settings
random_seed = 612
n_generations = 100  # quick trial run to sanity-check the mixed opponent pool before scaling up
save_freq = 10
n_rollouts = 3             # per-genome fitness rollouts, vs opponents sampled from the archive
n_challenge_rollouts = 20  # rollouts used to decide whether to dethrone
dethrone_margin = 0.5      # challenger's mean score (vs current strongest) must exceed this
archive_max_size = 10      # keep the seed + up to this many past champions
complexity_penalty = 0.01  # subtracted per genome node, to directly select against bloat

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config_selfplay_v6.txt")

logdir = "neat_run12"
if not os.path.exists(logdir):
  os.makedirs(logdir)

env = gym.make("SlimeVolley-v0")
env.seed(random_seed)

# only used for the final external benchmark, not during self-play training
baseline_policy = BaselinePolicy()


class TrackingPolicy:
  """ Simple scripted policy: move toward the ball's x position and jump
  when it's close and low. Used as the self-play archive's seed so that
  reacting to the ball matters from generation 0 (see run 4 postmortem
  above; a pure-random seed never rewards this at all). """

  def predict(self, obs):
    x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy = obs
    forward = 1 if bx > x + 0.05 else 0
    backward = 1 if bx < x - 0.05 else 0
    jump = 1 if (by < 0.5 and abs(bx - x) < 0.5) else 0
    return np.array([forward, backward, jump])


def has_path_to_node(start, target, connections, visited=None):
  """ Whether an enabled path exists from node `start` to `target`. """
  if visited is None:
    visited = set()
  if start == target:
    return True
  if start in visited:
    return False
  visited.add(start)
  for (src, dst), conn in connections.items():
    if src == start and conn.enabled and has_path_to_node(dst, target, connections, visited):
      return True
  return False


def has_path_to_output(start, connections):
  """ Whether an enabled path exists from node `start` to any of the 3
  output nodes (keys 0, 1, 2). Used for the connectivity_report() bloat/
  dead-pathway check below -- run 5's v4 attempt bloated to 158 nodes
  with zero working input-to-output paths, and would have been caught in
  minutes by a check like this instead of requiring manual inspection. """
  return any(has_path_to_node(start, target, connections) for target in (0, 1, 2))


def connectivity_report(genome):
  """ Returns (node count, enabled connection count, number of the 12
  inputs that have at least one enabled path all the way to an output). """
  n_nodes = len(genome.nodes)
  n_enabled = sum(1 for c in genome.connections.values() if c.enabled)
  working_inputs = sum(
      1 for k in range(-1, -13, -1)
      if any(c.enabled and has_path_to_output(dst, genome.connections)
             for (src, dst), c in genome.connections.items() if src == k)
  )
  return n_nodes, n_enabled, working_inputs


def load_external_opponents():
  """ This repo's own pretrained zoo models, evaluated vs BaselinePolicy
  before including them (see neat/EXPERIMENT_LOG.md, run 12):
    cmaes_sp:  mean -0.15 (slightly weaker than BaselinePolicy)
    ga_sp:     mean +0.23 (beats it)
    cmaes:     mean +1.00 (beats it solidly)
  Together with BaselinePolicy itself, this gives a rough difficulty
  ladder alongside the self-play lineage. """
  return [
      makeSlimePolicy(os.path.join(local_dir, "..", "zoo", "cmaes_sp", "slimevolley.cma.16.384.best.json")),
      makeSlimePolicyLite(os.path.join(local_dir, "..", "zoo", "ga_sp", "ga.json")),
      makeSlimePolicy(os.path.join(local_dir, "..", "zoo", "cmaes", "slimevolley.cma.64.96.best.json")),
      BaselinePolicy(),
  ]


class OpponentArchive:
  """ Holds two pools of opponents: `permanent` (the TrackingPolicy seed
  plus fixed external models -- these never change and can never be
  "crowned", since they have no genome) and `rotating` (up to
  `archive_max_size` past self-play champions). Genomes are evaluated
  against opponents sampled from across both pools, not just the single
  most recent champion, to reduce overfitting to one narrow lineage. """

  def __init__(self, permanent_opponents):
    self.permanent = list(permanent_opponents)
    self.rotating = []
    self.generation = 0

  def pool(self):
    return self.permanent + self.rotating

  def sample(self):
    return random.choice(self.pool())

  def maybe_add_champion(self, challenger_genome, config):
    challenger_policy = NeatPolicy(challenger_genome, config)
    # Test against several opponents sampled from across the whole pool,
    # not just the single most recent champion -- run 6 found that
    # gating on "beats only the latest champion" let genomes collapse to
    # a narrow fixed action that happened to exploit that one opponent,
    # rather than requiring genuinely broad competence (see
    # neat/EXPERIMENT_LOG.md, run 6 postmortem).
    pool = self.pool()
    n_test_opponents = min(4, len(pool))
    test_opponents = random.sample(pool, n_test_opponents)
    rollouts_each = max(1, n_challenge_rollouts // n_test_opponents)
    scores = []
    for opponent in test_opponents:
      for _ in range(rollouts_each):
        # opponent plays as policy_right; negate for the challenger's score.
        score, _ = rollout(env, opponent, challenger_policy)
        scores.append(-score)
    mean_score = sum(scores) / len(scores)
    if mean_score > dethrone_margin:
      self.rotating.append(challenger_policy)
      if len(self.rotating) > archive_max_size:
        del self.rotating[0]  # drop the oldest champion; permanent opponents are never evicted
      self.generation += 1
      model_filename = os.path.join(logdir, "champion_" + str(self.generation).zfill(4) + ".pkl")
      with open(model_filename, "wb") as out:
        pickle.dump(challenger_genome, out)
      n_nodes, n_enabled, working_inputs = connectivity_report(challenger_genome)
      print(f"SELFPLAY: new champion (gen {self.generation}), "
            f"mean score vs {n_test_opponents} sampled pool opponents: {mean_score:.2f}, "
            f"rotating archive size: {len(self.rotating)} (+{len(self.permanent)} permanent), "
            f"connectivity: {n_nodes} nodes / {n_enabled} enabled conns / "
            f"{working_inputs}/12 inputs reach an output")
      return True
    return False


archive = OpponentArchive([TrackingPolicy()] + load_external_opponents())

# Inputs that must always keep at least one enabled path to an output.
# Added after run 8's champion lost BOTH of these entirely (parsimony
# pressure pruned them away along with genuine bloat) -- the agent could
# no longer sense the ball's or its own horizontal position at all, i.e.
# exactly the "doesn't react to the ball" symptom the user first flagged
# back in run 4. Rather than continue tuning delete/penalty rates and
# hoping they survive, this repairs the genome directly every generation
# (see neat/EXPERIMENT_LOG.md, run 8 postmortem).
#
# run 9 update: guaranteeing "reaches *some* output" wasn't enough --
# the resulting champion could move toward the net but never back (its
# `backward` output was never wired to bx/x), so it walked straight into
# the net and got stuck there (85% of steps spent within 1 unit of the
# net over a 600-step rollout, confirmed by tracking agent_left.x
# directly; user: "真ん中の壁に張り付いてる...一度もボールに触ってない").
# Now each protected input must reach BOTH the forward and backward
# outputs specifically, not just "an" output.
protected_inputs = [-5, -1]  # bx (ball x), x (agent's own x)
protected_outputs = [0, 1]   # forward, backward -- both needed to track
                             # the ball in either direction


def repair_protected_inputs(genome, config):
  for src in protected_inputs:
    for target in protected_outputs:
      if has_path_to_node(src, target, genome.connections):
        continue
      key = (src, target)
      if key in genome.connections:
        genome.connections[key].enabled = True
      else:
        genome.add_connection(config.genome_config, src, target,
                               weight=random.gauss(0, 1), enabled=True)


def eval_genomes(genomes, config):
  best_genome = None
  for genome_id, genome in genomes:
    repair_protected_inputs(genome, config)
    candidate_policy = NeatPolicy(genome, config)
    scores = []
    for _ in range(n_rollouts):
      opponent = archive.sample()
      # opponent plays as policy_right; negate for the candidate's score.
      score, _ = rollout(env, opponent, candidate_policy)
      scores.append(-score)
    # parsimony pressure: directly penalize node count so bloat has to
    # "pay for itself" with a proportionally large score improvement,
    # rather than being only indirectly discouraged via delete rates
    # (see neat/EXPERIMENT_LOG.md, run 5's bloat postmortem).
    genome.fitness = sum(scores) / len(scores) - complexity_penalty * len(genome.nodes)
    if best_genome is None or genome.fitness > best_genome.fitness:
      best_genome = genome

  archive.maybe_add_champion(best_genome, config)


# Start from run 11's champion_0035 (the first genuinely competent
# rallying genome, see neat/EXPERIMENT_LOG.md) instead of a fresh random
# population, since the point of this run is "given a genome that can
# already rally, does training against stronger/external opponents from
# here push it further" -- not re-deriving basic competence from scratch.
seed_genome_path = os.path.join(local_dir, "neat_run11_full", "champion_0035.pkl")


def seed_population(population, config, seed_genome_path, pop_size):
  with open(seed_genome_path, "rb") as f:
    seed_genome = pickle.load(f)
  new_population = {}
  for i in range(pop_size):
    genome_id = next(population.reproduction.genome_indexer)
    g = copy.deepcopy(seed_genome)
    g.key = genome_id
    if i > 0:  # keep one exact copy, mutate the rest for initial diversity
      g.mutate(config.genome_config)
    new_population[genome_id] = g
  population.population = new_population
  population.species.speciate(config, population.population, population.generation)


def run():
  config = neat.Config(
      neat.DefaultGenome,
      neat.DefaultReproduction,
      neat.DefaultSpeciesSet,
      neat.DefaultStagnation,
      config_path,
  )

  population = neat.Population(config)
  seed_population(population, config, seed_genome_path, config.pop_size)
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

  print("best fitness (vs sampled archive opponents):", winner.fitness)
  print("self-play dethronings:", archive.generation)


if __name__ == "__main__":
  run()
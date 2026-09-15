# Trains an agent using NEAT (neat-python), evolving both the topology and
# weights of a small feedforward network.
#
# Run 16: reintroduce champion_0035 into the archive. Testing run 15's
# champion_0040 directly against champion_0035 on screen was a rout
# (0035 won 5-0) -- 0035 had long since rotated out of the archive
# (only the last 9 champions are kept), so nothing since had to prove
# itself against it. User's request: put 0035 back in and keep training,
# rather than let old, genuinely strong genomes be forgotten just
# because they're old. Resumes run 15 from its generation-2613
# checkpoint with champion_0035 manually appended to the archive
# alongside the normal reloaded set.
#
# Run 15: reject degenerate (non-moving) challengers before crowning
# them. champion_0039 (run 14) passed the 20-rollout dethroning score
# check but turned out to spend the whole match pinned against its own
# back wall, barely reacting to bx -- the user caught this on screen
# ("ずっと右の壁に張り付いてる...何もしてないんだよ") and asked for an
# explicit "is it actually moving" check in the dethroning process
# itself, not just a score threshold. Added `measure_x_range()`: run one
# rollout of the challenger and track its own x position; if the range
# (max-min) doesn't clear `min_movement_range`, reject immediately
# without even running the expensive 20-rollout score test (cheap gate
# first). Resumes run 14 from its generation-2522 checkpoint.
#
# Run 14: reduce fitness noise. Run 13 plateaued at champion_0038 for
# 1000+ generations; checking why, the population's best genome by its
# own (n_rollouts=3) fitness (1.94) was retested under the exact
# dethroning-check conditions (4 sampled archive opponents x 5 rollouts
# = 20 total) and scored -3.9 -- nowhere close. The per-generation
# fitness used for NEAT's own selection is still dominated by noise from
# averaging only 3 rollouts (the same diagnosis as runs 1-2, which never
# fully went away once self-play made per-opponent variance additionally
# noisy). Resumes run 13 from its generation-2021 checkpoint with
# n_rollouts raised 3 -> 6, rather than restarting from scratch -- the
# population's actual genomes are unaffected by this change, only how
# many rollouts their fitness gets averaged over going forward.
#
# Run 13: fix single-species genetic drift. Run 11 (this same file, at
# logdir "neat_run11_full") stayed in one species for 467+ generations
# and its best fitness clearly trended down over time (not just
# plateaued) -- see neat_config_selfplay_v7.txt's header comment for the
# measurements. Resumes run 11 from its generation-770 checkpoint (not
# from scratch -- that population's actual progress is real, just
# undiversified) under neat_config_selfplay_v7.txt's much lower
# compatibility_threshold (1.6, empirically calibrated against that
# checkpoint's live population), with an explicit re-speciate call
# afterward. Also reconstructs the self-play archive as it would have
# looked at that point (TrackingPolicy seed + the last 9 champions
# actually produced by run 11), rather than starting the archive over.
#
# Run 5: self-play curriculum, v2. Run 4's self-play (see
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

import glob
import itertools
import os
import pickle
import random
import shutil

import gym
import neat
import numpy as np

import slimevolleygym
from slimevolleygym import BaselinePolicy, multiagent_rollout as rollout
from slimevolleygym.neat_policy import NeatPolicy

# Settings
random_seed = 612
n_generations = 2387       # resuming from generation 2613 -> reaches 5000 total
save_freq = 10
n_rollouts = 6             # per-genome fitness rollouts, vs opponents sampled from the archive
n_challenge_rollouts = 20  # rollouts used to decide whether to dethrone
dethrone_margin = 0.5      # challenger's mean score (vs current strongest) must exceed this
archive_max_size = 10      # keep the seed + up to this many past champions
complexity_penalty = 0.01  # subtracted per genome node, to directly select against bloat
min_movement_std = 3.0     # challenger's own x must have at least this much stdev
                           # over a match or it's rejected as "stuck at a wall"
                           # before even running the score test (see
                           # measure_movement_std's docstring: range alone was
                           # fooled by a genome that moved briefly then froze)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config_selfplay_v7.txt")

resume_checkpoint = os.path.join(local_dir, "neat_run15_full", "checkpoint-2613")
# champions 1-35 in run 11's logdir, 36-38 in run 13's, 39 in run 14's, 40 in run 15's
resume_logdirs = [os.path.join(local_dir, "neat_run11_full"),
                   os.path.join(local_dir, "neat_run13_full"),
                   os.path.join(local_dir, "neat_run14_full"),
                   os.path.join(local_dir, "neat_run15_full")]
# manually reintroduced into the archive on top of the normal reload
# (see header comment) -- champion_0035 had rotated out and nothing
# since had to prove itself against it.
reintroduce_champions = [os.path.join(local_dir, "neat_run11_full", "champion_0035.pkl")]

logdir = "neat_run16_full"
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


def measure_movement_std(policy, opponent, n_steps=600):
  """ Roll `policy` out (as policy_left) against `opponent` and return
  the standard deviation of its own x position over the match. Added
  after champion_0039 (run 14) passed the 20-rollout dethroning score
  check while spending nearly the entire match pinned against its own
  back wall, barely reacting to bx -- the user caught this on screen and
  asked for an explicit "is it actually moving" check, not just a score
  threshold (see neat/EXPERIMENT_LOG.md, run 14 postmortem).

  First attempt used range (max-min) instead of stdev and was fooled:
  champion_0039 moved for its first ~20 of 568 steps then froze at one
  wall for the rest, which is still a range of ~10 (out of ~23.5 usable
  court width) even though it's degenerate -- stdev correctly separates
  "moved briefly, then stuck" (std ~1.0) from genuinely active play
  (champions 0035/0038: std ~6-7) since it's dominated by how the
  position is distributed over time, not just its extremes. """
  obs_right = env.reset()
  obs_left = obs_right
  done = False
  agent_left = env.unwrapped.game.agent_left
  xs = []
  t = 0
  while not done and t < n_steps:
    action_right = opponent.predict(obs_right)
    action_left = policy.predict(obs_left)
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']
    xs.append(agent_left.x)
    t += 1
  return np.std(xs) if xs else 0.0


class OpponentArchive:
  """ Holds a rotating set of self-play opponents: the original
  TrackingPolicy seed plus up to `archive_max_size` past champions.
  Genomes are evaluated against opponents sampled from the whole
  archive, not just the single most recent champion, to reduce
  overfitting to one narrow lineage. """

  def __init__(self):
    self.archive = [TrackingPolicy()]
    self.generation = 0

  def resume_from(self, resume_logdirs, config):
    """ Reconstruct the archive as it would have looked when the run
    being resumed left off -- load the champions it actually produced
    (possibly split across multiple prior runs' logdirs) instead of
    starting the self-play archive over from just the seed (see
    neat/EXPERIMENT_LOG.md, runs 13-14). """
    champion_files = sorted(
        f for d in resume_logdirs for f in glob.glob(os.path.join(d, "champion_*.pkl")))
    self.generation = len(champion_files)  # total dethronings so far, for future filenames
    for path in champion_files[-(archive_max_size - 1):]:
      with open(path, "rb") as f:
        genome = pickle.load(f)
      self.archive.append(NeatPolicy(genome, config))
    print(f"RESUME: loaded {len(self.archive) - 1} past champions into the archive "
          f"(of {self.generation} ever produced)")

  def sample(self):
    return random.choice(self.archive)

  def maybe_add_champion(self, challenger_genome, config):
    challenger_policy = NeatPolicy(challenger_genome, config)

    # Cheap gate first: reject a challenger that barely moves before
    # ever running the expensive 20-rollout score test (see
    # measure_movement_std's docstring for why this check exists).
    movement_std = measure_movement_std(challenger_policy, self.sample())
    if movement_std < min_movement_std:
      print(f"SELFPLAY: rejected challenger (movement_std={movement_std:.2f} < "
            f"{min_movement_std}, likely stuck at a wall)")
      return False

    # Test against several opponents sampled from across the whole
    # archive, not just the single most recent champion -- run 6 found
    # that gating on "beats only the latest champion" let genomes
    # collapse to a narrow fixed action that happened to exploit that
    # one opponent, rather than requiring genuinely broad competence
    # (see neat/EXPERIMENT_LOG.md, run 6 postmortem).
    n_test_opponents = min(4, len(self.archive))
    test_opponents = random.sample(self.archive, n_test_opponents)
    rollouts_each = max(1, n_challenge_rollouts // n_test_opponents)
    scores = []
    for opponent in test_opponents:
      for _ in range(rollouts_each):
        # opponent plays as policy_right; negate for the challenger's score.
        score, _ = rollout(env, opponent, challenger_policy)
        scores.append(-score)
    mean_score = sum(scores) / len(scores)
    if mean_score > dethrone_margin:
      self.archive.append(challenger_policy)
      if len(self.archive) > archive_max_size:
        del self.archive[1]  # drop the oldest champion, but keep the seed (index 0)
      self.generation += 1
      model_filename = os.path.join(logdir, "champion_" + str(self.generation).zfill(4) + ".pkl")
      with open(model_filename, "wb") as out:
        pickle.dump(challenger_genome, out)
      n_nodes, n_enabled, working_inputs = connectivity_report(challenger_genome)
      print(f"SELFPLAY: new champion (gen {self.generation}), "
            f"mean score vs {n_test_opponents} sampled archive opponents: {mean_score:.2f}, "
            f"archive size: {len(self.archive)}, "
            f"connectivity: {n_nodes} nodes / {n_enabled} enabled conns / "
            f"{working_inputs}/12 inputs reach an output")
      return True
    return False


archive = OpponentArchive()

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


def run():
  config = neat.Config(
      neat.DefaultGenome,
      neat.DefaultReproduction,
      neat.DefaultSpeciesSet,
      neat.DefaultStagnation,
      config_path,
  )

  archive.resume_from(resume_logdirs, config)

  for path in reintroduce_champions:
    with open(path, "rb") as f:
      reintroduced_genome = pickle.load(f)
    archive.archive.append(NeatPolicy(reintroduced_genome, config))
    print(f"RESUME: manually reintroduced {path} into the archive "
          f"(size now {len(archive.archive)})")

  # Resume run 11's actual population (real progress, just
  # undiversified) under the new, much lower compatibility_threshold.
  # restore_checkpoint keeps the OLD species-set object, whose single
  # existing species still has a representative genome from before --
  # NEAT's speciate() only compares against *existing* representatives
  # and opens a new species when nothing matches, so reusing that stale
  # single-species object reproduced only 1 species even at the new
  # threshold. Replacing it with a brand-new, empty DefaultSpeciesSet
  # before speciating (exactly how this was calibrated/tested
  # beforehand) makes it split for real.
  population = neat.Checkpointer.restore_checkpoint(resume_checkpoint, new_config=config)
  population.species = neat.DefaultSpeciesSet(config.species_set_config, population.reporters)
  population.species.speciate(config, population.population, population.generation)

  # config.genome_config.node_indexer (the counter that assigns new
  # hidden-node IDs) is NOT part of the saved innovation tracker and
  # isn't restored by restore_checkpoint -- it lazily initializes from
  # whichever genome's node_dict happens to be mutated first, which can
  # be far behind the true max node ID across the whole population after
  # a resume. That caused an AssertionError (duplicate node key) 3
  # generations into this run's first attempt. Set it explicitly above
  # every existing node ID in the restored population before running.
  max_node_id = max(
      (max(g.nodes.keys(), default=-1) for g in population.population.values()),
      default=-1)
  config.genome_config.node_indexer = itertools.count(max_node_id + 1)

  print(f"RESUME: restored generation {population.generation} from {resume_checkpoint}, "
        f"re-speciated into {len(population.species.species)} species under "
        f"compatibility_threshold={config.species_set_config.compatibility_threshold}, "
        f"node_indexer starting at {max_node_id + 1}")

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
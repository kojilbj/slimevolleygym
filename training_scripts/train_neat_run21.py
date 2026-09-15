# Trains an agent using NEAT (neat-python), evolving both the topology and
# weights of a small feedforward network.
#
# Run 21: gate on actually clearing the net, not just touching the ball.
# On-screen review of run20's current champion (0046) found it repeatedly
# hitting the ball on its own side without ever sending it over the net --
# user: "今のchampionはひたすら触り続けてボールを相手に返してなかったよ".
# The existing avg_touches gate (min_avg_touches=2.5) only checks that the
# genome plays the ball at all; it never checks whether any of those
# touches actually cross the net, so this exact failure mode passed
# straight through it. The run 20 decisiveness penalty doesn't catch it
# either, since it's keyed on touches-per-point-WON and this behavior may
# never actually win (or lose) the point it's stalling.
#
# Adds a net_clear_rate check to measure_behavior()/maybe_add_champion():
# after each of the candidate's touches, watch whether the ball's x
# subsequently reaches the far side of the net (abs(x) > REF_WALL_WIDTH/2)
# before either the candidate touches it again or the rollout ends: if not,
# that touch never actually returned the ball, no matter how many touches
# the genome racked up. Rejects any challenger whose average
# net_clear_rate is below min_net_clear_rate.
#
# Stops run 20 (was at generation ~262, no new champion crowned since the
# gen-121 resume) and resumes from its latest checkpoint with this new
# gate active, same as every previous run-to-run gate addition.
#
# Run 20: reward decisive returns, not just ball contact. Run 19's
# champion_0037 (10/12 inputs, tracks the ball well, 79 touches in one
# BaselinePolicy match) still lost matches it should arguably have
# closed out. User: "今はボールはめちゃくちゃ追えてるからあとは一回ボール
# を打ち返すための接触回数を減らしてより攻撃的にするほうがいいと思う".
# Measured "touches per point WON" for two known-good champions:
# champion_0035 averaged 1.41 touches/win, champion_0037 (which beat
# 0035 head-to-head) averaged 2.69 -- 0035 is actually more efficient
# per point despite the head-to-head loss, confirming touch-efficiency
# is a real, currently-unmeasured quality dimension.
#
# Deliberately measured only over points the candidate WINS: a genome
# that avoids the ball and loses fast contributes no data point to this
# ratio at all (rather than looking artificially "efficient"), so this
# can't be gamed by disengaging -- the existing movement/touch gates
# already handle that failure mode separately.
#
# rollout_with_stats() replaces the plain rollout() helper in
# eval_genomes (every genome, every generation, not just the dethroning
# gate) so this becomes actual selection pressure, not just a crowning
# checkpoint. Adds a small fitness penalty proportional to average
# touches-per-win, applied only when the genome won at least one point
# across its evaluation rollouts.
#
# Run 19: fix a species-count crash. Run 18 (seeded from champion_0035)
# crashed at generation 14 with "could not match pop_size=128 with
# min_species_size=2" -- species count grew 33 -> 45 -> 65 in just 14
# generations (v8's compatibility_threshold=0.7 was only calibrated
# against the generation-0 snapshot; genetic distance grows as genomes
# add structure over subsequent generations). Also found neat-python
# computes the *effective* min_species_size as
# max(config value, elitism) (reproduction.py:219) -- elitism=2 was
# silently setting the real floor to 2. neat_config_selfplay_v9.txt
# raises compatibility_threshold to 1.2 and drops elitism to 1 (halves
# the crash threshold from num_species>64 to num_species>128). Resumes
# run 18 from its generation-10 checkpoint (10 real generations of
# progress from the champion_0035 seed, including champions 0043-0052)
# rather than restarting from scratch again.
#
# Run 18: restart the population from champion_0035, not from wherever
# it drifted to. Testing champion_0042 (run 16's champion) against
# champion_0035 three times: 0035 won all three (-4, -5, -5). The
# population has been evolving since ~generation 900 (when 0035 was
# crowned) and has mostly been producing genomes weaker than 0035 ever
# since (borne out by run 17's own gates rejecting dozens of challengers
# as degenerate) -- user's question: "つまり王座は35の状態からやり直した
# ほうがいいってことだよね？". Rather than keep training a population that
# may be broadly drifted/contaminated, this seeds a fresh population of
# 128 mutated copies of champion_0035 (like run 12's seed_population(),
# but WITHOUT run 12's mistake of also mixing in a harsh external
# opponent pool from generation 0 -- keeps this a clean single-variable
# test). Archive reset to just [TrackingPolicy, champion_0035] (not the
# accumulated 36-42 lineage, since those are exactly the weaker genomes
# this run is trying to avoid re-deriving from). Run 15-17's movement
# and ball-touch gates stay in place so only genuinely-competent
# challengers can ever be crowned going forward.
#
# Run 17: also gate on ball touches, not just movement. Run 16's
# champion_0042 cleared the movement-std gate (it does move) but only
# touched the ball once in a 689-step match, moving in a way
# anti-correlated with the ball (corr -0.788) -- user: "接触1回とか論外
# だよ" / "接触回数も閾値を用意して除外したほうがいいんじゃない？". The
# movement gate alone doesn't distinguish "plays the ball" from "wanders
# around near it". `measure_behavior()` now returns both x-stdev and
# ball-touch count from the same rollout (no extra cost), averaged over
# `n_behavior_checks=3` rollouts against different sampled opponents
# (single-rollout touch counts are noisy -- even known-good champions
# scored as low as 1 touch in a single earlier test) before crowning.
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

import copy
import glob
import itertools
import re
import os
import pickle
import random
import shutil

import gym
import neat
import numpy as np

import slimevolleygym
from slimevolleygym import BaselinePolicy, REF_WALL_WIDTH, multiagent_rollout as rollout
from slimevolleygym.neat_policy import NeatPolicy

# Settings
random_seed = 612
n_generations = 2738       # resuming from generation 262 -> reaches 3000 total for this lineage
save_freq = 10
n_rollouts = 6             # per-genome fitness rollouts, vs opponents sampled from the archive
n_challenge_rollouts = 20  # rollouts used to decide whether to dethrone
dethrone_margin = 0.5      # challenger's mean score (vs current strongest) must exceed this
archive_max_size = 10      # keep the seed + up to this many past champions
complexity_penalty = 0.01  # subtracted per genome node, to directly select against bloat
decisiveness_penalty = 0.2  # subtracted per touch-per-point-won, to reward closing
                            # out points in fewer touches (only applied when the
                            # genome won >=1 point across its evaluation rollouts --
                            # see rollout_with_stats()/eval_genomes for why this
                            # can't be gamed by avoiding the ball)
min_movement_std = 3.0     # challenger's own x must have at least this much stdev
                           # (averaged over n_behavior_checks rollouts) or it's
                           # rejected as "stuck at a wall" (see
                           # measure_behavior's docstring: range alone was fooled
                           # by a genome that moved briefly then froze)
min_avg_touches = 2.5      # ...and must average at least this many ball touches
                           # per rollout, or it's rejected as "wanders around but
                           # never actually plays the ball" (run 16 postmortem).
                           # Calibrated against archive-sampled opponents (matching
                           # what this gate actually uses): known-good champions
                           # (0035, 0038, 0041) averaged 4.2-8.2 touches, known-bad
                           # ones (0040, 0042) averaged 0.4-1.4 -- 2.5 sits cleanly
                           # between them. (User caught that the first cut, 1.5,
                           # was too close to 0042's own score to mean anything.)
n_behavior_checks = 3      # rollouts to average the above over -- a single
                           # rollout's touch count is noisy (even known-good
                           # champions have scored as low as 1 in one match)
min_net_clear_rate = 0.10  # ...and at least this fraction of those touches must
                           # actually send the ball past the net (see
                           # measure_behavior's net_clear tracking below), or
                           # it's rejected as "plays the ball but doesn't return
                           # it" (run 21 postmortem: champion_0046 passed
                           # min_avg_touches while mostly juggling on its own
                           # side). Calibrated against TrackingPolicy, averaged
                           # over 5 rollouts: known-good champions 0035 and
                           # 0037 (run19) scored 0.30 and 0.16 respectively;
                           # champion_0046, the juggling genome that prompted
                           # this gate, scored 0.03 -- 0.10 sits cleanly
                           # between them.

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat_config_selfplay_v9.txt")

resume_checkpoint = os.path.join(local_dir, "neat_run20_full", "checkpoint-262")
# run 18's, 19's, and 20's own champions (run 20 crowned none, but include
# it for consistency/future-proofing), deliberately NOT the weaker 36-42
# lineage from before the champion_0035 restart.
resume_logdirs = [os.path.join(local_dir, "neat_run18_full"),
                   os.path.join(local_dir, "neat_run19_full"),
                   os.path.join(local_dir, "neat_run20_full")]
# manually reintroduced on top of the normal reload -- champion_0035 is
# the whole reason this lineage restarted and should stay available
# even once newer champions rotate past it.
reintroduce_champions = [os.path.join(local_dir, "neat_run11_full", "champion_0035.pkl")]

logdir = "neat_run21_full"
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


def measure_behavior(policy, opponent, n_steps=600):
  """ Roll `policy` out (as policy_left) against `opponent` and return
  (x_stdev, ball_touches, touches_returned) from the same rollout. x_stdev
  added after champion_0039 (run 14) passed the 20-rollout dethroning
  score check while spending nearly the entire match pinned against its
  own back wall, barely reacting to bx -- the user caught this on screen
  and asked for an explicit "is it actually moving" check, not just a
  score threshold (see neat/EXPERIMENT_LOG.md, run 14 postmortem).

  First attempt used range (max-min) instead of stdev and was fooled:
  champion_0039 moved for its first ~20 of 568 steps then froze at one
  wall for the rest, which is still a range of ~10 (out of ~23.5 usable
  court width) even though it's degenerate -- stdev correctly separates
  "moved briefly, then stuck" (std ~1.0) from genuinely active play
  (champions 0035/0038: std ~6-7) since it's dominated by how the
  position is distributed over time, not just its extremes.

  ball_touches added after champion_0042 (run 16) cleared the movement
  gate (it does move) but only touched the ball once in a 689-step match
  -- moving around near the ball isn't the same as playing it. Counts
  actual collisions by wrapping ball.bounce() (checking
  ball.isColliding() after env.step() returns is too late -- the
  collision already happened and the ball has moved on, see
  neat/EXPERIMENT_LOG.md's ball-touch-undercounting postmortem), and
  re-wraps on every Game.newMatch() (called on each scored point, which
  replaces self.ball with a fresh Particle, silently dropping any
  previous wrap).

  touches_returned added in run 21 after champion_0046 (run 19/20)
  cleared the ball_touches gate while, on screen, mostly juggling the
  ball on its own side without sending it back over the net -- the
  touch count alone can't tell "played the ball" from "actually returned
  it". After each of the candidate's touches, watches whether the
  ball's x subsequently clears the far edge of the net
  (REF_WALL_WIDTH/2, i.e. reaches agent_right's side) before either the
  candidate touches it again or the rollout ends; if not, that touch
  never returned anything, no matter how many more touches follow. """
  game = env.unwrapped.game
  touches = [0]
  touches_returned = [0]
  pending_return = [False]
  net_edge_x = REF_WALL_WIDTH / 2

  def wrap_bounce():
    agent_left = game.agent_left
    original_bounce = game.ball.bounce

    def counting_bounce(p):
      if p is agent_left:
        touches[0] += 1
        pending_return[0] = True
      return original_bounce(p)

    game.ball.bounce = counting_bounce

  obs_right = env.reset()
  wrap_bounce()
  original_new_match = game.newMatch

  def wrapped_new_match():
    original_new_match()
    wrap_bounce()
    pending_return[0] = False  # a fresh ball hasn't been touched yet

  game.newMatch = wrapped_new_match

  obs_left = obs_right
  done = False
  agent_left = game.agent_left
  xs = []
  t = 0
  while not done and t < n_steps:
    action_right = opponent.predict(obs_right)
    action_left = policy.predict(obs_left)
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']
    if pending_return[0] and game.ball.x > net_edge_x:
      touches_returned[0] += 1
      pending_return[0] = False
    xs.append(agent_left.x)
    t += 1
  game.newMatch = original_new_match
  return (np.std(xs) if xs else 0.0), touches[0], touches_returned[0]


def rollout_with_stats(policy, opponent, n_steps=3000):
  """ Like `rollout()`/`multiagent_rollout` (policy as policy_left,
  opponent as policy_right), but also returns ball-touch stats needed
  for the touches-per-point-won fitness term: (score, touches,
  points_won, touches_in_won_points). `score` is already negated to the
  candidate's (left's) perspective, matching how eval_genomes and
  maybe_add_champion use plain rollout() elsewhere in this file.

  touches_in_won_points only accumulates touches from points the
  candidate actually won -- a genome that avoids the ball and loses
  fast contributes no data here at all, rather than looking
  artificially "efficient" (see run 20's header comment). Reuses
  measure_behavior's collision-counting approach (wrapping
  ball.bounce(), re-wrapped on every Game.newMatch()). """
  game = env.unwrapped.game
  touches = [0]
  touches_since_point = [0]
  points_won = [0]
  touches_in_won_points = [0]

  def wrap_bounce():
    agent_left = game.agent_left
    original_bounce = game.ball.bounce

    def counting_bounce(p):
      if p is agent_left:
        touches[0] += 1
        touches_since_point[0] += 1
      return original_bounce(p)

    game.ball.bounce = counting_bounce

  obs_right = env.reset()
  wrap_bounce()
  original_new_match = game.newMatch

  def wrapped_new_match():
    original_new_match()
    wrap_bounce()

  game.newMatch = wrapped_new_match

  obs_left = obs_right
  done = False
  total_reward = 0.0
  t = 0
  while not done and t < n_steps:
    action_right = opponent.predict(obs_right)
    action_left = policy.predict(obs_left)
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']
    total_reward += reward
    if reward != 0:
      if reward < 0:  # right (opponent) lost the point -> left (candidate) won it
        points_won[0] += 1
        touches_in_won_points[0] += touches_since_point[0]
      touches_since_point[0] = 0
    t += 1

  game.newMatch = original_new_match
  # total_reward is right's (opponent's) perspective; negate for the
  # candidate's, matching plain rollout()'s convention as used elsewhere
  # in this file.
  return -total_reward, touches[0], points_won[0], touches_in_won_points[0]


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
    # Parse the actual champion number from each filename rather than
    # just counting files -- run 19 set self.generation = file count,
    # which undercounted whenever resume_logdirs didn't include every
    # prior run's directory (e.g. resuming from only run 18's 10 files
    # started renumbering at 10 instead of continuing from the true
    # historical count of 52), causing new champions' filenames to
    # collide/renumber from a much lower point than intended.
    numbers = [int(re.search(r"champion_(\d+)\.pkl$", f).group(1)) for f in champion_files]
    self.generation = max(numbers) if numbers else 0
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

    # Cheap gate first: reject a challenger that barely moves, or moves
    # but never actually plays the ball, before ever running the
    # expensive 20-rollout score test (see measure_behavior's docstring
    # for why both checks exist). Averaged over several rollouts against
    # different sampled opponents since a single rollout's touch count
    # is noisy.
    behaviors = [measure_behavior(challenger_policy, self.sample())
                 for _ in range(n_behavior_checks)]
    avg_std = sum(b[0] for b in behaviors) / len(behaviors)
    avg_touches = sum(b[1] for b in behaviors) / len(behaviors)
    total_touches = sum(b[1] for b in behaviors)
    total_touches_returned = sum(b[2] for b in behaviors)
    if avg_std < min_movement_std:
      print(f"SELFPLAY: rejected challenger (avg movement_std={avg_std:.2f} < "
            f"{min_movement_std}, likely stuck at a wall)")
      return False
    if avg_touches < min_avg_touches:
      print(f"SELFPLAY: rejected challenger (avg ball_touches={avg_touches:.2f} < "
            f"{min_avg_touches}, moves but doesn't play the ball)")
      return False
    # Aggregated across all behavior checks rather than averaging each
    # check's own ratio, so a single low-touch check can't swing the
    # rate on a small denominator -- the avg_touches gate above already
    # guarantees total_touches > 0 here.
    net_clear_rate = total_touches_returned / total_touches
    if net_clear_rate < min_net_clear_rate:
      print(f"SELFPLAY: rejected challenger (net_clear_rate={net_clear_rate:.2f} < "
            f"{min_net_clear_rate}, plays the ball but doesn't return it)")
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
    total_points_won = 0
    total_touches_in_won_points = 0
    for _ in range(n_rollouts):
      opponent = archive.sample()
      score, touches, points_won, touches_in_won_points = rollout_with_stats(candidate_policy, opponent)
      scores.append(score)
      total_points_won += points_won
      total_touches_in_won_points += touches_in_won_points

    # parsimony pressure: directly penalize node count so bloat has to
    # "pay for itself" with a proportionally large score improvement,
    # rather than being only indirectly discouraged via delete rates
    # (see neat/EXPERIMENT_LOG.md, run 5's bloat postmortem).
    fitness = sum(scores) / len(scores) - complexity_penalty * len(genome.nodes)

    # decisiveness: penalize needing many touches to close out a point
    # won, only when there's data to judge (the genome won at least one
    # point across its n_rollouts matches) -- a genome that avoids the
    # ball and loses fast contributes no points_won at all, so this
    # can't be gamed by disengaging (the movement/touch gates in
    # maybe_add_champion already handle that failure mode). See run 20's
    # header comment: champion_0035 averaged 1.41 touches/win,
    # champion_0037 (which beat 0035 head-to-head) averaged 2.69 --
    # touch-efficiency per point is a real, previously-unmeasured
    # quality dimension.
    if total_points_won > 0:
      touches_per_win = total_touches_in_won_points / total_points_won
      fitness -= decisiveness_penalty * touches_per_win

    genome.fitness = fitness
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

  # Archive = run 18's own champions (43-52) + champion_0035 explicitly
  # kept alongside them (see header comment).
  archive.resume_from(resume_logdirs, config)
  for path in reintroduce_champions:
    with open(path, "rb") as f:
      reintroduced_genome = pickle.load(f)
    archive.archive.append(NeatPolicy(reintroduced_genome, config))
    print(f"RESUME: manually reintroduced {path} into the archive "
          f"(size now {len(archive.archive)})")

  # Resume run 18's actual population from its generation-10 checkpoint
  # (10 real generations of progress from the champion_0035 seed) rather
  # than restarting from scratch again. Re-speciate with a brand-new,
  # empty DefaultSpeciesSet under the new (higher) compatibility_threshold
  # -- reusing the old species object would keep comparing against its
  # stale representatives instead of re-clustering for real (see run
  # 13's postmortem).
  population = neat.Checkpointer.restore_checkpoint(resume_checkpoint, new_config=config)
  population.species = neat.DefaultSpeciesSet(config.species_set_config, population.reporters)
  population.species.speciate(config, population.population, population.generation)

  # config.genome_config.node_indexer safety fix (see run 14's
  # postmortem) -- always apply after any population manipulation.
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
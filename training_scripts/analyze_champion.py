# Analyzes a saved NEAT self-play champion: connectivity (which inputs
# reach an output), and behavior in an actual rollout vs TrackingPolicy
# (score, ball touches, correlation between own x and ball x). Optional
# on-screen render. Used for periodic checkpoints during long runs.

import argparse
import pickle
import time

import gym
import neat
import numpy as np

import slimevolleygym
from slimevolleygym.neat_policy import NeatPolicy


class TrackingPolicy:
  def predict(self, obs):
    x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy = obs
    forward = 1 if bx > x + 0.05 else 0
    backward = 1 if bx < x - 0.05 else 0
    jump = 1 if (by < 0.5 and abs(bx - x) < 0.5) else 0
    return np.array([forward, backward, jump])


def has_path_to_node(start, target, connections, visited=None):
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


INPUT_LABELS = {-1: 'x', -2: 'y', -3: 'vx', -4: 'vy', -5: 'bx', -6: 'by',
                -7: 'bvx', -8: 'bvy', -9: 'ox', -10: 'oy', -11: 'ovx', -12: 'ovy'}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--genome', required=True)
  parser.add_argument('--config', default='neat_config_selfplay_v6.txt')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--steps', type=int, default=3000)  # matches env's own t_limit, so a match always plays to its real conclusion
  args = parser.parse_args()

  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
  with open(args.genome, 'rb') as f:
    genome = pickle.load(f)
  policy = NeatPolicy(genome, config)
  tracker = TrackingPolicy()

  n_nodes = len(genome.nodes)
  n_enabled = sum(1 for c in genome.connections.values() if c.enabled)
  working = sorted(
      INPUT_LABELS[k] for k in range(-1, -13, -1)
      if any(c.enabled and has_path_to_node(dst, t, genome.connections)
             for (s, dst), c in genome.connections.items() if s == k
             for t in (0, 1, 2))
  )
  print(f"connectivity: nodes={n_nodes} enabled_conns={n_enabled} "
        f"working_inputs={len(working)}/12 {working}")

  # Candidate plays as policy_right (the env's own default "self" side --
  # reward is already documented/coded as right-side perspective, see
  # slimevolley.py:583 "negated, since we want reward to be from the
  # perspective of right agent being trained"), TrackingPolicy as
  # policy_left. So `total_reward` below is directly the candidate's own
  # score (positive = candidate winning) with no sign-flip needed to
  # interpret it -- avoids the mixed-up-perspective mistake made when
  # this script briefly had the candidate on the left instead.
  env = gym.make("SlimeVolley-v0")
  env.seed(args.seed)
  obs_right = env.reset()
  obs_left = obs_right
  done = False
  agent_x, ball_x = [], []
  touches = [0]  # list so the closure below can mutate it
  total_reward = 0
  t = 0

  # Count actual collisions by wrapping ball.bounce() directly, called
  # inside Game.step() the instant a collision is detected. Checking
  # ball.isColliding(agent_right) *after* env.step() returns is too late
  # -- by then the ball has already been bounced away and is no longer
  # colliding, so that check almost never fires (user caught this: saw
  # 10+ real touches on screen while the script reported only 0-2).
  #
  # Every scored point calls Game.newMatch(), which replaces
  # self.ball with a brand-new Particle -- wrapping bounce() on the
  # ball object once, before the loop, only counts touches up to the
  # first point scored, then silently stops working (also caught by the
  # user's higher on-screen count). Re-wrap the fresh ball's bounce()
  # every time newMatch() runs instead.
  game = env.unwrapped.game
  agent_right = game.agent_right

  def wrap_ball_bounce():
    original_bounce = game.ball.bounce

    def counting_bounce(p):
      if p is agent_right:
        touches[0] += 1
      return original_bounce(p)

    game.ball.bounce = counting_bounce

  wrap_ball_bounce()
  original_new_match = game.newMatch

  def wrapped_new_match():
    original_new_match()
    wrap_ball_bounce()

  game.newMatch = wrapped_new_match

  while not done and t < args.steps:
    action_right = policy.predict(obs_right)
    action_left = tracker.predict(obs_left)
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']
    total_reward += reward
    agent_x.append(agent_right.x)
    ball_x.append(game.ball.x)
    if args.render:
      env.render()
      time.sleep(0.02)
    t += 1

  agent_x = np.array(agent_x)
  ball_x = np.array(ball_x)
  corr = np.corrcoef(agent_x, ball_x)[0, 1] if len(agent_x) > 1 else float('nan')
  print(f"behavior: steps={t} score(candidate persp)={total_reward} "
        f"ball_touches={touches[0]} corr(agent_x,ball_x)={corr:.3f}")


if __name__ == "__main__":
  main()

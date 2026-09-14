# NEAT vs BaselinePolicy — Experiment Log

Goal: evolve a SlimeVolley-v0 agent against the built-in `BaselinePolicy`
using neat-python, and record the trial-and-error along the way.

## Setup

- `slimevolleygym/neat_policy.py` — wraps a neat-python genome into the
  `predict(obs) -> action` interface used elsewhere in this repo. Network:
  12 inputs (env observation), 3 sigmoid outputs thresholded at 0.5 to
  produce the `MultiBinary(3)` action.
- `training_scripts/train_neat.py` — evolution loop. Each genome plays one
  episode as `policy_left` against `BaselinePolicy` as `policy_right`
  (`multiagent_rollout`); fitness = `-score` (score is reported from
  `policy_right`'s / baseline's perspective).
- `training_scripts/neat_config.txt` — pop_size 128, feed-forward,
  starts fully connected with 0 hidden nodes, `compatibility_threshold`
  3.0, standard NEAT mutation/speciation rates.
- `training_scripts/eval_neat.py` — added afterwards to evaluate a saved
  genome over many episodes (single-episode fitness turned out to be a
  very noisy signal, see below).
- Dependency: `neat-python==2.0.0` added to `requirements.txt`.

## Run 1 (production run)

`python train_neat.py`, seed 612, pop_size 128, 1000 generations,
checkpoints every 10 generations.

Progress (from `neat_train.log`):

| generation | best fitness | #species | note |
|---|---|---|---|
| ~15 | -3.0 | 1 | stagnation counter climbing (8-9) — worried this would collapse to one species |
| 286 | -2.0 | 2 | species split, best score so far |
| 365 | -3.0 | 5 | more species, but fitness back down (noise) |
| 999 (final) | -3.0 | 4 | `best.pkl` saved, `checkpoint-1000` written |

Wall clock: ~1.9s/generation → ~32 minutes total.

Takeaway while it was running: fitness looked like it was slowly
improving (-5 → -3 → -2) then plateaued around -3 for the rest of the
run despite species turnover continuing. Read at the time as "healthy
evolutionary dynamics, difficulty finding further improvement."

## Evaluating `best.pkl`

Single-episode fitness during training is extremely noisy (each genome
is judged on exactly one rollout), so before trusting "-3" as the
genome's real skill, `eval_neat.py` was used to replay it over 100
episodes:

```
episodes: 100
mean score: -4.880 (std 0.382)
wins/draws/losses: 0/0/100
score distribution: {-5: 90, -4: 8, -3: 2}
```

So the "best" genome actually loses essentially every episode by close
to the maximum margin (5-0) — the single training rollout that produced
`-3.0`/`-2.0` was a lucky draw, not representative.

### Sanity check: is the candidate's action even mattering?

The mean/distribution above looked suspiciously identical to a
random-action policy evaluated the same way, so this was checked
directly:

- `RandomPolicy` vs baseline, 100 episodes, same seed: mean **-4.88**,
  distribution `{-5: 90, -4: 8, -3: 2}` — identical to `best.pkl`.
- Three constant-action policies (`[0,0,0]` do-nothing, `[0,0,1]`
  always-jump, `[1,0,0]` always-move-right), 20 episodes, same seed:
  all three gave mean **-4.95**, distribution `{-5: 19, -4: 1}` —
  identical to each other.

This raised the question of whether `otherAction` was even reaching the
left agent (a routing bug would produce exactly this symptom). Verified
directly by stepping the env with different constant left-actions and
reading `env.unwrapped.game.agent_left.x`: the position clearly responds
to the action (moves right, moves left, or stays put as expected). So
action routing is correct — this is not a framework bug.

**Conclusion:** `BaselinePolicy` is strong enough that, within this
setup, essentially any opponent that hasn't learned real ball-return
skill (random, static, or a NEAT genome that only got this far) loses by
about the same margin regardless of its exact behavior. 1000 generations
of pop-128 NEAT, with a single noisy rollout per genome as the fitness
signal, was not enough to escape that regime.

## Run 2 — average fitness over 3 rollouts

Hypothesis: run 1's fitness was dominated by single-episode noise, which
made selection nearly random. Fix: average each genome's fitness over
`n_rollouts = 3` episodes instead of 1 (`train_neat.py`, `logdir =
"neat_run2"`, same seed/pop_size/generations otherwise).

Cost: ~5.5-5.7s/generation (vs. ~1.9s in run 1), ~93 minutes total for
1000 generations, since each genome now needs 3x the simulation.

Progress (from `neat_run2_train.log`):

| generation | best fitness (avg of 3) | #species |
|---|---|---|
| ~130 | -4.33 | several |
| 325 | -4.00 | 3 |
| 999 (final) | -4.33 (checkpoint) / -4.0 reported | 4 |

Note this "-4.0 to -4.33" is already close to the true skill level —
much closer to reality than run 1's "-2.0 to -3.0" during training,
which confirms the noise diagnosis was correct.

### Evaluating run 2's `best.pkl` (100 episodes vs BaselinePolicy)

```
episodes: 100
mean score: -4.880 (std 0.354)
wins/draws/losses: 0/0/100
score distribution: {-5: 89, -4: 10, -3: 1}
```

**Essentially identical to run 1's evaluation** (mean -4.88, same shape
of distribution, 0 wins). Averaging rollouts made the *training-time*
fitness numbers honest (no more illusory "-2.0" plateaus), but it did
**not** improve the actual policy — the population still never learns
to return the ball in a way that denies BaselinePolicy points.

### Conclusion

Noise reduction alone doesn't fix this. Confirms run 1's read on the
real bottleneck: the fitness landscape near "loses to BaselinePolicy"
is close to flat regardless of the opponent's exact behavior (see run
1's random/constant-action sanity check), so there's no gradient for
evolution to climb no matter how cleanly it's measured. The remaining
ideas below (especially #1, reward shaping) target that flatness
directly instead of just reducing measurement noise.

## Run 3 — reward shaping (survival bonus)

Hypothesis: shape fitness with `SurvivalRewardEnv` (+0.01/timestep) so
partial credit for keeping the rally going gives a less-flat gradient
than raw win/loss margin. Implementation switched the candidate to play
as the environment's default "self" (right) agent with `BaselinePolicy`
auto-controlling the left agent (`otherAction=None`), instead of the
explicit dual-`NeatPolicy`-vs-`BaselinePolicy` `multiagent_rollout` used
in runs 1-2 — this is what let `SurvivalRewardEnv` (a plain
`gym.RewardWrapper` around single-argument `env.step(action)`) be reused
directly. `logdir = "neat_run3"`, `n_rollouts = 3` kept from run 2.

### What actually happened: early termination, not convergence

The run stopped at **generation 503**, not 1000, because
`neat_config.txt`'s `fitness_threshold = 5.0` (left over from runs 1-2,
where raw score is capped at ±5) got crossed by the *shaped* fitness
scale, which has no such cap (survival bonus alone can add up to
`0.01 * 3000 = 30` per episode). `neat.Population.run()` checks this
threshold after every single generation regardless of the `n` passed in
(see `neat/population.py:120-124`), so as soon as one genome's shaped
fitness reached 5.0, training ended — this was a stale-config bug, not
evidence of having "solved" the task.

### Evaluating `best.pkl` (100 episodes vs BaselinePolicy, unshaped env)

```
episodes: 100
mean score: -4.880 (std 0.354)
wins/draws/losses: 0/0/100
score distribution: {-5: 89, -4: 10, -3: 1}
```

**Identical to runs 1 and 2 again.** The shaped fitness went from ~-4.3
to 5.63, but real match performance did not move at all.

### Root cause: the survival bonus was reward hacking, not signal

Checked episode length (`rollout()`'s second return value) for run 3's
`best.pkl` vs. a plain random-action policy, both vs. `BaselinePolicy`,
20 episodes each:

| policy | mean episode length |
|---|---|
| run 3 `best.pkl` | 594 steps |
| random actions | 590 steps |

Essentially the same. Episode length here is mostly driven by how long
`BaselinePolicy` itself takes to close out points, not by anything the
candidate does — so the +0.01/step bonus was rewarding a quantity the
candidate barely controls. That made the (already flat) fitness
landscape noisy in a new way rather than genuinely less flat, and let a
lucky-long episode's accumulated bonus cross the stale 5.0 threshold
without the underlying policy improving at all.

### Conclusion

Naive per-timestep survival bonus is the wrong shaping signal here
because episode length isn't a good proxy for "the candidate is
returning the ball better" in this environment. A shaping signal tied
to something the candidate actually influences (e.g. reward per ball
contact/successful return, or the existing reward mode 2 which still
includes real score) would need to be tried instead, and any shaped-fitness
run needs its own `fitness_threshold`/`no_fitness_termination=True`
rather than reusing runs 1-2's config value.

## Run 4 — self-play curriculum

Hypothesis: `BaselinePolicy` is too strong an opponent to learn anything
against from generation 0 (runs 1-3 all confirm this). Instead, use a
self-play arms race: each genome plays a "champion" opponent that starts
as random actions and only gets replaced ("dethroned") once a
generation's best genome beats it by a clear margin — mirrors
`train_ppo_selfplay.py`'s dethroning logic.

Implementation (`train_neat.py`, `logdir = "neat_run4"`):

- `SelfPlayOpponent` holds the current champion (`RandomPolicy`
  initially); `eval_genomes` plays every genome against it
  (`n_rollouts = 3`, as in run 2), then challenges the champion with the
  generation's best genome over `n_challenge_rollouts = 20` episodes; if
  its mean score exceeds `dethrone_margin = 0.5` the champion is
  replaced and checkpointed (`champion_NNNN.pkl`).
- Dedicated `neat_config_selfplay.txt` with `no_fitness_termination =
  True` so this run can't hit run 3's stale-threshold bug (candidates
  legitimately score above the old fixed-opponent 5.0 ceiling early on,
  since the opponent starts at random-action strength).
- No block loop this time — `population.run(eval_genomes,
  n_generations)` called once; `neat.Checkpointer` still saves periodic
  population snapshots, champion checkpoints track self-play progress
  separately.

### Progress

1000 generations completed in full (no early termination this time —
`no_fitness_termination=True` correctly ran to completion; the "meets
fitness threshold" message neat-python prints at the very end is just
its normal "final report" line when that flag is set, not an early
exit). Wall clock ~2h. **172 dethronings** over the run, at a steady
pace throughout (roughly one every ~6 generations), each new champion
typically beating the previous one by +0.5 to +1.5 points.

Checked interim champions against `BaselinePolicy` mid-run for a reality
check (`eval_neat.py`, 50 episodes each):

| checkpoint | vs BaselinePolicy |
|---|---|
| `champion_0040.pkl` (~gen 140 of training) | mean -4.84, 0/50 wins, dist `{-5:43,-4:6,-3:1}` |
| `champion_0118.pkl` (~gen 670 of training) | mean -4.84, 0/50 wins, **identical distribution** |

### Evaluating the final `best.pkl` (100 episodes vs BaselinePolicy)

```
episodes: 100
mean score: -4.880 (std 0.354)
wins/draws/losses: 0/0/100
score distribution: {-5: 89, -4: 10, -3: 1}
```

**Identical to runs 1, 2, and 3.** Despite 172 rounds of self-play
dethroning (each new champion clearly beating the previous one
head-to-head), the final champion performs exactly as badly against
`BaselinePolicy` as every previous run's genome, and as badly as a
random-action policy (see run 1's sanity check).

### Conclusion

All four approaches (raw fitness, averaged fitness, reward shaping,
self-play) converge on the exact same outcome vs. `BaselinePolicy`: mean
score -4.84 to -4.88, 0 wins, near-identical score distributions. For
self-play specifically: the population climbed a real ladder relative
to *itself* (172 clean dethronings, steadily increasing margins), but
that ladder's absolute skill ceiling never reached the level needed to
affect the outcome against `BaselinePolicy` at all — a classic self-play
failure mode where an arms race improves relative standings within a
population without the population ever crossing into genuinely good
absolute play. Likely contributing factors: a feedforward-only network
(no memory), the modest `pop_size=128`/1000-generation budget compared
to what PPO+self-play needed in this same repo (`train_ppo_selfplay.py`
runs `NUM_TIMESTEPS = 1e9`), and no mechanism forcing the self-play
population to diversify beyond beating its own recent lineage.

## Next-step ideas (none tried yet)

1. Much longer self-play run (10x-100x the generations/timesteps), in
   line with what `train_ppo_selfplay.py` needed — 1000 generations may
   simply be far too few for this to reach useful skill.
2. Recurrent NEAT (`neat.nn.RecurrentNetwork`) for temporal memory.
3. Mix self-play with periodic evaluation against `BaselinePolicy` (or
   an archive of past champions, not just the most recent one) to guard
   against the population drifting into a narrow, exploitable style. See
   "why #3 was set aside" below — likely not worth doing before #1.
4. Reward shaping tied to something the candidate actually controls
   (e.g. bonus per successful ball contact/return) layered on top of
   self-play, rather than either alone.
5. Lower `compatibility_threshold` / raise `pop_size` for more sustained
   speciation and genetic diversity.

### Why #3 (mixing in BaselinePolicy) was set aside for now

Discussed before stopping: since every genome tried so far (runs 1-4)
performs statistically identically to a random-action policy against
`BaselinePolicy` (same mean score, same score distribution), the
self-play population hasn't yet crossed whatever skill threshold would
make `BaselinePolicy` give a non-flat fitness signal at all. Adding
`BaselinePolicy` matches into the self-play loop now would very likely
cost extra compute per generation while contributing near-zero
selection pressure (all genomes would still lose to it by about the
same margin, indistinguishably). This tactic looks more useful *after*
#1 (much longer self-play) has produced champions that show at least
some measurable variation against `BaselinePolicy` — at that point it
could help the population generalize instead of overfitting to its own
recent lineage. Doing #1 first is the more promising order.

## Stopping point (2026-09-14)

Four approaches tried in one session (raw single-rollout fitness,
averaged-rollout fitness, survival-bonus reward shaping, and a 172-round
self-play curriculum), all ending at the same wall: 0 wins out of 100
against `BaselinePolicy`, indistinguishable from random play. The
common thread across every run is that `BaselinePolicy` gives no usable
fitness gradient to any policy below some skill threshold that none of
these runs reached, and that 1000 generations / pop_size 128 of
feedforward-only NEAT is a small budget compared to what other methods
in this repo needed (`train_ppo_selfplay.py` alone runs for 1e9
timesteps). Stopping here rather than continuing to iterate; the
"next-step ideas" above are recorded for whoever picks this back up,
roughly in priority order (longer self-play budget first).

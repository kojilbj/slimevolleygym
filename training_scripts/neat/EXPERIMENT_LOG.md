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

## Run 5 — fixing self-play's degenerate equilibrium

Goal for this round, explicitly set with the user: actually beat
`BaselinePolicy` (win rate > 0%), not just show incremental metric
improvement. Before scaling self-play up, the user watched run 4's
`best.pkl` play on screen and noticed it just stood in the middle
jumping, never reacting to the ball. That observation turned out to be
exactly right and led to the real diagnosis below.

### Root cause found by inspection, not just visual observation

Probing run 4's `best.pkl` with synthetic/random observations showed
its 3 outputs were **completely constant regardless of input**
(`[1.0, 0.048, 1.0]` for literally any observation, including random
noise). Tracing the genome's connection graph: the only path to one
output went through a hidden node with zero enabled incoming
connections (an orphan, likely left over from a `mutate_add_node` split
whose incoming half was later deleted); the other two outputs had no
input-facing path at all. So run 4's 172 "dethronings" were really just
an arms race between mutually blind fixed-action strategies — self-play
against a random-action seed never rewards reacting to the ball at all,
so there was no pressure to keep sensory pathways wired up, and NEAT's
default deletion rates pruned them away.

### Three fixes applied together (`train_neat.py`, `logdir = "neat_run5*"`)

1. **`TrackingPolicy`** — a simple scripted opponent (move toward the
   ball's x position, jump when close/low) replaces `RandomPolicy` as
   the self-play archive's seed, so reacting to the ball matters from
   generation 0.
2. **`OpponentArchive`** — keeps the seed plus up to 10 past champions;
   genomes are evaluated against opponents *sampled* from the archive
   rather than only the single most recent one, to reduce overfitting to
   one narrow lineage.
3. **Lower `conn_delete_prob`/`node_delete_prob`** in
   `neat_config_selfplay_v2.txt` (0.5→0.2, 0.2→0.1) so sensory pathways
   survive long enough to matter.

### Iterating in short 100-generation trials before committing to a long run

| trial | config change vs. previous | finding (from direct connection-graph inspection + on-screen play) |
|---|---|---|
| `neat_run5` (v2) | fixes 1-3 above | 2 of 3 outputs still saturated constant; agent walked into the wall, no jump — user: "stuck to the middle wall" |
| `neat_run5b` (v3) | + lower bias volatility: `bias_init_stdev` 1.0→0.5, `bias_mutate_power` 0.5→0.2, `bias_max/min_value` ±30→±5 (neat-python's `sigmoid(5*z)` is steep enough that bias alone easily saturates a node regardless of input) | all 3 outputs numerically reactive to *random* input, but a systematic one-input-at-a-time sweep showed **zero enabled connections from `bx` (ball x) or the agent's own `x`** — the network still couldn't sense left/right at all. User, watching it play: "not reacting to the ball" |
| `neat_run5c` (v4) | `conn_delete_prob` 0.2→0.05, `node_delete_prob` 0.1→0.02 (deletion is ~irreversible; disabling is reversible via `enabled_mutate_rate`, so push nearly all pruning through disabling instead) | `bx` now has an enabled outgoing connection, but it dead-ends at an orphaned hidden node with no further path to any output — connected but not yet functional. Judged as expected mid-flight state for a 100-generation run, not a new bug |

Beat `BaselinePolicy` was the stated goal, so after 3 short trials
showing steady (if incomplete) progress, moved to a real run: **5000
generations, `neat_config_selfplay_v4.txt`, `logdir = "neat_run5_full"`**
(estimated ~10-12h at the measured ~8s/generation).

### The long run revealed a new problem: unconstrained bloat

At generation ~800 (~1.5h in), the population had collapsed to a single
species (`species_elitism=2` means a lone species is always "top 2" and
so is immune to the stagnation-based culling that would normally remove
a non-improving species — discussed with the user and accepted as
non-fatal on its own). More seriously, inspecting the current self-play
champion directly:

```
total nodes: 158   total connections: 507   enabled: 33
105 of 158 nodes have ZERO enabled connections (in or out) -- pure deadweight
NONE of the 12 inputs reach ANY output through an enabled path
```

So the v4 fix (near-zero deletion rates) traded run 4's problem (useful
connections pruned away) for a worse one: with `node_delete_prob=0.02`,
nodes accumulate almost monotonically (158 nodes for a 12-input/3-output
task), and every new node/connection added by mutation lands in an
exponentially larger graph, making it statistically less and less likely
that any given mutation completes a working end-to-end path rather than
wiring together two irrelevant interior nodes. The genome became a much
bigger haystack with the same (or fewer) working needles. **Stopped the
run at generation ~800** rather than let it continue for another ~7
hours in a confirmed-nonfunctional state.

### Conclusion

Deletion rates that are too high (run 4's defaults) lose sensory
pathways; deletion rates that are too low (run 5's v4) cause unbounded,
mostly-non-functional node bloat that dilutes the search space instead.
Neither extreme worked. A real fix likely needs one of:

- A genome-size/complexity penalty in the fitness function (classic NEAT
  "parsimony pressure") so bloat is actively selected against, rather
  than only controlled via the delete-rate knob.
- A cap on `node_add_prob` relative to `conn_add_prob` so new nodes are
  added more slowly than the connections needed to wire them up.
- Re-checking connectivity (not just fitness) periodically during a long
  run — e.g. an automated check like the ones done by hand above,
  logged every N generations — so a regression like this is caught in
  minutes instead of requiring a manual inspection to notice.

Not yet re-attempted; picking `neat_config_selfplay_v4.txt`'s deletion
rates back up somewhat (between v2's 0.2/0.1 and v4's 0.05/0.02) plus
one of the bloat controls above is the likely next move.

## Interim stopping point after run 4 (2026-09-14, superseded)

Four approaches tried up to this point (raw single-rollout fitness,
averaged-rollout fitness, survival-bonus reward shaping, and a 172-round
self-play curriculum) all ended at the same wall: 0 wins out of 100
against `BaselinePolicy`, indistinguishable from random play. The
common thread across every run was that `BaselinePolicy` gives no usable
fitness gradient to any policy below some skill threshold that none of
these runs reached, and that 1000 generations / pop_size 128 of
feedforward-only NEAT is a small budget compared to what other methods
in this repo needed (`train_ppo_selfplay.py` alone runs for 1e9
timesteps).

This was treated as a stopping point at the time, but the user decided
to continue with an explicit goal (actually beat `BaselinePolicy`) —
see run 5 above, which found and partially fixed a much more specific
problem (self-play converging to input-blind fixed actions) before
running into the bloat issue described there. Left in place for the
history; the *current* stopping point is the end of the run 5 section
above.

Stopped run 5's 5000-generation attempt at generation ~800 after
confirming its champion had regressed to zero functional input-to-output
paths (node bloat, not a connectivity-loss problem this time).

## Run 5, v5 — deletion rates between v2 and v4, plus a parsimony penalty

Retuned `conn_delete_prob` 0.05→0.1, `node_delete_prob` 0.02→0.05,
`node_add_prob` 0.2→0.1, and added `complexity_penalty = 0.01` per node
directly in `eval_genomes`'s fitness calculation (`neat_run5d`, 100-gen
trial). Also added `connectivity_report()` (node/connection counts + how
many of the 12 inputs have an enabled path to an output), printed on
every dethroning from then on, so this class of regression shows up in
the log instead of requiring manual inspection each time.

Result: no bloat (nodes stayed 3-7 throughout), but connectivity
oscillated between 2/12 and 7/12 without a clear upward trend — final
`best.pkl` had only 4/12 inputs working. Scaled to a real 5000-gen run
(`neat_run5_full_v5`) anyway; at generation ~800 (1.5h in) it was still
oscillating in the same 2-5/12 range with no bloat, but also no growth,
and the population had collapsed to 1 species (discussed with the user
and accepted as non-fatal, since `species_elitism=2` protecting a lone
species from stagnation-culling doesn't by itself stop fitness
progress). **Stopped by user choice** ("一旦止めて、parsimonyペナルティ
を外してみよう") to try removing the penalty, but the conversation
turned to a more informative comparison first (below) before
re-launching.

## Comparison against this repo's own pretrained self-play model

Before iterating further, checked whether self-play can work in this
environment at all, using ground truth already in the repo:
`zoo/ga_sp/ga.json`, produced by `train_ga_selfplay.py`, loaded via
`slimevolleygym.mlp.makeSlimePolicyLite`. Evaluated over 100 episodes vs
`BaselinePolicy`:

```
mean score: 0.23 (std 0.73)
wins/draws/losses: 29/61/10
```

**This actually beats BaselinePolicy** — confirming self-play itself is
a viable approach here; every NEAT run so far (1-5) had failed only to
match random play, nowhere close to this. The key structural difference:
`train_ga_selfplay.py` uses a **fixed topology** (12→10→10→3 tanh MLP,
273 params, from `slimevolleygym/mlp.py`'s `games['slimevolleylite']`)
and evolves only weights via mutation, plus a very different self-play
protocol — continuous random-pair tournaments across the whole
population (500,000 of them), rather than a single evolving "champion"
gated by a dethroning check.

This reframed the working hypothesis: NEAT's defining "start minimal,
grow topology as needed" approach (`num_hidden=0`) may itself be the
main obstacle — building a working multi-layer pathway one node/
connection at a time via mutation is slow and fragile, which matches
every failure mode seen in runs 4-5 (lost pathways, dead ends, bloat).
The user's constraint: NEAT is required for this project (a from-scratch
NumPy NEAT reimplementation is planned next, see memory), so switching
to plain GA was explicitly rejected — the fixes below stay within NEAT.

## Run 6 — start with real hidden-layer capacity (`num_hidden=10`)

Departs from NEAT's usual "start minimal" practice on purpose: config
`neat_config_selfplay_v6.txt` sets `num_hidden=10`,
`initial_connection=full_nodirect` (input→hidden→output, no direct
input-output edges, matching the layered MLP shape above),
`complexity_penalty` removed (10 baseline hidden nodes are necessary
capacity, not bloat). First attempt crashed immediately —
`compatibility_threshold=3.0` (tuned for the old 36-connection genomes)
was far too tight for the new 150-connection genomes: measured pairwise
genetic distance among a freshly initialized population averaged ~3.6,
so nearly every genome became its own species and reproduction couldn't
satisfy `pop_size >= num_species * min_species_size`. Fixed by raising
`compatibility_threshold` to 6.0 (verified empirically: gen-0 population
formed 1 species at this threshold).

100-generation trial (`neat_run6`) result: **12/12 inputs reached an
output for the first ~13 dethronings** — clearly better than any
previous run's start. But by generation 100 it had collapsed back to
0/12 (final `best.pkl`: 16 nodes, 11 enabled connections, 0/12 working).
`num_hidden=10` alone wasn't enough to hold onto working connectivity
over time.

## Run 7 — broader (multi-opponent) dethroning check

Root-caused run 6's late collapse: `maybe_add_champion` only tested a
challenger against the single most recent champion. A genome could join
the archive by exploiting that one specific opponent's blind spot
without being generally competent — the same failure GA self-play
avoids by having random population pairs play continuously rather than
gating entry into one lineage. Fix: `maybe_add_champion` now samples up
to 4 opponents from across the whole archive and requires beating that
broader sample on average, at the same total rollout budget (4 opponents
× 5 rollouts instead of 1 opponent × 20).

100-gen trial (`neat_run7`) result: ended at **7/12** working inputs
(`bx` and the agent's own `x` both present) — better than run 6's 0/12,
though a direct sweep showed `bx`'s effect on the output was very weak
in the normal input range (only visible at an extreme, unrealistic
value). On screen, the user still couldn't see left-right reaction
("感じられなかったな"), consistent with that measurement. Judged as a
"weights need more time to strengthen an existing pathway" problem
rather than a structural one, and scaled to a 5000-generation run
(`neat_run7_full`).

**The long run regressed anyway.** By generation ~20 connectivity had
already started falling from 10-12/12, and generations 23-68 sat at
0-2/12 while node count climbed steadily to 23-31 with enabled
connections falling to 6-20 — bloat, the exact run-5-v4 failure mode,
just starting from `num_hidden=10`'s higher floor instead of 0. Cause:
`complexity_penalty` had been removed for run 6 (under the reasoning
that the num_hidden=10 baseline wasn't "bloat") and was never
reinstated for run 7's dethroning-check fix, so nothing was left to
resist unbounded growth *beyond* that baseline. **Stopped again** by
user choice ("そうしてみて" — reinstate the parsimony penalty).

## Run 8 — all three fixes combined

`num_hidden=10` (run 6) + broader archive-sampled dethroning (run 7) +
`complexity_penalty=0.01` reinstated (run 5v5), all at once instead of
one at a time. 100-gen trial (`neat_run8`) result: nodes stayed in a
healthy 7-16 range throughout (no bloat), connectivity eased from 12/12
down to 4/12 by generation 18 but never collapsed to 0 — the best
outcome of any 100-gen trial so far, balancing all three previous
failure modes reasonably well. Scaled to a 5000-generation run
(`neat_run8_full`), currently in progress.

## Run 8, full 5000-gen attempt — bloat control worked, but pruned bx/x entirely

`neat_run8_full` avoided bloat (nodes settled to a tiny 3-7 throughout,
no runaway growth) but by generation ~190 the parsimony penalty had
pruned away **both** `bx` and the agent's own `x` completely — the
network could no longer sense horizontal position at all. Confirmed on
screen: user watching `champion_0192.pkl` play reported "ずっと真ん中の
壁に張り付いてる" (stuck against the middle wall the whole time), which
a direct connectivity check matched exactly (`working inputs: ['by',
'vx', 'vy']` — no `bx`, no `x`). **Stopped.**

## Run 9 — repair protected inputs directly instead of hoping they survive

Rather than continue re-tuning delete/penalty rates and hoping `bx`/`x`
happen to survive, added `repair_protected_inputs()`: every generation,
for each of `bx` and the agent's own `x`, if no enabled path to *any*
output exists, force-create/re-enable a direct connection to output 0.
Verified in isolation first (unit test: delete `bx`/`x` connections from
a genome, call the repair, confirm they're back) before running.

100-gen trial (`neat_run9`) looked great by the earlier connectivity
metric and even showed a clean bx sweep (`action` correctly flips
forward/backward around the ball's position). But tracking the
candidate's actual x position over a real rollout told a different
story: **509 of 600 steps (85%) were spent within 1 unit of the net** —
it drove straight to the net and got stuck, exactly matching what the
user reported ("ずっと真ん中の壁に張り付いてる...一度もボールに触ってない").
Diagnosis: the repair guaranteed *some* path to *an* output, but nothing
required *both* forward (output 0) and backward (output 1) to be
reachable — the genome's `backward` output turned out to be constant
regardless of `bx`, so the agent could approach the net but never
retreat.

## Run 9b — protect both forward and backward outputs specifically

Generalized `has_path_to_output` into `has_path_to_node(start, target,
...)` and changed `repair_protected_inputs` to require each protected
input reach **both** output 0 (forward) and output 1 (backward)
individually, not just "an" output. Verified with a unit test again.

100-gen trial (`neat_run9b`) fixed the net-freeze — the bx sweep now
correctly flipped forward/backward — but the champion instead froze at
the *opposite* wall (`x` from -12.58 to -22.5, then stuck; 0/600 steps
near the net). Tracing the actual observations during play revealed the
real bug: the network was firing **forward=1 AND backward=1
simultaneously**. `Agent.setAction` (`slimevolley.py:383-386`) treats
that combination as "stand still" (`desired_vx` stays at its default of
0 unless exactly one of forward/backward is true) — so two
independently-guaranteed-reachable outputs could both fire at once and
silently cancel each other out. Guaranteeing reachability wasn't the
same as guaranteeing mutually-exclusive activation.

## Fix: break forward/backward ties in `NeatPolicy.predict()`

Rather than trying to force opposite-signed weights during training
(which wouldn't stop *other*, non-protected connections from also
pushing both outputs high), fixed this at the policy layer in
`slimevolleygym/neat_policy.py`: when both `forward` and `backward`
threshold past 0.5, keep whichever raw sigmoid output is larger instead
of letting the game interpret both as "stand still". This fixes the
issue for *any* genome, not just ones built by the training-time repair.
Verified immediately against run 9b's already-frozen `best.pkl` *without
retraining* — same genome, x now moved between -22.5 and -2.0 instead of
being stuck at one wall.

## Run 10 — retrain with the tie-break fix

100-gen trial (`neat_run10`) result: fitness spiked to 3.60 at one point
(highest ever), and the final `best.pkl` genuinely moved back and forth
(x ranged -11.4 to -3.8, no permanent freeze) with corr(agent_x,
ball_x) = -0.76, but **zero ball touches** in a 600-step rollout, and
the user described the motion as "振動してるだけ" (just oscillating) once
watched on screen — plausibly driven by other connected inputs (by,
vx, vy) rather than genuine `bx`-based tracking. Judged as a
structurally-sound-but-still-unskilled network (a normal "needs more
training time" problem) rather than a new structural bug, since every
previously-discovered failure mode (lost pathways, bloat, freezing) was
now absent.

## Run 11 — scaled to 5000 generations, with per-100-generation check-ins

Same `train_neat.py` setup as run 10 (`neat_config_selfplay_v6.txt`,
`complexity_penalty=0.01`, multi-opponent dethroning, protected-input
repair for both forward/backward, tie-break fix in `NeatPolicy`),
launched as `neat_run11_full`. Added `training_scripts/analyze_champion.py`
as a reusable tool for periodic checkpoints: connectivity report + an
actual rollout vs `TrackingPolicy` with ball-touch count and
agent-x/ball-x correlation, with `--render` to watch on screen. A
background monitor emits an event every ~100 generations so each
milestone's latest champion gets checked without manual polling.

### Two more bugs found and fixed in `analyze_champion.py` itself (not training)

1. **Reward sign confusion.** The script initially placed the candidate
   as `policy_left` (matching training's convention) and printed the
   raw `env.step()` reward, which is documented and coded
   (`slimevolley.py:583`, `:588`) as *right-side* perspective. A
   negative value therefore meant the *left* side (the candidate) won,
   not lost — misread out loud as the opposite once ("TrackingPolicy側
   の勝ち"), caught by the user. Fixed properly by swapping the
   candidate onto `policy_right` (the env's own default "self" side)
   so the printed score is directly the candidate's own perspective,
   no mental sign-flip needed. Verified the training code itself
   (`eval_genomes`, `maybe_add_champion` in `train_neat.py`) was never
   affected — both already correctly negate the right-side `rollout()`
   score to get the *left-side* candidate's perspective there, since
   training always plays the candidate on the left against the archive.
2. **Ball-touch undercounting.** Checking `ball.isColliding(agent_right)`
   *after* `env.step()` returns is too late — the internal collision
   check and bounce already happened inside `Game.step()`, so the ball
   has already moved away by the time the script checks (reported 0
   touches when the user counted 10+ on screen). Fixed by wrapping
   `ball.bounce()` directly, which is called the instant a collision is
   detected. That undercounted too (2 instead of 10+) because
   `Game.newMatch()` (called on every scored point) replaces
   `self.ball` with a brand-new `Particle`, silently dropping the
   wrapped method — fixed by also wrapping `newMatch()` to re-wrap the
   fresh ball's `bounce()` each time. Final count (5) matched the user's
   own on-screen count exactly.

### First genuinely competent-looking result

At generation ~280 (35th dethroning), `champion_0035.pkl`: **16 ball
touches**, score +5 (candidate perspective, candidate on the right),
corr(agent_x, ball_x) = 0.889, over an 829-step rollout vs
`TrackingPolicy`. Confirmed on screen — first time in 11 runs the
candidate visibly and repeatedly tracks and returns the ball rather than
freezing, oscillating aimlessly, or winning only via the opponent's own
mistakes. User: "ブラボーめっちゃいいじゃない".

champion_0035 was also tried against two more opponents, on screen:

- **champion_0034** (its own immediate self-play predecessor): champion
  0035 won +4 over 1516 steps, both genomes visibly rallying the ball
  back and forth. User: "動いてる、両方ともちゃんと打ち返してる。感動".
- **`BaselinePolicy`** (the actual stated goal): ran the full 3000-step
  limit, candidate lost by just **-1**, with **72 ball touches** — by far
  the longest, most competitive match of the whole project (runs 1-4
  never touched the ball at all against `BaselinePolicy`).
- **`zoo/ga_sp/ga.json`** (this repo's own GA self-play model, used
  earlier as external validation that self-play works here at all):
  candidate lost -5 over 2282 steps but with 36 ball touches — a real
  rally, just not yet winning. User: "動きは悪くないけど、反応が少し遅い
  のかもね".

Given champion_0035 was now clearly in the right regime (competitive
with `BaselinePolicy`, not just with its own weak self-play seed), asked
whether to keep pure self-play going or start mixing in stronger fixed
opponents. User confirmed external opponents can never become a
"champion" themselves (they have no genome) — they only enrich the
opponent pool used for fitness/dethroning; only NEAT genomes evolved by
the population get crowned.

## Run 12 — mixed opponent pool, seeded from champion_0035

Copied `train_neat.py` to `train_neat_run12.py` (kept as a separate file
specifically so run 11 could keep running unaffected on the original
script — editing a `.py` file on disk does not affect an already-running
process, but keeping them as distinct files avoids any confusion about
which script produced which run). Two changes from run 11:

1. `OpponentArchive` split into `permanent` (never evicted: the
   `TrackingPolicy` seed + this repo's zoo models) and `rotating` (self-
   play champions, capped as before). External models added, each
   pre-checked vs `BaselinePolicy` over 20 episodes: `cmaes_sp` (mean
   -0.15, slightly weaker than baseline), `ga_sp` (+0.23), `cmaes`
   (+1.00, solidly stronger), plus `BaselinePolicy` itself.
2. `seed_population()` initializes generation 0 from 128 mutated copies
   of `champion_0035` (via `reproduction.genome_indexer` for fresh keys
   + `genome.mutate()`), instead of NEAT's usual fresh-random-genomes
   start, via `population.population = {...}` followed by
   `population.species.speciate(...)`. The point was "given a genome
   that can already rally, does training against tougher opponents push
   it further" — not re-deriving basic competence from scratch.

### Result: regression, not improvement

100-generation trial (`neat_run12`): **zero dethronings** — no
challenger ever beat a sampled set of (mostly permanent-pool) opponents
by the same `dethrone_margin=0.5` that self-play alone cleared 13-35
times per 100 generations. Population fitness stayed solidly negative
throughout (best usually 1-3, mean around -3.2). More importantly, the
population's actual best genome after 100 generations
(`neat_run12/best.pkl`) had **regressed sharply from the champion_0035
seed**: 1 ball touch and corr(agent_x, ball_x) = -0.191 vs
`TrackingPolicy`, compared to champion_0035's 16 touches and corr 0.889
under the same test. Mixing in tough fixed opponents (particularly
`cmaes` at +1.00 vs `BaselinePolicy`) from the very first generation
made the fitness landscape uniformly harsh enough that useful mutations
couldn't be distinguished from harmful ones within 100 generations —
the opposite of the intended "harder curriculum" effect. **Paused** by
user choice; not scaled to a full run.

Run 12's mixed-opponent-pool direction is shelved for now (candidate
follow-up if revisited: ease in gradually -- e.g. drop `cmaes` initially
and/or lower `dethrone_margin` -- rather than the full harsh pool from
generation 0).

## Diagnosing run 11's real problem: declining fitness, not a strong champion

By generation ~700, `neat_run11_full` still hadn't dethroned
champion_0035 (from ~gen 280) after 400+ generations. Sampling "Best
fitness" every 50 generations across the whole run showed a clear
**downward trend**, not a plateau: early values 2.92/3.30/2.64/2.28/3.64
vs. later values -0.70/0.30/-1.03/-0.03/1.29/-0.03/-1.36/0.95/-0.74/-1.70.
A genuinely strong, stable champion would produce a plateau; a declining
trend across hundreds of generations means the population itself was
getting *worse* over time -- classic genetic drift. Root cause:
`species_elitism=2` makes a lone species immune to stagnation-culling
(discussed and accepted as "not fatal on its own" back in run 4/8), but
over 467+ generations in a single species, real harm compounds: no niche
protection for novel structural experiments, and no cross-species
comparison to catch quality decline. Measured the live population's
actual pairwise genetic distance directly from `checkpoint-770`: mean
1.43, max 2.30 -- the `compatibility_threshold=6.0` in place (calibrated
for run 6's freshly-initialized, much less differentiated population)
could never have split this population into multiple species.

## Run 13 — resume with a recalibrated speciation threshold

Rather than restart from scratch, resumed run 11's actual
`checkpoint-770` (the real population, just undiversified) via
`neat.Checkpointer.restore_checkpoint(path, new_config=...)`, using
`neat_config_selfplay_v7.txt` (`compatibility_threshold` 6.0 -> 1.6,
empirically retested against that same checkpoint: 1.2 -> 30 species/too
fragmented, 1.6 -> 9 species sizes 2-42/reasonable, 1.8 -> 6, 2.0 -> 2,
2.2 -> 1). Also reconstructed the self-play archive as it would have
looked at that point by loading run 11's actual last 9 champion files
(`OpponentArchive.resume_from()`) instead of restarting the archive from
just the `TrackingPolicy` seed.

**First attempt still produced only 1 species after "re-speciating".**
Cause: `restore_checkpoint` brings back the OLD `species` object intact,
and NEAT's `speciate()` only compares new genomes against *existing*
species' representative genomes, opening a new species only when a
genome matches none of them -- reusing that stale single-species object
(with its one old representative) meant most genomes still nominally
"matched" it even under the new threshold, unlike a from-scratch
clustering. Fixed by replacing `population.species` with a brand-new,
empty `neat.DefaultSpeciesSet` right before calling `.speciate()` --
exactly mirroring how the threshold was calibrated/tested beforehand.
This produced the expected 9 species immediately.

`neat_run13_full` ran for ~1250 generations (770 -> 2021) without
declining, unlike run 11 -- confirmed healthy: "Best fitness" sampled
every 40 generations stayed in a positive, non-trending range (roughly
-1.05 to 3.6), with the best genome coming from a constantly-changing
mix of species numbers rather than one dominant lineage. Champion_0038
(promoted early in this stretch, ~generation 902) still hadn't been
dethroned after 1000+ generations, though -- worth checking why.

## Diagnosing the champion_0038 plateau: still measurement noise

Pulled the actual best genome from a late checkpoint (`checkpoint-1971`,
generation-local fitness 1.94 -- looked close to competitive) and
re-tested it under the *exact* dethroning-check conditions (4 sampled
archive opponents x 5 rollouts = 20 total, same as
`maybe_add_champion`): scored **-3.9**, nowhere close to champion_0038.
The per-generation fitness NEAT itself selects on is still averaged over
only `n_rollouts=3` -- the same noise problem diagnosed all the way back
in runs 1-2, which never fully went away once self-play made
per-opponent variance an *additional* source of noise on top of
per-episode variance. On-screen, a mirror match (champion_0038 vs.
itself) went the full 3000-step limit at a 1-point margin -- about as
close to a coin flip as this environment's asymmetries allow, which is
a reasonable sanity check that the genome itself isn't degenerate, just
that 3-rollout fitness isn't discriminating enough anymore.

## Run 14 — reduce fitness noise, and a neat-python resume bug

Resumed run 13 from its generation-2021 checkpoint (again not from
scratch) with `n_rollouts` raised 3 -> 6, and the self-play archive
reconstructed across *both* run 11's and run 13's champion files (they
went to different `logdir`s: run 11's champions 1-35 are in
`neat_run11_full/`, run 13's 36-38 are in `neat_run13_full/` --
`OpponentArchive.resume_from()` generalized to search a list of
directories, sorted together since the run-number ordering happens to
sort correctly lexicographically here: "neat_run11_full" < "neat_run13_full").

**First attempt crashed 3 generations in** with `AssertionError` in
neat-python's `get_new_node_key` (`neat/genome.py:122`): a structural
mutation tried to assign a new hidden-node ID that already existed.
Root cause: `config.genome_config.node_indexer` (the counter that hands
out new node IDs) is a plain Python `count()` on the `GenomeConfig`
object -- it is **not** part of the innovation tracker that
`restore_checkpoint` explicitly preserves, and it lazily initializes
from `max(node_dict) + 1` of whichever genome happens to be mutated
*first* after a resume, not the true maximum across the whole
population. After chaining two resumes (run 11 -> 13 -> 14), the
genome mutated first apparently had a lower max node ID than others
elsewhere in the population, so the indexer started too low and
collided with an ID already in use elsewhere. Fixed by explicitly
setting `config.genome_config.node_indexer =
itertools.count(max_node_id_across_whole_population + 1)` right after
restoring the checkpoint and before calling `population.run()`. Worth
remembering for any future resume, not just chained ones -- this could
in principle happen on a *first* resume too, given the right population
state; run 13 simply didn't hit it by luck.

`neat_run14_full` ran for ~470 generations (2021 -> 2500ish) with fitness
staying in a healthy, non-declining range. It produced one new champion,
**champion_0039**, which passed the 20-rollout dethroning score test —
but on screen, both in a mirror-ish match and directly against
`BaselinePolicy`, it was clearly worse than champion_0035/0038 (1 ball
touch, lost -5 to `BaselinePolicy` in 568 steps). User: "泥試合でした" →
"どっちも弱かったんだよ" → **"ずっと右の壁に張り付いてる...何もしてないんだよ"**.

## Diagnosing champion_0039: passed the score test while stuck at a wall

Swept `bx` and the agent's own `x` through champion_0039's network
directly: `backward` (retreat toward its own wall) fired for nearly
every realistic `bx` value (only the extreme `bx=-2` produced
`forward`), and for its own `x` only very negative values (already past
the net) triggered `forward` -- in practice, from any normal starting
position, it just walks to its own back wall and sits there. This is
not the same bug as run 9b's simultaneous forward+backward firing (fixed
in `NeatPolicy`, and confirmed still working here) -- it's a plain
weight-bias problem: this genome's horizontal decision is dominated by
a near-constant backward bias regardless of the ball, and it happened to
be *good enough* against the specific 4 archive opponents sampled during
its dethroning check to average past `dethrone_margin=0.5` anyway.
Comparing it against champion_0035 on screen (972 steps, champion_0035
won) initially looked like an interesting non-monotonicity in self-play
skill, but the user correctly reframed it: champion_0039 isn't "losing a
competitive match", it's simply not functioning, so of course a
genuinely-playing opponent beats it.

## Run 15 — reject degenerate (non-moving) challengers directly

User's request: add a check for "is it actually moving" to the
dethroning process itself, not just a score threshold. Added
`measure_movement_std()`: roll the challenger out once against a sampled
archive opponent and compute the standard deviation of its own x
position; below `min_movement_std=3.0`, reject immediately without even
running the expensive 20-rollout score test (cheap gate first, saving
compute on obviously-broken challengers).

First version used **range** (max-min) instead of stdev and was
immediately fooled by champion_0039 itself: it scored `x_range=9.92` --
comfortably above a naive threshold -- because it moves for its first
~20 of 568 steps before freezing at one wall for the rest, which still
produces a decent-looking range. Switched to standard deviation, which
is dominated by how the position is distributed over *time*, not just
its extremes: champion_0039 measured **std=1.00-2.32** across several
challengers rejected in the first 9 generations of run 15, clearly
separated from champions 0035/0038's **std=5.7-6.6** under the same
test. Verified directly against known genomes before relying on it.

Resumed run 14 from its generation-2522 checkpoint (with the by-now
routine re-speciation + `node_indexer` fixes from run 13/14 also
applied) with this new gate in place. Within the first 9 generations, 6
challengers were rejected for being stuck at a wall — a strong signal
that this failure mode (a genome exploiting the current archive without
actually playing) is common, not a one-off, and this check is pulling
real weight.

## Current status (2026-09-15)

`neat_run15_full` is running (resumed from run 14's generation 2522,
same config, `n_rollouts=6`, movement-std gate added to
`maybe_add_champion`, archive reloaded with the last 9 of 39 real
champions across all three prior runs' logdirs), targeting generation
5000 total (`n_generations=2478` more). Automatic ~100-generation
milestone check-ins continue via a background monitor (now also
reporting cumulative rejection count). Not yet evaluated against
`BaselinePolicy` with a full 100-episode `eval_neat.py` readout (only
single on-screen matches so far) — that remains the next step once
run 15 finishes, produces a champion that clears the new movement gate
by a comfortable margin, or plateaus clearly.

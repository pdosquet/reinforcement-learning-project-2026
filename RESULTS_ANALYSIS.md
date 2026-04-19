# Results Analysis

Track of all training runs, grouped by experiment batch.

---

## Experiment Groups

### PPO Hover First Sweep (2026-04-15)

**Goal:** Establish PPO baselines on hover and compare gamma × n_steps.

**Config:** mode=0, seeds=[0, 1], timesteps=1e5

| Config | gamma | n_steps | W&B group | Zoo runs |
|--------|-------|---------|-----------|----------|
| `ppo_hover_g098_ns2048_100k_mode0` | 0.98 | 2048 | `ppo_hover_g098_ns2048_100k_mode0` | _4, _5 |
| `ppo_hover_g098_ns4096_100k_mode0` | 0.98 | 4096 | `ppo_hover_g098_ns4096_100k_mode0` | _6, _7 |
| `ppo_hover_g099_ns2048_100k_mode0` | 0.99 | 2048 | `ppo_hover_g099_ns2048_100k_mode0` | _8, _9 |
| `ppo_hover_g099_ns4096_100k_mode0` | 0.99 | 4096 | `ppo_hover_g099_ns4096_100k_mode0` | _10, _11 |

**Notes:**
- Zoo runs _1–3 are aborted early attempts (before run naming was fixed), ignore them.
- 1e5 steps is likely too short for PPO to converge — treat as a pilot.

**Results:**

| Config | seed 0 | seed 1 | verdict |
|--------|--------|--------|---------|
| `ppo_hover_g098_ns2048` | ✅ learned | ✅ learned | best |
| `ppo_hover_g098_ns4096` | ❌ no convergence | ❌ no convergence | too few update cycles |
| `ppo_hover_g099_ns2048` | ✅ learned | ⚠️ slower | mixed |
| `ppo_hover_g099_ns4096` | ❌ no convergence | ❌ no convergence | too few update cycles |

**Conclusion:**
At 100k steps, `ns=4096` gets only ~24 rollout cycles vs ~48 for `ns=2048` — not enough update cycles to converge within the budget. `ns=4096` is not fundamentally worse, it just needs proportionally more timesteps. **gamma had no meaningful effect** at this scale: the discount horizon difference between 0.98 and 0.99 is too small to be visible within 100k steps. The only axis that mattered was `n_steps`.

**Next steps to consider:**
- Extend `ns=2048` runs further to confirm policy quality
- Try the best config (`ns=2048`) across other flight modes — gamma can be fixed to either value

---

### PPO Hover Second Sweep mode 0, ns=2048, 500k (2026-04-16)

**Goal:** Confirm policy quality at 500k for the best ns=2048 configs; compare gamma=0.98 vs 0.99 with more budget.

**Config:** mode=0, seeds=[0, 1], timesteps=5e5, n_steps=2048

| Config | gamma | n_steps | W&B group | Zoo runs |
|--------|-------|---------|-----------|----------|
| `ppo_hover_g098_ns2048_500k_mode0` | 0.98 | 2048 | `ppo_hover_g098_ns2048_500k_mode0` | — |
| `ppo_hover_g099_ns2048_500k_mode0` | 0.99 | 2048 | `ppo_hover_g099_ns2048_500k_mode0` | — |

**Notes:**
- ns=4096 was excluded — Group 1 showed it needs more update cycles and ns=2048 already produces a working policy.

**Results:**

| Config | mean reward | notes |
|--------|-------------|-------|
| `ppo_hover_g098_ns2048_500k_mode0` | ~1100 (plateau) | ep_rew_mean dips lower early on both seeds (one seed dips deeper), then catches up and reaches plateau **faster** than 0.99 |
| `ppo_hover_g099_ns2048_500k_mode0` | ~1100 (plateau) | higher ep_rew_mean early on, but converges to plateau more slowly |

**Conclusion:**
Both gammas converge to a plateau around 1100. gamma=0.98 exhibits a lower ep_rew_mean at the start of training but ultimately reaches the plateau faster than gamma=0.99. The early advantage of 0.99 disappears by the end. Either config yields a working hover policy; **gamma=0.98 is marginally preferred** for its faster convergence to peak performance.

---

### PPO Hover First Sweep mode 6 (2026-04-16)

**Goal:** Establish PPO baselines on hover mode 6 and compare gamma × n_steps at 100k.

**Config:** mode=6, seeds=[0, 1], timesteps=1e5

| Config | gamma | n_steps | W&B group | Zoo runs |
|--------|-------|---------|-----------|----------|
| `ppo_hover_g098_ns2048_100k_mode6` | 0.98 | 2048 | `ppo_hover_g098_ns2048_100k_mode6` | — |
| `ppo_hover_g098_ns4096_100k_mode6` | 0.98 | 4096 | `ppo_hover_g098_ns4096_100k_mode6` | — |
| `ppo_hover_g099_ns2048_100k_mode6` | 0.99 | 2048 | `ppo_hover_g099_ns2048_100k_mode6` | — |
| `ppo_hover_g099_ns4096_100k_mode6` | 0.99 | 4096 | `ppo_hover_g099_ns4096_100k_mode6` | — |

**Results:**

| Config | mean reward | ep_rew_mean | verdict |
|--------|-------------|-------------|---------|
| `ppo_hover_g098_ns2048_100k_mode6` | collapses at end | slow (steps) | ❌ unstable |
| `ppo_hover_g098_ns4096_100k_mode6` | stable | reaches plateau faster (steps) | ✅ |
| `ppo_hover_g099_ns2048_100k_mode6` | stable | slow (steps) | ✅ |
| `ppo_hover_g099_ns4096_100k_mode6` | stable | reaches plateau faster (steps) | ✅ |

**Conclusion:**
Mode 6 behaves differently from mode 0 at 100k steps: **ns=4096 reaches ep_rew_mean plateau faster in terms of steps**, while ns=2048 gets there more slowly. Final mean_reward is comparable across all configs except `g098_ns2048` which collapses near the end — gamma=0.98 is too short-sighted when combined with small rollout windows on mode 6. ns=4096 is more **step-efficient** for mode 6; gamma=0.99 is the safer choice.

---

### PPO Hover Second Sweep mode 6, 500k (2026-04-16)

**Goal:** Extend mode 6 configs to 500k to confirm convergence and compare gamma × n_steps.

**Config:** mode=6, seeds=[0, 1], timesteps=5e5

| Config | gamma | n_steps | W&B group | Zoo runs |
|--------|-------|---------|-----------|----------|
| `ppo_hover_g098_ns2048_500k_mode6` | 0.98 | 2048 | `ppo_hover_g098_ns2048_500k_mode6` | — |
| `ppo_hover_g098_ns4096_500k_mode6` | 0.98 | 4096 | `ppo_hover_g098_ns4096_500k_mode6` | — |
| `ppo_hover_g099_ns2048_500k_mode6` | 0.99 | 2048 | `ppo_hover_g099_ns2048_500k_mode6` | — |
| `ppo_hover_g099_ns4096_500k_mode6` | 0.99 | 4096 | `ppo_hover_g099_ns4096_500k_mode6` | — |

**Notes:**
- mean_reward curves are noisy (only 5 eval episodes); ep_rew_mean is the reliable signal here.
- eval-episodes bumped to 20 for future runs.

**Results (ep_rew_mean):**

| Config | plateau (~1100) | convergence speed |
|--------|-----------------|-------------------|
| `g098_ns4096` | ✅ | fastest |
| `g099_ns4096` | ✅ | intermediate |
| `g098_ns2048` | ✅ | intermediate |
| `g099_ns2048` | ✅ | slowest |

**Conclusion:**
All four configs converge to the same plateau (~1100 ep_rew_mean). ns=4096 with gamma=0.98 is the most step-efficient. gamma=0.99 with ns=2048 is the slowest to converge. Unlike mode 0 where gamma=0.98 caused instability with ns=2048, here all configs are stable at 500k — the 100k collapse of g098_ns2048 does not reappear with the larger budget.

---

### SAC Hover First Sweep mode 0 (2026-04-19)

**Goal:** Establish SAC baselines on hover mode 0, compare gamma × UTD.

**Config:** mode=0, seeds=[0, 1], timesteps=1e5

| Config | gamma | UTD | W&B group | Zoo runs |
|--------|-------|-----|-----------|----------|
| `sac_hover_g098_utd3_mode0` | 0.98 | 3 | `sac_hover_g098_utd3_mode0` | — |
| `sac_hover_g098_utd4_mode0` | 0.98 | 4 | `sac_hover_g098_utd4_mode0` | — |
| `sac_hover_g099_utd3_mode0` | 0.99 | 3 | `sac_hover_g099_utd3_mode0` | — |
| `sac_hover_g099_utd4_mode0` | 0.99 | 4 | `sac_hover_g099_utd4_mode0` | — |

**Results:**

| Config | plateau | notes |
|--------|---------|-------|
| `g098_utd3` | ~1100 ✅ | converged |
| `g098_utd4` | ~1100 ✅ | converged |
| `g099_utd3` | not reached | needs more steps |
| `g099_utd4` | not reached | needs more steps |

**Conclusion:**
gamma=0.98 converges clearly to ~1100 within 100k steps regardless of UTD (3 or 4 — no meaningful difference). gamma=0.99 has not converged within the 100k budget and shows unstable `ent_coef_loss` with large spikes — suggesting the entropy tuning is struggling, likely because the longer discount horizon makes the value landscape harder to learn within 100k steps. UTD ratio is not a differentiating factor at this scale for either gamma.

---

### SAC Hover First Sweep mode 6 (2026-04-19)

**Goal:** Establish SAC baselines on hover mode 6, compare gamma × UTD.

**Config:** mode=6, seeds=[0, 1], timesteps=1e5

| Config | gamma | UTD | W&B group | Zoo runs |
|--------|-------|-----|-----------|----------|
| `sac_hover_g098_utd3_mode6` | 0.98 | 3 | `sac_hover_g098_utd3_mode6` | — |
| `sac_hover_g098_utd4_mode6` | 0.98 | 4 | `sac_hover_g098_utd4_mode6` | — |
| `sac_hover_g099_utd3_mode6` | 0.99 | 3 | `sac_hover_g099_utd3_mode6` | — |
| `sac_hover_g099_utd4_mode6` | 0.99 | 4 | `sac_hover_g099_utd4_mode6` | — |

**Results:**

| Config | plateau | convergence speed | actor loss plateau |
|--------|---------|-------------------|--------------------|
| `g098_utd3` | ✅ | slower | higher |
| `g098_utd4` | ✅ | slower | higher |
| `g099_utd3` | ✅ | faster | lower |
| `g099_utd4` | ✅ | faster | lower |

**Conclusion:**
All configs converge to the same reward plateau within 100k steps — unlike mode 0 where gamma=0.99 struggled. gamma=0.99 reaches the plateau faster and settles at a lower actor loss, suggesting the longer discount horizon produces a more stable and confident policy on mode 6. UTD (3 vs 4) has no meaningful impact. **gamma=0.99 is the preferred choice for mode 6 SAC.**

---

### SAC Waypoints First Sweep — mode 6, 100k (2026-04-19)

**Goal:** First look at SAC on the waypoints task; compare UTD 2 vs 3 and two seeds, both at gamma=0.99 mode 6.

**Config:** mode=6, timesteps=1e5, gamma=0.99, eval-episodes=20

| Config | UTD | seed | Zoo run |
|--------|-----|------|---------|
| `sac_waypoints_g099_utd2_mode6` | 2 | 0 | `PyFlyt-QuadX-Waypoints-v4_1` |
| `sac_waypoints_g099_utd2_mode6` | 2 | 1 | `PyFlyt-QuadX-Waypoints-v4_2` |
| `sac_waypoints_g099_utd3_mode6` | 3 | 0 | `PyFlyt-QuadX-Waypoints-v4_3` |

**Numerical results — training curve (monitor.csv):**

| Run | overall mean ± std | reward range | best 10k window |
|-----|--------------------|--------------|-----------------|
| UTD2 seed0 | -317.6 ± 145.9 | [-626, -84] | -211.8 (50k) |
| UTD2 seed1 | -364.8 ± 144.7 | [-665, -20] | -263.2 (30k) |
| UTD3 seed0 | -408.7 ± 156.3 | [-720, -102] | -297.3 (70k) |

Training ep_rew_mean per 10k window — all runs plateau in the **-300 to -450** range with no monotonic improvement:

| 10k step | UTD2 s0 | UTD2 s1 | UTD3 s0 |
|----------|---------|---------|---------|
| 10k | -604.9 | -660.8 | -603.1 |
| 20k | -313.8 | -316.6 | -360.9 |
| 30k | -297.2 | -263.2 | -369.3 |
| 40k | -274.7 | -374.3 | -320.3 |
| 50k | -211.8 | -328.6 | -457.8 |
| 60k | -321.1 | -433.6 | -486.0 |
| 70k | -338.0 | -401.8 | -297.3 |
| 80k | -463.9 | -294.2 | -495.5 |
| 90k | -352.2 | -455.0 | -422.8 |
| 100k | -318.5 | -279.5 | — |

**Eval rewards (evaluations.npz, 20-episode deterministic):**

| Run | final mean ± std | best eval mean |
|-----|-----------------|----------------|
| UTD2 seed0 | -97.7 ± 2.5 | -82.8 (20k) |
| UTD2 seed1 | -91.8 ± 22.1 | -86.5 (40k) |
| UTD3 seed0 | -98.2 ± 3.0 | -88.0 (10k) |

Note the large gap between training rewards (~-300 to -450) and eval rewards (~-90): the deterministic best-model policy avoids crashing but the training curve noise reveals the policy is not actually navigating to waypoints.

**Interpretation:**

1. **Crash avoidance only.** The initial reward (~-600) corresponds to the drone crashing within the first ~20-30 steps, accumulating large step-wise penalties. By 20k steps the agent learns to stay airborne, which cuts the penalty in half. After that: **total plateau**. The drone flies away from the starting position without crashing but never reaches a waypoint.

2. **Sparse reward problem.** The waypoints task provides no dense guidance — the reward signal only triggers when the drone comes within `goal_reach_distance=4.0m` of a target. With random or early policies in a `flight_dome_size=150m` arena, the probability of accidentally reaching a waypoint in 100k steps is near zero. The agent cannot bootstrap from the sparse signal.

3. **No progress signal from UTD.** Higher update-to-data ratio (UTD=3) does not help: UTD3 seed0 actually trends slightly worse (~-408 mean) than UTD2 (~-318 and -364). More gradient steps per transition amplifies noise when there are no informative reward signals to exploit.

4. **Reward variance.** The large std (145-156) reflects the bimodal nature of episodes: either the drone crashes early (reward ~-600) or survives and drifts (-100 to -200). There is no middle ground corresponding to reaching waypoints.

**Conclusion:**

100k steps is **fundamentally insufficient** for the waypoints task. The agent learns a single non-trivial behavior (crash avoidance) but hits a hard ceiling because the sparse waypoint reward is never triggered. There is no learning signal to guide navigation. UTD ratio is irrelevant in this regime. **The minimum viable budget is estimated at 500k–1M steps**, with a denser reward shaping or curriculum (e.g. smaller dome, larger goal_reach_distance as a warm-up, or potential-based shaping) as an alternative path.

**Next steps:**
- Rerun `sac_waypoints_g099_utd2_mode6` at 1M steps (single seed first) to check if the sparse reward is eventually discovered
- Consider reward shaping: add a dense component proportional to negative distance to the current waypoint
- Consider curriculum: start with `flight_dome_size=30`, `goal_reach_distance=8`, gradually increase difficulty

---

## SAC Reference

| Config | task | gamma | UTD | mode | seeds | Zoo runs | mean reward |
|--------|------|-------|-----|------|-------|----------|-------------|
| `sac_hover_g098_utd3` | hover | 0.98 | 3 | 0 | 0 | sac/_1 (2k step smoke test) | — |

---

## Template for new groups

```
### Group N — <description> (YYYY-MM-DD)

**Goal:**

**Config:** mode=, seeds=[], timesteps=

| Config | key param | W&B group | Zoo runs |
|--------|-----------|-----------|----------|
| | | | |

**Notes:**

**Results:**

| Config | mean reward (seed 0) | mean reward (seed 1) | notes |
|--------|---------------------|---------------------|-------|
| | | | |
```

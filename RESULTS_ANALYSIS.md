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

### PPO Waypoints Sweep — mode 6, shaping ablation (2026-04-19)

**Goal:** Establish PPO baselines on waypoints with reward shaping; ablate `ent_coef` and `shaping_coef`.

**Config:** mode=6, seeds=[0, 1], gamma=0.99, n_steps=2048, `WaypointDistanceShaping` active on all runs

| Config | ent_coef | shaping_coef | timesteps | Zoo runs |
|--------|----------|--------------|-----------|----------|
| `ppo_waypoints_mode6` | 0.01 | 0.01 | 100k | _1, _2 |
| `ppo_waypoints_mode6` | 0.01 | 0.01 | 500k | _3, _4 |
| `ppo_waypoints_ent005_shaping005_mode6` | 0.05 | 0.05 | 500k | _5, _6 |
| `ppo_waypoints_ent001_shaping02_mode6` | 0.01 | 0.20 | 500k | _7, _8 |

**Eval results (deterministic, 20 episodes):**

| Zoo run | config | steps | final eval | best eval | final ep_len |
|---------|--------|-------|------------|-----------|--------------|
| _1 | baseline seed0 | 100k | -92.1 | -81.5 | 42 |
| _2 | baseline seed1 | 100k | -97.0 | -96.5 | 18 |
| _3 | baseline seed0 | 500k | -91.3 | -80.5 | 20 |
| _4 | baseline seed1 | 500k | -94.5 | -76.8 | 38 |
| _5 | ent005 seed0 | 500k | -98.6 | -77.8 | 22 |
| _6 | ent005 seed1 | 500k | -98.4 | -88.7 | 25 |
| _7 | ent001_shaping02 seed0 | 500k | -90.9 | -82.5 | 46 |
| _8 | ent001_shaping02 seed1 | 300k | -93.4 | -79.8 | 45 |

**Key observations:**

1. **Shaping alone (0.01) is insufficient.** Runs 1–4 (baseline, shaping=0.01) show no improvement from 100k to 500k. Best eval stays around -77 to -81, ep_len fluctuates 18–42. The dense signal is too weak relative to the crash penalty — the drone learns crash avoidance but not navigation.

2. **High ent_coef (0.05) causes entropy trap.** Runs 5–6 show `rollout/ep_len_mean` at 3600 (full episodes) during training — the stochastic policy stays airborne. But `eval/mean_ep_len` drops to 22–25 — the deterministic policy crashes immediately. The policy has learned to output high-variance random actions to stay airborne by chance; the underlying action mean is useless. Visually confirmed: the drone flies away immediately in deterministic mode.

3. **Stronger shaping (0.2) with ent=0.01 is the best so far.** Runs 7–8 have the highest final ep_len (45–46) in eval, and `eval/mean_ep_len` was observed to progress during training rather than stay flat. The stronger dense signal (0.2) guides the deterministic policy toward waypoints without masking it with entropy noise.

4. **500k is still insufficient for sparse waypoint reward breakthrough.** No run reaches a positive eval reward — the drone navigates better but has not consistently entered the 4m goal radius in deterministic eval. The `rollout/ep_rew_mean` was observed rising in runs 7–8 (shaped reward increasing), but the sparse component hasn't triggered.

**Conclusion:**
The best PPO configuration found so far is `ent_coef=0.01, shaping_coef=0.2`. The drone shows meaningful improvement in deterministic episode length. However 500k steps in a 150m dome with a 4m goal radius remains too hard — the agent needs either more steps, a larger goal radius, or a curriculum. **Curriculum approach (smaller dome, larger goal, 1 target) launched next.**

---

### SAC Waypoints with Reward Shaping — mode 6, 100k (2026-04-19)

**Goal:** Rerun `sac_waypoints_g099_utd2_mode6` with `WaypointDistanceShaping` (shaping_coef=0.01) to compare against the unshapped first sweep.

**Config:** mode=6, gamma=0.99, UTD=2, seeds=[0, 1], timesteps=100k

| Zoo run | seed | final eval | best eval | final ep_len |
|---------|------|------------|-----------|--------------|
| _4 | 0 | -90.3 | -89.4 | 48 |
| _5 | 1 | -98.2 | -90.8 | 15 |

**Comparison vs unshapped (runs _1, _2):**

| | unshapped seed0 | unshapped seed1 | shaped seed0 | shaped seed1 |
|---|---|---|---|---|
| best eval | -82.8 | -86.5 | -89.4 | -90.8 |
| final ep_len | 19 | 16 | 48 | 15 |

Shaped seed0 shows higher ep_len (48 vs 19) suggesting the deterministic policy survives longer, but best eval reward is actually slightly worse. At 100k steps, shaping_coef=0.01 is too weak to make a meaningful difference for SAC either — the same conclusion as PPO.

**Next step:** Run SAC with shaping_coef=0.2 at higher timesteps, or focus on curriculum first.

---

### PPO Waypoints Curriculum — mode 6, easy settings (2026-04-19)

**Goal:** Break the sparse reward barrier by simplifying the task: 1 target, 30m dome, 10m goal radius. Establish whether the policy can learn single-waypoint navigation before scaling up.

**Config:** mode=6, seeds=[0, 1], timesteps=500k–1M, ent_coef=0.01, shaping_coef=0.2
- `num_targets=1`, `flight_dome_size=30`, `goal_reach_distance=10`, `max_waypoints=1` (obs size 24)

| Config | seeds | timesteps | best eval | final eval |
|--------|-------|-----------|-----------|------------|
| `ppo_waypoints_curriculum_mode6` | 0, 1 | 500k | -70.3 (s0), -84.0 (s1) | -94.6 (s0), -97.8 (s1) |
| `ppo_waypoints_curriculum_s2_mode6` | 0 | 410k | -92.6 | -96.4 |
| `ppo_waypoints_curriculum_v2_mode6` | 0, 1 | 1M | -83.7 (s0), -66.8 (s1) | -97.2 (s0), -88.3 (s1) |

SAC curriculum also ran (`sac_waypoints_curriculum_mode6`, seed 0, 540k steps): final eval -104.3, best -96.1 — worse than baseline, drifting negatively over time.

**Conclusion:**
The curriculum approach did not break through on mode 6. PPO curriculum v2 seed1 reaches a best of -66.8 around 730k steps but collapses back afterward. No run achieves a positive eval reward. The easy task settings (30m dome, 10m goal) are not sufficient to bridge to mode 6 navigation. SAC curriculum actively degrades.

---

### SAC Waypoints mode 0 — undocumented runs, retroactive analysis (2026-04-19)

**Note:** These runs were executed as part of the same experiment batch as the mode 6 waypoints sweeps but were never written up. The configs are in `configs/sac/mode0/`. Results are in `results/artifacts/` (not in the zoo).

**Config:** `PyFlyt/QuadX-Waypoints-v4`, same env settings as mode 6 (`flight_dome_size=150`, `goal_reach_distance=4`, `num_targets=4`), same `WaypointDistanceShaping(shaping_coef=0.01)` + `FlattenWaypointEnv`. Only difference from mode 6 configs: `flight_mode=0`.

| Config | seed | best eval | final eval | artifact |
|--------|------|-----------|------------|----------|
| `sac_waypoints_g099_utd2_mode0` | 0 | **316.5** | 316.5 | `sac_waypoints_g099_utd2_waypoints_mode0_seed0` |
| `sac_waypoints_g099_utd2_mode0` | 1 | 67.4 | 27.3 | `sac_waypoints_g099_utd2_waypoints_mode0_seed1` |
| `sac_waypoints_g099_utd3_mode0` | 0 | **477.1** | -37.1 | `sac_waypoints_g099_utd3_waypoints_mode0_seed0` |
| `sac_waypoints_g099_utd3_mode0` | 1 | 34.2 | -109.9 | `sac_waypoints_g099_utd3_waypoints_mode0_seed1` |
| `sac_waypoints_g0995_utd2_mode0` | 0 | 162.4 | -11.8 | `sac_waypoints_g0995_utd2_waypoints_mode0_seed0` |
| `sac_waypoints_g0995_utd2_mode0` | 1 | **396.9** | 166.6 | `sac_waypoints_g0995_utd2_waypoints_mode0_seed1` |
| `sac_waypoints_g0995_utd3_mode0` | 0 | **389.5** | 389.5 | `sac_waypoints_g0995_utd3_waypoints_mode0_seed0` |
| `sac_waypoints_g0995_utd3_mode0` | 1 | 67.9 | -63.8 | `sac_waypoints_g0995_utd3_waypoints_mode0_seed1` |

For completeness, mode 6 artifact results (same configs, same 100k budget):

| Config | seed | best eval | final eval |
|--------|------|-----------|------------|
| `sac_waypoints_g099_utd2_mode6` | 0 | -222.5 | -222.5 |
| `sac_waypoints_g099_utd2_mode6` | 1 | -155.2 | -194.4 |
| `sac_waypoints_g099_utd3_mode6` | 0 | -250.5 | -361.5 |
| `sac_waypoints_g099_utd3_mode6` | 1 | -156.3 | -339.1 |
| `sac_waypoints_g0995_utd2_mode6` | 0 | -224.5 | -224.5 |
| `sac_waypoints_g0995_utd2_mode6` | 1 | -177.4 | -249.0 |
| `sac_waypoints_g0995_utd3_mode6` | 0 | -190.9 | -206.8 |
| `sac_waypoints_g0995_utd3_mode6` | 1 | -194.8 | -329.3 |

**Statistical caveat:** The artifact evals use only **5 episodes per checkpoint**, vs 20 in the zoo runs. With such small samples, a single high-reward episode can dominate the mean. The reported means must be read alongside the per-episode min/max.

**Key observations:**

1. **Mode 0 does occasionally reach waypoints; mode 6 does not.** The shaping coefficient is 0.01 and distances are O(10–100m), so the shaping term contributes at most a few units per step. Episode rewards of +500 to +1800 can only come from the sparse goal bonus — the drone is physically reaching waypoints. No mode 6 episode across any run exceeds +5.

2. **Mode 0 positive results are real but not stable.** The high means are largely driven by single outlier episodes. For example at 100k steps: `g099_utd2_seed0` shows mean=316 but min=-106 and max=1313 (5 episodes); `g0995_utd3_seed0` shows mean=390 but min=-118 and max=1825. The most convincing evidence of genuine learning is `g0995_utd3_seed0` at steps 80k–90k where **all 5 eval episodes are positive** (80k: min=2.9, max=889; 90k: min=14.3, max=288). After that the policy regresses.

3. **No run has a consistently positive policy.** The mode 0 agent occasionally discovers waypoint-reaching behavior but does not retain it reliably. This is exploratory success, not convergence.

4. **Mode 6 is stuck at a different reward scale entirely.** Mode 6 artifact evals sit in the -200 to -400 range (raw training checkpoints), consistent with the zoo runs (-90 to -103 on the best saved model). No waypoint is ever reached.

5. **The mode 0 vs mode 6 gap is real but the cause is unclear.** The configs are identical. Mode 0 (angular velocity + thrust) gives the agent more direct control over the drone's motion; mode 6 (world-frame velocity commands) routes through a multi-layer PID stack that introduces lag and dampening. Whether this explains the gap or whether it is a sample-efficiency artifact of SAC's replay buffer with these dynamics is unknown.

**Conclusion:**
Mode 0 SAC shows early signs of waypoint navigation within 100k steps but the policy is unstable and highly variable. Mode 6 shows no waypoint navigation at all. Both findings are based on low-sample evals (5 episodes) — **these results should be treated as exploratory, not conclusive**. The right next step for mode 0 is a longer run (500k+) with more eval episodes to establish whether the policy can be stabilized. Mode 6 remains a separate unsolved problem.

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

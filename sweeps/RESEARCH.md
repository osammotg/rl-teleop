# ACT sweep research — what is worth training, and how to rank it without val loss

Stream E, overnight run of 2026-08-08. Research and design only: **no hardware was touched
and no training was launched.** The GPU was used for two read-only measurements (an
inference microbenchmark and 345 forward passes for the proxy prototype), each a few
seconds; nothing was written into `rl-teleop/outputs/` (symlink to the live production
tree).

Everything below marked **[measured]** was computed tonight on this box from the artefacts
on disk and can be reproduced from the snippets given. Everything marked **[inherited]**
comes from `~/Desktop/lab/act-research/ACT_RESEARCH_GUIDELINES.md` (2026-07-26) and I say
where I verified or contradicted it. **[web]** is new literature research done tonight.

---

## 0. The seven conclusions, before the evidence

1. **No hyperparameter has ever been varied on this rig.** All four training runs on disk
   are byte-for-byte identical in every policy hyperparameter, and identical to upstream
   LeRobot ACT defaults. The only things that ever changed were the dataset, the number of
   steps, and what the run was initialised from. There is no sweep to extend — there is
   only a sweep to start. §1.

2. **The number everyone has been calling "val loss" is training loss, and there has never
   been a validation set.** LeRobot's train script only evaluates when `cfg.env is not
   None`; here it is `None` in all four runs, so `is_eval_step` never fires. There is no
   held-out split anywhere in this pipeline. §4.1.

3. **The loss is also mis-scaled in a way that makes it non-comparable across exactly the
   comparisons that were made.** ACT's L1 term is divided by `B × chunk_size × action_dim`
   *including the masked padding positions*, so the reported number scales with the
   supervised fraction. That fraction is **44.8% on `yam_grasp_v1` and 65.1% on
   `yam_grasp_v2_wrist`** [measured]. Identical per-step accuracy therefore prints a ~1.45×
   *higher* loss on v2. This is a concrete, previously-unidentified mechanism behind "the
   best-loss checkpoint was the worst one". §4.2.

4. **The corpus is 15 recordings, not 69 episodes.** The 69 training windows come from 15
   teleop takes on 3 calendar days; 4 takes supply 32 of the 69 [measured]. Effective
   sample size for anything a hyperparameter could change is ~15, not 69. No
   hyperparameter fixes a 15-scene corpus, and no offline metric computed on those same
   15 scenes can rank checkpoints that trained on all of them. §4.4.

5. **A holdout already existed, and on it the proxy beats the loss and inverts its ranking.**
   The zone filter dropped 8 grasps from v1 when building v2, so those 8 are unseen by every
   v2-line checkpoint. Scored there: the **deployed** `ft50k/006000` prints a **31% worse**
   loss than `v2 scratch 50000` and beats it **7/7 paired** on held-out grasps (sign test
   p ≈ 0.016), with a generalisation gap of **1.43× vs 2.31×**. The warm start from the
   two-camera v1 pretrain is the reason. §4.7.

6. **The corpus is not partly corrupt** — settled by another stream: `|stored p − FK(q)|` is
   identically 0.000000 mm on all 77, and the "24 of 77" was leader-vs-follower teleop tracking
   error, not a data defect. I have removed that caveat throughout. It **strengthens** the
   central argument rather than weakening it: with the action labels exact, "the data is
   corrupt" is eliminated as a competing explanation for why loss fails to predict rollout
   success, leaving the two mechanisms measured here — no holdout (§4.1) and a padding-scaled
   loss (§4.2).

7. **The two highest-expected-value *deploy* experiments require zero training runs.**
   `n_action_steps` and `temporal_ensemble_coeff` are both pure inference-side knobs
   (verified: the temporal ensembler holds no learned parameters), and the deployed
   override `N_ACTION_STEPS = 16` is contradicted by the only quantitative evidence that
   exists — LeRobot's own 500-episode eval, where dropping the executed horizon collapsed
   ACT from 87.6% to 2.0%. Nobody has measured 16-vs-100 on this rig. §3.0.

---

## 1. What has already been tried

### 1.1 The four runs, diffed

I dumped all four `train_config.json` and diffed them against each other and against the
installed `lerobot==0.4.3` `ACTConfig` defaults.

**Every policy hyperparameter is identical in all four runs and equal to the upstream
default.** All of these, in all four runs:

```
chunk_size 100   n_action_steps 100   n_obs_steps 1   dim_model 512   n_heads 8
dim_feedforward 3200   n_encoder_layers 4   n_decoder_layers 1   pre_norm false
vision_backbone resnet18 (ImageNet1K_V1)   replace_final_stride_with_dilation false
use_vae true   latent_dim 32   n_vae_encoder_layers 4   temporal_ensemble_coeff null
dropout 0.1   kl_weight 10.0   normalization VISUAL/STATE/ACTION = MEAN_STD
optimizer adamw lr 1e-5   lr_backbone 1e-5   weight_decay 1e-4   grad_clip_norm 10.0
betas (0.9,0.999)   scheduler null (constant LR)   batch_size 8   num_workers 4
image_transforms: the identical 7-entry `dark_noise` recipe, max 3 of 7, random order
```

Diff against upstream ACT defaults: **zero deltas.** `grad_clip_norm 10.0` and
`batch_size 8` are LeRobot trainer defaults too. The only non-default thing in this
project is the `dark_noise` augmentation recipe, which LeRobot ships disabled
(`ImageTransformsConfig.enable = False`) and which is identical across all four runs. So
the augmentation axis is also untested here: it has been *on*, at one setting, always.

The complete set of things that differ:

| | `act_grasp_dark_noise_20260728` | `act_wrist_zone_v2` | `act_wrist_zone_v2_ft50k` ← **deployed** | `act_wrist_zone_v3_cont15k` |
|---|---|---|---|---|
| dataset | `yam_grasp_v1` | `yam_grasp_v2_wrist` | `yam_grasp_v2_wrist` | `yam_grasp_v2_wrist` |
| cameras | top 720×1280 **+** wrist 480×640 | wrist only | wrist only | wrist only |
| episodes / frames | 77 / 6 785 | 69 / 9 772 | 69 / 9 772 | 69 / 9 772 |
| window length | **90 frames (3.0 s)** | **150 frames (5.0 s)** | 150 | 150 |
| steps | 50 000 | 50 000 | **6 000** | 15 000 |
| init from | scratch | scratch | `outputs/pretrained/act_50k_wristonly` | `…ft50k/checkpoints/006000` |
| seed | 1000 | 1000 | 1000 | **2000** |
| save_freq | 5 000 | 5 000 | 1 000 | 3 000 |
| video_backend | pyav | torchcodec | torchcodec | torchcodec |
| epochs over frames @ bs 8 | 58.9 | 40.9 | 4.9 | 12.3 (on top of 54.9) |
| disk | 5.8 GB | 5.8 GB | 3.5 GB | 2.9 GB |

### 1.2 The deployed checkpoint's lineage is not what the directory name says

`outputs/pretrained/act_50k_wristonly/model.safetensors` is **bit-identical**
(`md5 ef7b9a14…`) to `act_grasp_dark_noise_20260728/checkpoints/050000/…/model.safetensors`.
Only `config.json` was rewritten (mtime Aug 3 23:03) to declare a single
`observation.images.wrist` input. That is what `tools/rebase_checkpoint_cameras.py` does.

So the deployed policy is:

```
77 grasps, TOP + WRIST, 90-frame windows, 50 000 steps from scratch
  → config hand-edited to drop the top camera
  → 6 000 steps of fine-tune on 69 grasps, WRIST only, 150-frame windows
  = act_wrist_zone_v2_ft50k/checkpoints/006000     ← in the sort server since 2026-08-03
```

This works structurally — LeRobot ACT shares one ResNet18 across all camera keys and the
image position embedding is sinusoidal, so dropping a key just removes tokens. But it means
the deployed weights spent 90% of their gradient steps looking at a camera they will never
see again, on windows 40% shorter than the ones they were fine-tuned on. **`act_wrist_zone_v2`
(50 000 steps from scratch, wrist-only, matched windows) has never been compared head-to-head
against it.** That comparison is free — both checkpoints exist — and is item 2 in the run list.

### 1.3 Experiments already run — do not repeat

- **Camera set 2 → 1** (top+wrist → wrist). Run, but *confounded* with the zone filter
  (77→69 windows) **and** with the window length change (90→150 frames) **and** with
  scratch-vs-finetune. It is not an ablation; it is four changes at once.
- **Zone filter** `y ≤ -0.13`, 77 → 69. Run. `PLAN-ACT-V2-ZONE-FILTERED.md` already
  records that the `x ≥ 0.325` half of the gate is a no-op (zero grasps in [0.25, 0.325)).
- **Training length** 5k…50k on v1, 5k…50k on v2, 1k…6k on the fine-tune, 3k…15k on the
  continuation. The length axis is thoroughly covered *as a checkpoint ladder*; what is
  missing is any way to choose among the rungs.
- **Seed**: exactly one change (1000 → 2000), and it was confounded with a different
  initialisation. **Run-to-run variance has never been measured.** This matters more than
  it sounds — see §3.1.

### 1.4 Logs

Only one training log survives: `outputs/logs/act_wrist_zone_v3_cont15k.log`. It prints
`loss:` only (the `l1_loss`/`kld_loss` split goes to W&B, which is disabled), and the value
drifts 0.042 → 0.038 over 15k steps with no schedule. `grad_norm` sits at 3.3–3.9
throughout, nowhere near `grad_clip_norm=10`, so clipping never engaged.

---

## 2. Literature

### 2.1 What I inherited and re-verified

`ACT_RESEARCH_GUIDELINES.md` cites `lerobot==0.5.0`; **this box runs `lerobot==0.4.3`.** Its
line numbers are therefore all wrong for the installed code (by 10–16 lines in
`configuration_act.py`). I re-checked every substantive code claim in it against 0.4.3:

| Its claim | Status in 0.4.3 |
|---|---|
| defaults `chunk_size=100`, `n_action_steps=100` | **confirmed**, `configuration_act.py:80-81` |
| `select_action` only re-runs the model when the queue drains | **confirmed**, `modeling_act.py:114-122` |
| `temporal_ensemble_coeff is not None` forces `n_action_steps == 1` | **confirmed**, `__post_init__` raises `NotImplementedError` |
| `n_obs_steps != 1` is rejected | **confirmed** |
| `use_vae` default true, `kl_weight` default 10.0 | **confirmed** |
| `observation.environment_state` is a first-class ACT input with its own projection and position embedding | **confirmed**, `modeling_act.py:344-347, 357-358, 465-466`; and `ENV` is still absent from the default `normalization_mapping`, so its warning holds |
| the VAE encoder does not see `env_state` | **confirmed**, `modeling_act.py:296-320` |

So: **its engineering conclusions survive the version gap; its line references do not.** Cite
the facts, not the line numbers.

**Where it was adopted, and where it was ignored** (against its own Part VI table):

| Its recommendation | Adopted? |
|---|---|
| F1 `n_action_steps` 100 → 12–20 | **yes** — `act_runner.py:63` sets 16. But see §2.3: this is the one recommendation the new evidence attacks. |
| F2 camera ablation 1 / 2 / 3 | **partly** — went straight to 1, never ran the ablation, and confounded it |
| F3 stop using loss for checkpoint selection; build a rollout harness as the objective | **no** — no harness exists; selection is still by hand |
| F4 DART noise at recording time | **no** — and it cannot be retrofitted |
| F5 diversity over demo count | **no** — 15 takes over 3 days is the opposite |
| F6 add random crop to the aug stack | **no** — the aug recipe is unchanged in all four runs |
| "≥35K steps, 90/10 holdout" | **half** — 50k steps yes; **holdout never existed** |
| keep `use_vae`, keep `kl_weight=10`, keep `n_obs_steps=1` | yes (by inaction) |
| temporal ensembling trial at 0.01, gated on a latency measurement | **no**, and the latency was never measured. I measured it — §2.4. |

### 2.2 Upstream defaults, confirmed against source

Verified independently against the installed 0.4.3 and against
`github.com/huggingface/lerobot/…/policies/act/configuration_act.py` (identical): the table
in §1.1 is the upstream default set. Two things worth knowing:

- **`n_decoder_layers = 1`, not the paper's 7.** Upstream comment: the original ACT code has
  a bug where only the first decoder layer is used, and LeRobot matches the *behaviour*
  (github.com/tonyzhaozh/act/issues/25). Do not "fix" this by raising it — you would be
  training a different architecture than every published ACT number.
- **`get_scheduler_preset()` returns `None`.** ACT in LeRobot trains at a constant LR with no
  warmup and no decay. That is also what the original repo effectively does.

Original ACT paper (arXiv:2304.13705) Table III: `lr 1e-5`, `batch 8`, 4 encoder / 7 decoder
layers, ffn 3200, hidden 512, 8 heads, `chunk 100`, `beta 10`, `dropout 0.1`. Control runs at
**50 Hz** (`DT = 0.02` in the repo's `constants.py`), so **the paper's chunk 100 is 2.00 s**.
At our 30 fps it is **3.33 s**. Time-matching the paper would put `chunk_size ≈ 60`. No
published source gives chunk-size-vs-fps guidance; that arithmetic is the whole argument.
The question is asked and left unanswered by maintainers in
`github.com/huggingface/lerobot/issues/2213`.

### 2.3 Action horizon — the new evidence contradicts the deployed setting

This is the most important thing the web research turned up, and it points the opposite way
from the inherited doc.

**ACT paper Fig 8a** (chunk-size ablation, temporal ensembling disabled, separate policies
per k, k ∈ {1, 10, 100, 200, 400}): success rises **1% at k=1 → 44% at k=100**, then tapers
slightly at 200/400, attributed to loss of reactivity.

**LeRobot PR #319**, eval of `lerobot/act_aloha_sim_transfer_cube_human` over **500 episodes**:

| setup | success |
|---|---|
| no temporal ensembling, `n_action_steps = 100` | **87.6%** |
| temporal ensembling m = 0.01 (the ACT paper's value) | 73.8% |
| m = 0 (uniform) | 76.8% |
| m = −0.01 (newer actions weighted more) | 79.0% |
| **`n_action_steps = 1`, no ensembling** (50 eps) | **2.0%** |

Two readings follow, and they matter here:

1. **Chunking's benefit is in *executing* the chunk open-loop, not in *predicting* it.**
   Discarding 99 of 100 predicted steps destroyed the policy. Our deployment discards 84 of
   100. Nobody has measured what that costs.
2. **LeRobot's own replication contradicts the ACT paper on temporal ensembling.** The paper
   reports +3.3% for ACT; LeRobot's 500-episode eval has *every* ensembling setting below the
   no-ensembling baseline, and the best coefficient has the *opposite sign* to the paper's.

Caveat, stated plainly: PR #319 is one sim task, 500-step episodes where a 100-chunk is 1/5 of
an episode. Here a 100-chunk is 2/3 of a 150-frame episode, the task is 3 s long, and the whole
stated purpose of the learned block is to absorb IK/hand-eye error during the descent — which a
fully open-loop chunk cannot do. **Both arguments are sound and they disagree. That is exactly
what makes it worth measuring rather than assuming, and it costs nothing to measure.**

There is no published ACT data at intermediate horizons (10, 25, 50 of 100). The 10–20 figure
circulating in practitioner threads comes from diffusion/VLA policies (Diffusion Policy's
convention is predict 16 / execute 8), not ACT.

### 2.4 Temporal ensembling is affordable here — [measured]

The inherited doc gated any temporal-ensembling trial on "measure `predict_action_chunk`
wall-clock first". Measured tonight, deployed checkpoint, wrist-only, batch 1, RTX 5090,
60 iterations after 10 warm-up:

```
predict_action_chunk:  mean 6.16 ms   p50 5.97 ms   p95 8.22 ms   max 12.01 ms
51.6 M params, 279 MB peak activation
```

The 30 Hz budget is 33.3 ms. **One full forward per control tick uses 18% of it.** The gate is
cleared: temporal ensembling (which forces `n_action_steps=1`) is computationally feasible on
this rig. Whether it is a *good idea* is a separate question that PR #319 answers unfavourably.

Also verified: `ACTTemporalEnsembler` is a plain class holding
`torch.exp(-coeff * arange(chunk_size))` — **no learned parameters**. So temporal ensembling
can be switched on for any existing checkpoint by editing `config.json`, with no retraining.
**It is a deploy knob, not a training axis.**

### 2.5 The camera decision is the one the literature actively disagrees with

No camera-count ablation exists in the ACT paper (I had the researcher grep the extracted PDF;
its four ablations are chunk size, temporal ensembling, CVAE, and a 50 Hz-vs-5 Hz user study).
Its *sim* tasks use a single **overhead** camera and reach ~90%; its real tasks use four.

The ablation that does exist is **AV-ALOHA** (arXiv:2409.17435), which trained ACT via LeRobot
on all 7 combinations of {static, wrist, active-vision} across 6 tasks at 50 demos each:

- **wrist-only ranged 8–44%** and was consistently the weakest single configuration;
- the winners were static+wrist or AV+static;
- *"Using all the cameras simultaneously did not perform well across the tasks and never
  ranked in the top three for any task."*

Corroborated by a Unitree G1 study (arXiv:2603.28422) across 14 sensor combinations:
*"strategic sensor selection can outperform complex configurations in data-limited regimes"* —
extra modalities **degraded** ACT when data was scarce.

This is a real tension with the inherited F2 (Hsu et al. 2022, eye-in-hand helps OOD
generalisation), and with the wrist-only line this project has committed to. **But there is a
local argument the literature does not know about**: `AUDIT.md` §S5.3 flags that a two-camera
export compares a RealSense hardware clock against `time.time()` and may silently drop frames,
while `camera_left` (the wrist USB cam) is same-domain and safe. So wrist-only may be right *by
accident*. That is testable offline in minutes (§5, run 0b) and should be tested before any
two-camera retrain is launched.

### 2.6 Small datasets

- **The ACT paper is itself a ~50-demo result** — 50 demos per task (100 for Thread Velcro),
  10–20 minutes of data, 8–14 s episodes ≈ 20k frames. So 69 windows is not absurd in itself.
  But its episodes are whole tasks; ours are 5-second slices from **15 takes**.
- **Data scaling** (Lin et al., arXiv:2410.18647, ICLR 2025, >40k demos, >15k real rollouts):
  generalisation follows a power law in the number of **environments and objects**, and *"once
  the number of demonstrations per environment or object reaches a certain threshold,
  additional demonstrations have minimal effect."* Their recipe is 32 environments × 50 demos.
  We have ~3 sessions × 15 layouts.
- LeRobot's own guidance: ≥50 episodes, ≥10 per location, fixed cameras, consistent grasping.

**Read for us:** the binding constraint is 15 scenes, and it is a *data-collection* constraint.
No entry in the run list below can move it. Say so out loud to whoever reads the sweep results.

### 2.7 Augmentation and KL weight — genuine literature gaps

- **The original ACT uses no image augmentation at all** (only ImageNet normalisation), and
  LeRobot ships `ImageTransformsConfig.enable = False`. There is **no published ablation of
  augmentation on ACT**. Notably, the default LeRobot bank has **no random crop** — the
  augmentation robomimic-lineage work relies on most.
- **There is no published sweep of `kl_weight`.** The only VAE ablation in the paper is β=10
  vs no-CVAE, and on *human* data removing it drops success **35.3% → 2%** (on scripted data
  it makes no difference). Since all our data is human GELLO teleop, `use_vae=true` is settled
  and `kl_weight` is unmeasured everywhere, not just here. The "β=20–50 for consistency"
  advice circulating on tutorial sites has no primary source behind it.
- `optimizer_lr_backbone` defaults **equal to** `optimizer_lr` (1e-5) in both ACT and LeRobot.
  The widely-repeated "ACT uses a 10× lower backbone LR" is a DETR inheritance that ACT
  overrides. The plumbing exists; the differential does not. LeRobot's own source carries a
  `TODO(aliberts, rcadene): As of now, lr_backbone == lr / Should we remove this`.

### 2.8 Offline metrics that predict rollout success

- **robomimic** (Mandlekar et al., CoRL 2021) is the canonical citation, and the number is
  quotable: *"the best validation loss does not correspond to the best performing policy. The
  best validation policy is **50 to 100% worse** than the best performing policy."* Their
  protocol is max-success-over-checkpoints, evaluated on the robot.
- The **ACT README** itself says *"success rate and smoothness can improve way after loss
  plateaus"* and recommends training 3–4× longer than the plateau — while its own eval script
  loads `policy_best.ckpt` selected by minimum validation loss. The authors contradict
  themselves in one file.
- **CI-MSE** (arXiv:2606.29898) is the most quantitative recent work on this exact question,
  and its finding is the design principle I build §4 on: raw validation MSE against rollout
  success gives Pearson **r = −0.56**, Spearman **ρ = −0.61**; restricting the error to
  **task-critical segments** and adding chunk alignment lifts it to **r = −0.74, ρ = −0.87**.
  Evaluated on π₀.₅ / X-VLA / GR00T and real diffusion policies, **not on ACT** — so the
  mechanism transfers, the numbers do not. *The fix is not "abandon offline metrics"; it is
  "stop averaging over the whole trajectory".*
- **LeRobot has no validation functionality for real datasets** (issue #250). And there is a
  live footgun: issue #2851 reports that adding a validation split made ACT/π₀/SmolVLA
  **freeze and not move** at inference, suspected to be normalisation stats computed over all
  episodes regardless of the split. Anyone adding a holdout here must check the stats. §5
  addresses this.

---

## 3. The sweep plan, ranked

### 3.0 Tier 0 — no training runs at all. Do these before anything else.

| # | experiment | why it is first | cost |
|---|---|---|---|
| **T0.1** | `N_ACTION_STEPS` ∈ {16 (current), 32, 60, 100} at rollout | The only quantitative evidence (§2.3) says the current 16 may be costing most of ACT's benefit — 87.6% → 2.0% at the extreme. The counter-argument (closed-loop correction of IK error) is equally sound. It is a one-line change in `act_runner.py:63` and needs no retraining. **Expected to be the largest single effect in the whole document.** | 0 GB, 0 GPU-h |
| **T0.2** | `temporal_ensemble_coeff` ∈ {None (current), 0.01, −0.01} with `n_action_steps=1` | Verified deploy-only, no learned params; latency measured at 6.2 ms of a 33 ms budget (§2.4). LeRobot's own eval says it will lose to T0.1's best; worth one arm as a control, and m=−0.01 beat m=0.01 there. | 0 GB, 0 GPU-h |
| **T0.3** | Recover the dataset→grasp manifest that `AUDIT.md` §S2 says does not exist | Done tonight, ~20 lines, in `sweeps/grasp_proxy.py::episode_to_recording`. Nearest-neighbour on the 6 arm joints at the annotated grasp instant matches **69/69 episodes to distinct `demo_grasps.json` keys, max residual 0.004 rad**. Without it there is no grouped holdout and therefore no honest sweep. | 0 |
| **T0.4** | Run `AUDIT.md` §S5.3's clock-domain probe on one recording | Decides whether a two-camera retrain (S1.4) is even meaningful, before spending runs on it. Offline, no hardware. | 0 |

T0.1 and T0.2 need rollouts, which need hardware, which is out of scope tonight — but they
need **no GPU and no disk**, and they should be scheduled ahead of every training run below.

### 3.1 Tier 1 — the axes that actually matter for *this* data

**S1.1 — `chunk_size`. The single training axis I am confident about.**

Four independent reasons, three of them measured here:

1. **[measured]** At `chunk_size=100` on 150-frame windows, **34.9%** of the decoder's output
   positions are padding and receive no gradient. On the v1 90-frame windows it was **55.2%**.
   Nobody chose those numbers; they are what falls out of leaving chunk at the default while
   the export window changed from 3.0 s to 5.0 s.
2. **[measured]** The demonstrated grasp instant sits at **frame 90 of 150** in every
   full-length episode (`pre_s × fps`). Query there and only 60 of the 100 predicted steps
   have any target at all — **40% padding at the decisive moment.**
3. **[web]** ACT's tuned `chunk_size=100` is 2.00 s at its 50 Hz. Ours is 3.33 s. Time-matching
   gives ~60.
4. **[measured]** Because the L1 term divides by the full padded tensor (§4.2), reducing
   `chunk_size` *raises* the printed loss for identical accuracy. Any past intuition that "the
   loss looks fine" was partly measuring the padding fraction.

   Sweep `{100 (control), 60, 30}`. Do **not** go below 30: ACT's own ablation puts k=1 at 1%
   and k=10 well below k=100, and PR #319 puts execution-horizon-1 at 2.0%. 30 frames = 1.0 s
   is the aggressive end worth risking; 16 is not.

   Footgun: `__post_init__` raises if `n_action_steps > chunk_size`, so **both flags must be
   set together** or the run dies at config parse.

**S1.2 — the run-to-run variance floor. Not a sweep; a prerequisite.**

`PLAN-ACT-V2-ZONE-FILTERED.md` already worries in writing that its 8-window change "may land
inside run-to-run variance". Nobody has ever measured that variance: exactly one seed change
exists on disk and it was confounded with a different initialisation. **Three identical runs at
seeds 1000/2000/3000 give the noise floor against which every other arm in this sweep is
judged.** Without it, a 3-arm chunk-size sweep on a 19-window holdout produces three numbers
and no way to know whether any of them differ. This is the cheapest run in the list and the
one that makes the others interpretable.

**S1.3 — scratch-vs-fine-tune, under matched conditions.**

The deployed checkpoint is a 6k fine-tune of a *two-camera* model whose config was hand-edited
(§1.2); `act_wrist_zone_v2/050000` is 50k from scratch on the same data. They have never been
compared. Both already exist, so the *offline* comparison is free — but it is only meaningful on
held-out recordings, which neither of them has. So this becomes two runs on the grouped split.

**S1.4 — camera set, gated on T0.4.**

Every published ACT camera ablation ranks wrist-only last (§2.5). The switch to wrist-only here
was confounded with three other changes. One run — top+wrist on the *same* 69 zone-filtered
grasps with the *same* 150-frame windows — turns four simultaneous changes into one. **Gate it
on T0.4**: if the clock-domain probe shows `camera_top` frames are being silently dropped, the
two-camera export is corrupt and the run is wasted. Also costs ~4× the wall-clock (the top
camera is 720×1280) and needs a v2-equivalent two-camera export first (~1.5 GB predecode).

### 3.2 Tier 2 — worth one run each, lower confidence

**S2.1 — random crop in the augmentation stack.** The inherited F6, still unadopted, and
**there is no published ACT augmentation ablation at all** (§2.7) — so this arm produces new
information regardless of outcome. With 15 scenes, augmentation is the only lever that
manufactures effective diversity. Keep it modest (≥90% of frame) for a precision task.

**S2.2 — training length on the grouped split.** 50k steps is 41 epochs over frames on v2.
The ACT README says train 3–4× past the loss plateau, and the grad-norm trace shows no sign of
convergence trouble. Not a separate run — it falls out of the checkpoint ladder of every run
above, *provided* the proxy is computed per checkpoint rather than only on `last`.

### 3.3 Axes I recommend **not** sweeping, and why

This is the most valuable section. Each of these would burn GPU-hours to produce a number the
corpus cannot resolve.

| axis | why not |
|---|---|
| **`n_action_steps`** | Read only by `select_action` to size the action queue; `chunk_size` is what shapes the network. Deploy-time knob — training it away costs a run and buys nothing. The training script's own docstring already says this and it is correct. → T0.1. |
| **`temporal_ensemble_coeff`** | Verified: the ensembler holds no learned parameters. Pure inference-side. → T0.2. |
| **`kl_weight`** | No published sweep exists anywhere (§2.7). The only ACT ablation is 10 vs off, and off is catastrophic on human data. A 19-window holdout on 15 scenes cannot resolve β=5 vs 10 vs 20 — the difference would sit far inside the seed variance S1.2 will measure. |
| **`use_vae`** | Settled: 35.3% → 2% on human demos. All our data is human GELLO teleop. Turning it off is a known-bad experiment. |
| **`optimizer_lr`, `lr_backbone`, `weight_decay`, betas, LR schedule** | 1e-5 constant is the published ACT value, the LeRobot default, and what all four runs used. Grad-norm sits at 3.3–3.9 against a clip of 10 — the optimisation is not straining. Upstream deliberately ships no scheduler. Nothing in the evidence points at the optimiser, and an LR sweep on 15 scenes measures seed noise. If you must touch one thing, it is *steps*, and that is already free (S2.2). |
| **`dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `latent_dim`** | 51.6 M parameters against 9 772 frames from 15 scenes. Capacity is not the bottleneck by three orders of magnitude. Every one of these is also a departure from the only configuration any published ACT number was measured on. |
| **`n_decoder_layers`** | The default of 1 is deliberate upstream bug-compatibility with the original ACT (only the first decoder layer is ever used). Raising it to the paper's 7 trains an architecture nobody has published a number for. |
| **`vision_backbone` resnet34/50** | More capacity on less data, and 3–4× the inference cost against a 33 ms budget that T0.2 may want. |
| **Freezing the backbone** | No ACT-specific evidence exists in either direction (§2.7). Defensible as a *single* cheap arm if a slot is spare, but it is a coin-flip, not a hypothesis. |
| **`n_obs_steps`** | Rejected by `__post_init__`; and the copycat literature says observation history is actively dangerous in BC. |
| **DART noise** | Cannot be added retroactively — it is a *recording-protocol* change (inherited F4) and belongs to Stream B, not to any training run. Flagging it here so it is not mistaken for a sweep axis and quietly dropped. |
| **`environment_state` target token** | Requires the label at recording time. It is *now* recoverable post-hoc (T0.3 gives the grasp-instant EE pose for all 69), but wiring it is model surgery plus a normalisation-mapping change, and the inherited doc's own decision was "log the label, don't wire the token yet". That decision still stands. |

---

## 4. A proxy objective that is not val loss

### 4.1 First: there is no val loss, and there never was

`lerobot_train.py:420` — `is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0`, and
`:461` — `if cfg.env and is_eval_step:`. **`cfg.env` is `null` in all four runs.** LeRobot has
no validation-loss path for real datasets at all (its own issue #250). So:

> The quantity that "did not predict rollout success" was the **training loss printed to the
> console, computed on the same 69 windows the model was fitting**. The finding is not
> "validation loss is a weak signal". It is "there has never been a held-out measurement of
> anything on this rig."

That reframing matters, because it means the fix is not a cleverer metric. It is a split.

### 4.2 Second: even as a training loss, the number is mis-scaled

`modeling_act.py:144-146`:

```python
l1_loss = (
    F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
).mean()
```

The mask zeroes padded positions, then `.mean()` divides by the **full** `B × chunk_size ×
action_dim`. So

```
printed_l1  =  (mean |error| over supervised steps)  ×  (supervised fraction)
```

[measured] supervised fraction: **0.448 on `yam_grasp_v1`** (90-frame windows, chunk 100) and
**0.651 on `yam_grasp_v2_wrist`** (150-frame windows, chunk 100). Two consequences:

- **Losses are not comparable between the v1 and v2 lines.** Identical per-step accuracy prints
  ~1.45× higher on v2. Any cross-line "best-loss / worst-loss" ranking was reading the window
  length.
- **Losses will not be comparable across a `chunk_size` sweep either** (0.651 → 0.827 at 60 →
  0.898 at 30). Which is a second, independent reason the sweep cannot be ranked on loss.

Also: the console prints only the *total*, `l1 + 10 × KLD`. The `l1_loss`/`kld_loss` split goes
to W&B, which is disabled in every run. Nobody knows what fraction of 0.038 is even the L1 term.

### 4.3 The proposed proxy: grasp-instant chunk agreement

Not "how well does it fit the trajectory" — **"does it reproduce the grasp, at the moment the
grasp happens, on recordings it has never seen."** Three components, in falsifiability order.
Implemented in `sweeps/grasp_proxy.py`; a working prototype ran tonight.

Protocol, per held-out episode:

1. Find the demonstrated gripper-close frame `c` — first index where `action[:,6]` crosses
   below `C.GRIPPER_CLOSE_ENTER = 0.45`, the same threshold that labelled the corpus.
2. Query the policy **once**, from the real observation at frame `c − lead` (lead = 1.0–2.5 s),
   and read the entire predicted chunk. One forward pass — no simulator, no rollout.
3. Score:

| component | definition | what it catches |
|---|---|---|
| **`close_rate`** | fraction of held-out episodes whose predicted chunk crosses below 0.45 at all | *the exact failure that broke the best-loss checkpoint.* A policy that never closes its jaws scores 0 here and is indistinguishable by loss. |
| **`close_dt`** | median \|predicted close step − demonstrated close step\| ÷ fps, over episodes that do close | closing at the wrong depth/time. Deliberately a **timing** measure, not a magnitude one: [measured] `export_lerobot.py:418-419` min-max normalises the gripper channel **per recording**, so absolute widths are not comparable across episodes — but threshold crossings are. |
| **`approach_l1`** | mean per-step L1 over the 6 arm joints, **over the non-padded part of the chunk only**, in radians | descent quality; divided by valid steps, so unlike the training loss it *is* comparable across `chunk_size`. |

Combined: `score = close_rate − 0.5·min(close_dt, 1.0) − 2.0·approach_l1`. The weights are a
judgement call; report the three components separately and let a human look at them.

**Why this should rank better than loss.** Three reasons, one of them cited:

- **It stops averaging over the trajectory.** In a 150-frame window the gripper is in motion for
  a handful of frames; ~96% of the frames are free-space approach and lift. A policy that never
  closes pays ~4% of the loss for it. This is exactly CI-MSE's mechanism (§2.8): restricting
  the error to task-critical segments moved Spearman ρ from −0.61 to −0.87 on a comparable
  problem.
- **It is scale-free where the data is not.** `close_rate` and `close_dt` are threshold
  crossings, immune to the per-recording gripper normalisation; `approach_l1` is per-step,
  immune to the padding-fraction scaling of §4.2. Both of the known ways the loss lies are
  closed.
- **It measures the thing the failure log measures.** Every failure in
  `results/sort_runs/*.json` is `no-grasp` — "never lifted 30 mm", "best 0 mm". The failure
  mode is at the grasp instant. So is the metric.

### 4.4 The honest limits — including a negative result from tonight

**I ran the prototype and it does not discriminate in-sample. That is the finding, not a bug.**

Five checkpoints, all 69 episodes (i.e. their own training data), four operating points:

```
                 lead 1.0s        lead 2.0s        lead 2.5s     lead 2.0s + state noise
                 (rate, dt, l1)   (rate, dt, l1)   (rate,dt,l1)  (rate, dt, l1)
v2_ft50k/6000    1.000 .033 .0128  1.000 .033 .0122  1.000 .067 .0146  1.000 .067 .0167
v2/50000         1.000 .033 .0121  1.000 .033 .0112  1.000 .033 .0123  1.000 .067 .0165
v2/15000         0.969 .067 .0172  0.946 .067 .0151  1.000 .050 .0161  0.946 .067 .0218
v2/5000          0.984 .100 .0243  0.982 .133 .0198  0.981 .100 .0222  0.982 .133 .0240
v3_cont/15000    1.000 .033 .0119  1.000 .033 .0095  1.000 .033 .0110  1.000 .033 .0139
```

`close_rate` **saturates at 1.0 for every mature checkpoint.** `close_dt` and `approach_l1` do
separate the clearly under-trained 5 000-step checkpoint (0.100 s / 0.024 rad against 0.033 s /
0.012 rad) — the metric has resolution in the "not trained enough" direction — but among mature
checkpoints the spread is inside anything the seed variance is likely to be.

The reason is structural and it applies to **any** offline metric:

> Every checkpoint on disk trained on all 15 recordings. There is no held-out data for any of
> them. **The existing checkpoints are unrankable offline, by this metric or any other**, and no
> amount of metric design changes that.

So the proxy is not a tool for choosing among what exists. It is a tool that only works if the
sweep is *designed* around it — which is why §5 makes a grouped split the first thing every run
does, and why S1.2 (the variance floor) is a prerequisite rather than a nicety.

Four further limits, stated plainly:

1. **It is a single-query metric, not a rollout.** It cannot see compounding error, and the
   O(εT²) argument says compounding is the dominant failure mode in BC. A policy that predicts
   a perfect chunk from the demonstrated state and diverges catastrophically two steps off it
   scores perfectly. **Mitigation:** the state-noise column above is a first-order probe of
   this; widen it to a proper perturbation sweep (±2 cm equivalent) once a holdout exists.
2. **19 held-out windows from 3 recordings is a thin estimate.** `close_rate` on n=19 has a
   ±0.1 standard error at best. Report confidence intervals or do not report the third digit.
3. **`close_dt` inherits the corpus's own labelling.** The threshold 0.45 is the same constant
   that produced the labels; if segmentation mis-labelled a close, the proxy repeats the
   mistake. It is not independent ground truth.
4. **It says nothing about placement.** `AUDIT.md` §S4.1 — "placed" only proves the jaws opened.
   The proxy scores grasping only, which is what the ACT block owns; do not let a good proxy
   score be read as a good cycle.

### 4.5 The validation was attempted on the four production checkpoints. It failed, and the failure is the result.

The ask was: run the proxy against checkpoints whose on-robot behaviour is known, and trust it
only if it ranks them correctly. Done. **It does not, and neither could anything else.**

The full `act_grasp_dark_noise_20260728` ladder (v1, top+wrist, 90-frame windows), scored at
lead 1.0 s on n=71 episodes, with the training loss reconstructed alongside it (128 samples,
train mode, so the KL branch runs and the numbers are on the same scale as the console print):

```
   ckpt   printed   l1_term       kld | close_rate  close_dt  appr_l1    n
   5000    0.2770   0.06317   0.02138 |      1.000     0.067   0.0191   71
  15000    0.0670   0.03821   0.00288 |      1.000     0.033   0.0139   71
  25000    0.0408   0.03045   0.00103 |      1.000     0.033   0.0121   71
  30000    0.0327   0.02611   0.00065 |      1.000     0.033   0.0112   71   ← sort_server's default
  40000    0.0271   0.02583   0.00013 |      1.000     0.067   0.0109   71
  50000    0.0223   0.02183   0.00004 |      1.000     0.033   0.0097   71
```

- `close_rate` is **1.000 at every rung**, including the 5 000-step checkpoint. No resolution.
- `close_dt` alternates between 0.033 and 0.067 s — that is ±1 frame, i.e. quantisation noise.
  No resolution.
- `appr_l1` decreases monotonically and **tracks the loss exactly**. It carries no information
  the loss did not already have.

So the proxy, evaluated the way the ask specified, reproduces the loss ordering and adds
nothing. I am reporting that rather than tuning the weights until the answer comes out right.

**The reason is not the metric. It is that there is nothing to measure.** All six checkpoints
trained on all 77 episodes; the metric is being asked to distinguish six models on data all six
memorised. A 51.6 M-parameter model on 6 785 frames fits its training set, and every offline
statistic computed there saturates. This is the same wall §4.4 hit on the v2 line.

The conclusion generalises, and it is the single most actionable thing in this document:

> **No offline metric can rank checkpoints that trained on all the data. The four production
> checkpoints are unrankable offline — not by loss, not by this proxy, not by anything — and
> so are any new runs that also train on all 69 episodes.** The proxy is not a validated
> instrument yet, and the only experiment that would validate it is one where the eval
> recordings were withheld at training time.

One thing the exercise *did* establish, on the checkpoints themselves rather than by anecdote:

```
v2 scratch 50000          printed 0.0392
v3_cont/015000            printed 0.0364
ft50k/006000  DEPLOYED    printed 0.0509   ← worst loss of the three, and it is the one in production
```

The deployed checkpoint has a **31% higher loss** than a sibling trained on the same data, and
it is the one the operator kept. Likewise `sort_server.py:1771` defaults to v1's `030000`, not
its best-loss `050000`. Two independent instances, measured tonight, of the operator's
rollout-based choice disagreeing with the loss ranking. That corroborates the premise; it does
not rescue the proxy.

> **SUPERSEDED IN PART by §4.7 — kept, not deleted, per the AUTOPILOT rule.** The claim
> "no offline metric can rank the existing checkpoints" is **correct as stated for the v1
> ladder** and correct for any comparison run on training data. It was **too broad**: I had
> overlooked that a genuine holdout already exists on disk. The zone filter dropped 8 grasps
> from v1 when building v2, so those 8 are unseen by every checkpoint in the v2 line. Scored
> there, the proxy does rank them, and it inverts the loss ordering. See §4.7.

### 4.7 The validation, on data that was genuinely held out — the proxy wins, 7/7

The zone filter (`y ≤ -0.13`) is what took `yam_grasp_v1`'s 77 grasps down to
`yam_grasp_v2_wrist`'s 69. **The 8 it removed are therefore training data for the v1 line and
unseen data for the entire v2 line** — including the deployed checkpoint. They sit in
`yam_grasp_v1`, which carries the same wrist camera, so a wrist-only policy can be scored on
them by feeding just the wrist key. Indices written to `sweeps/holdout_v1_indices.json`
(v1 episodes 0, 11, 14, 23, 28, 32, 38, 43; 7 usable — one closes too early for a 1.0 s lead).

Everything below is scored through one pipeline on `yam_grasp_v1`; the only thing that differs
between the two columns is whether the checkpoint trained on that episode.

```
checkpoint               |          UNSEEN (n=7)          |          SEEN (n=64)           | printed
                         |  rate     dt   appr_l1         |  rate     dt   appr_l1         |  loss
v2 scratch  5000         | 1.000  0.300    0.0584         | 0.984  0.133    0.0345         | 0.3169
v2 scratch 20000         | 1.000  0.267    0.0573         | 0.953  0.067    0.0281         | 0.0749
v2 scratch 50000         | 1.000  0.300    0.0608         | 1.000  0.033    0.0263         | 0.0392
ft50k/006000  DEPLOYED   | 1.000  0.133    0.0351         | 1.000  0.033    0.0245         | 0.0509
v3_cont/015000           | 1.000  0.100    0.0442         | 1.000  0.033    0.0230         | 0.0364
```

**The loss and the proxy disagree, and the proxy agrees with the operator.**

- `v2 scratch 50000` has the *better* printed loss (0.0392) and is **worst of the mature
  checkpoints on unseen data** — `close_dt` 0.300 s, `approach_l1` 0.0608.
- `ft50k/006000` — **the checkpoint actually in production** — prints a **31% worse** loss
  (0.0509) and is **2.3× better on unseen close timing** and **42% better on unseen approach**.

Paired, per held-out episode, deployed vs from-scratch:

```
 v1 ep | scratch50k  dt      l1 | DEPLOYED  dt      l1 | winner
     0 |          0.567  0.0502 |       0.133  0.0207 | DEPLOYED
    11 |          0.300  0.0428 |       0.100  0.0336 | DEPLOYED
    14 |          0.300  0.0490 |       0.133  0.0296 | DEPLOYED
    23 |          0.200  0.0796 |       0.067  0.0395 | DEPLOYED
    28 |          0.100  0.0715 |       0.033  0.0504 | DEPLOYED
    38 |          0.367  0.0604 |       0.167  0.0244 | DEPLOYED
    43 |          0.167  0.0718 |       0.133  0.0473 | DEPLOYED
```

**7/7 on `approach_l1`, 7/7 on `close_dt`.** Sign test p ≈ 0.016 two-sided. Not one outlier.

**The generalisation gap is the mechanism, and it is directly readable:**

| checkpoint | seen `appr_l1` | unseen `appr_l1` | gap |
|---|---|---|---|
| v2 scratch 50000 | 0.0263 | 0.0608 | **2.31×** |
| v3_cont/015000 | 0.0230 | 0.0442 | 1.92× |
| **ft50k/006000 (deployed)** | 0.0245 | **0.0351** | **1.43×** |

Training wrist-only from scratch on 69 windows overfits by 2.3×. Warm-starting from the
two-camera v1 pretrain and fine-tuning 6 000 steps overfits by 1.4×, at a *worse* training
loss. That is textbook small-data behaviour, and it is the first quantitative evidence on this
rig that the loss ranking is not merely uninformative but **actively inverted**.

**Honest limits — this is n=7 and the holdout is not random.** The 8 excluded grasps are the
corner band `y > -0.13`, which `PLAN-ACT-V2-ZONE-FILTERED.md` measures at a 21% demonstration
failure rate against 7% in the main band. They are the hardest region, held out *by
construction* rather than at random. So this is a **stress test of out-of-zone generalisation**,
not an unbiased estimate of held-out performance — which arguably makes it the more relevant
question for a policy that meets IK hover error, but it must not be quoted as a general
holdout number. The v1-line checkpoint cannot appear in this table at all: it trained on these
8, and it needs two cameras. The grouped 12/3 split of §5 remains the right instrument for the
sweep; this is the instrument that happened to already exist.

### 4.6 Posterior collapse — measured, and it rewrites two of the queued arms

Reconstructing the loss split (which no run ever logged, W&B being disabled) surfaced something
nobody was looking for. The KL term does not settle — **it collapses**:

| | 5 k | 15–20 k | 50 k |
|---|---|---|---|
| v1 line `kld` | 0.0214 | 0.0029 | **0.00004** |
| v2 line `kld` | 0.0208 | 0.0017 | **0.00015** |

At the deployed `ft50k/006000`, `kld = 0.000044`; times `kl_weight = 10` that is **0.9% of a
0.0509 loss**. The CVAE posterior has gone to the prior: μ→0, σ→1, the latent carries no
information about the demonstration. Since inference already sets z = 0, **`use_vae=true` is
doing nothing at any deployed checkpoint** — it costs a 4-layer transformer encoder at training
time and buys a latent that has stopped encoding.

Two consequences for the queue, and one of them reverses a judgement I made earlier in this
document (§5.1, left standing with the correction attached, per the AUTOPILOT rule):

- **`kl1` is promoted.** I first wrote it off as unresolvable folklore. With collapse measured,
  it is the one arm with a mechanism behind it: `kl_weight = 1.0` is the direct intervention
  against the collapse, and it predicts a *higher* printed loss with a *live* latent. Keep it,
  and read it by `kld` at 20 k, not by the total.
- **`novae` is reclassified**, not from "known-bad" to "good" but from "known-bad" to
  "well-posed control". My §5.1 objection cited the ACT paper's 35.3% → 2% CVAE ablation on
  human data — that stands as literature, but it is measured at *their* training length and
  cannot explain a latent that is already dead here. The honest prediction is now
  **`novae` ≈ `base`**, and if it is not, the difference localises what the VAE encoder is
  actually contributing (a training-time regulariser on the decoder, not a latent). Keep it.

Neither of these can be read from the total loss, and neither can be read at all if the runs
have no holdout. Log `kld` per arm.

**The proxy still needs validating against rollouts.** The only honest way: run T0.1/T0.2 and the
sweep winners on hardware, log ≥30 cycles each with `best_lift_mm` (the physically-grounded
number, not `result: placed`), and correlate. Note that `results/sort_runs/*.json` **does not
record which checkpoint produced the run** — `sort_server.py:638` puts it in `tl.meta` and it
never reaches the summary. **That is a two-line fix and it should land before the next rollout**,
or this correlation can never be computed. There are ~380 logged cycles on disk and not one of
them can be attributed to a checkpoint.

---

## 5. Concrete run list

### Ground rules for every command

- **`outputs/` is a symlink into the live production tree.** Every run below writes to
  `sweeps/runs/` (a real directory in `kitting-v2`). Never `--output_dir=outputs/…`.
- **`LEROBOT_PREDECODED_ROOT` is mandatory, not an optimisation.** `torchcodec` cannot load on
  this box (libavutil missing for every FFmpeg it supports), so without the predecoded path
  training **dies at the first batch**. `train_act_dark_noise.py` imports
  `~/Desktop/lab/lerobot-fast/predecoded_patch.py` when the variable is set, and refuses to run
  if the patch is missing. Both exist: `yam_grasp_v2_wrist` (638 MB) and `yam_grasp_v1` (1.7 GB).
- **Never `uv run`** — it re-resolves to a cu126 torch with no sm_120 kernels and the policy
  silently loses CUDA. `.venv/bin/python` only.
- **Both `chunk_size` and `n_action_steps` must be set together** or `__post_init__` raises.
- **Check `df -h` before each run** and prune after each (recipe below).

### The grouped split

Written tonight to `sweeps/split_group3.json`. Three held-out recordings, **one per recording
day**, so the holdout spans the real distribution shift (session lighting / mat layout), not
just adjacent grasps from the same take:

```
holdout: episode_181321_dfa58b86 (0709)  episode_220623_6f9a09d4 (0726)  episode_140454_648f75b2 (0727)
train:  50 episodes / 12 recordings      eval: 19 episodes / 3 recordings

--dataset.episodes='[0,1,2,3,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                     33,34,35,36,37,38,39,40,46,47,48,49,50,51,52,53,54,55,56,63,64,65,66,67,68]'
```

**Known leak, accepted:** LeRobot's normalisation stats come from `meta/stats.json`, computed
over all 69 episodes regardless of `--dataset.episodes` (this is the mechanism behind LeRobot
issue #2851, where a validation split made policies freeze). The leak is mean/std only, not
behaviour. Verify after the first run that the policy actually moves; if it does not, that
issue is the first place to look.

### Disk arithmetic

One checkpoint = 207 MB `model.safetensors` + 413 MB `optimizer_state.safetensors` + ~16 MB
= **621 MB**. At `save_freq=10000` over 50 000 steps that is 5 checkpoints = **3.1 GB peak per
run**. After pruning `training_state` from all but the last: **1.45 GB resident per run**.

```bash
# prune, run by run, AFTER the proxy has been computed on that run's ladder
LAST=$(readlink sweeps/runs/<name>/checkpoints/last)
for d in sweeps/runs/<name>/checkpoints/*/; do
  [ "$(basename $d)" = "$LAST" ] || rm -rf "$d/training_state"
done
```

Wall clock: measured from the existing runs, wrist-only at batch 8 is **~18.5 steps/s**, so
50 000 steps ≈ **45 min**. GPU time is not the constraint. Disk and the 15-scene corpus are.

### The list, ordered by expected value

| # | run | steps | why here | resident GB |
|---|---|---|---|---|
| 0a | *(no run)* T0.3 manifest + split | — | done tonight; `sweeps/split_group3.json` | 0 |
| 0b | *(no run)* T0.4 clock probe | — | **gates run 8**; offline, minutes | 0 |
| 1 | `seed1000` — baseline on the split | 50k | control arm for everything | 1.45 |
| 2 | `seed2000` | 50k | **variance floor.** Without runs 2–3 no other comparison is interpretable | 1.45 |
| 3 | `seed3000` | 50k | " | 1.45 |
| 4 | `chunk60` | 50k | time-matches ACT's 2.0 s; supervised fraction 0.651 → 0.827 | 1.45 |
| 5 | `chunk30` | 50k | aggressive end; 0.898 supervised. Below this the ACT ablation says do not go | 1.45 |
| 6 | `ft_from_v1_50k` | 6k | reproduces the deployed lineage **on the split**, so it is finally comparable | 0.83 |
| 7 | `randomcrop` | 50k | only lever that manufactures diversity from 15 scenes; no ACT ablation exists anywhere | 1.45 |
| 8 | `twocam` (gated on 0b) | 50k | every published camera ablation ranks wrist-only last; here it was confounded 4 ways. Needs a two-camera v2 export + predecode first (~1.5 GB) | 1.45 + 1.5 |
| 9 | `chunk60_seed2000` | 50k | only if run 4 beats the run 1–3 band; confirms it is not a seed | 1.45 |

**Total resident ≈ 14.4 GB** (13 GB without run 8's export), peak transient +1.7 GB during any
single run. Against 62 GB free that is comfortable with room for a second fold. Runs 1–7 are
~5.5 GPU-hours.

Run 8's export and predecode must happen before it and are the only steps that touch
`export_lerobot.py`; note `AUDIT.md` §S7.1 — the bad-take filter looks for the tag `"x"` which
can never exist, so it has never fired. It does not damage the *current* corpus (no episode on
disk carries `bad`), but it should be fixed before any re-export.

### Commands

```bash
cd ~/Desktop/kitting-v2/rl-teleop
export PD=$HOME/.cache/lerobot-predecoded/yam_grasp_v2_wrist
export TRAIN='[0,1,2,3,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,46,47,48,49,50,51,52,53,54,55,56,63,64,65,66,67,68]'

# ---- runs 1-3 : the variance floor -----------------------------------------
for S in 1000 2000 3000; do
  LEROBOT_PREDECODED_ROOT=$PD .venv/bin/python tools/train_act_dark_noise.py \
    --dataset.repo_id=ETHRC/yam_grasp_v2_wrist --dataset.episodes="$TRAIN" \
    --policy.type=act --policy.device=cuda \
    --steps=50000 --save_freq=10000 --batch_size=8 --seed=$S \
    --output_dir=sweeps/runs/seed$S 2>&1 | tee sweeps/runs/seed$S.log
done

# ---- runs 4-5 : chunk_size (BOTH flags, or __post_init__ raises) -----------
for C in 60 30; do
  LEROBOT_PREDECODED_ROOT=$PD .venv/bin/python tools/train_act_dark_noise.py \
    --dataset.repo_id=ETHRC/yam_grasp_v2_wrist --dataset.episodes="$TRAIN" \
    --policy.type=act --policy.device=cuda \
    --policy.chunk_size=$C --policy.n_action_steps=$C \
    --steps=50000 --save_freq=10000 --batch_size=8 --seed=1000 \
    --output_dir=sweeps/runs/chunk$C 2>&1 | tee sweeps/runs/chunk$C.log
done

# ---- run 6 : the deployed lineage, on the split ----------------------------
LEROBOT_PREDECODED_ROOT=$PD .venv/bin/python tools/train_act_dark_noise.py \
  --dataset.repo_id=ETHRC/yam_grasp_v2_wrist --dataset.episodes="$TRAIN" \
  --policy.type=act --policy.device=cuda \
  --policy.pretrained_path=outputs/pretrained/act_50k_wristonly \
  --steps=6000 --save_freq=1000 --batch_size=8 --seed=1000 \
  --output_dir=sweeps/runs/ft_from_v1_50k 2>&1 | tee sweeps/runs/ft_from_v1_50k.log

# ---- run 7 : + random crop -------------------------------------------------
#   needs a `dark_noise_crop()` recipe in tools/act_aug.py adding a RandomCrop /
#   RandomResizedCrop(scale=(0.90,1.0)) entry, and a --aug flag on the trainer.
#   Stream E owns tools/train_* and tools/act_aug.py; this is the only code
#   change any run in this list requires.

# ---- scoring : after EVERY run, on the checkpoint ladder, not just `last` ---
LEROBOT_PREDECODED_ROOT=$PD .venv/bin/python sweeps/grasp_proxy.py \
  sweeps/runs/*/checkpoints/*/pretrained_model
```

`grasp_proxy.py` currently scores whatever episodes it is handed; **before run 1 it needs its
`groups` argument wired to the eval list in `split_group3.json`** so it scores the 19 held-out
episodes rather than all 69. That is a ten-line change and it is the difference between the
metric working and the saturation shown in §4.4.

### 5.1 Against the queue that already exists

`sweeps/queue.json` was built independently while this was being written. Read against the
above, four notes — one of them load-bearing:

1. **No queued run holds anything out.** Every entry trains on all 69 episodes. That means
   **no run in the queue can be ranked by any offline metric**, including the one in §4 —
   §4.4 shows the metric saturates on training data. Adding `--dataset.episodes="$TRAIN"`
   (§5) to every entry is a one-line change per run and is the difference between a sweep
   that produces a ranking and a sweep that produces nine unrankable checkpoints.
2. **`novae` is a known-bad experiment.** The ACT paper's own CVAE ablation is 35.3% → 2% on
   *human* demonstrations (no effect on scripted data). All our data is human GELLO teleop.
   The queue note reasons that "with a single scripted approach there may be little
   multimodality" — but the demonstrations are not scripted, and a floppy bag has many valid
   grasp points. Recommend dropping this arm; the slot is better spent on random crop.
3. **`kl1` and `lr3e5` are unresolvable at this corpus size.** No published sweep of
   `kl_weight` exists anywhere (§2.7), and `lr=1e-5` is the published ACT value that all four
   existing runs used with grad-norms at a third of the clip threshold. With seed variance
   unmeasured and a 19-window holdout, neither can be distinguished from noise. Keep them only
   if slots are spare after the split is applied.
4. `base` at 20 000 steps is a reasonable control, but it is not "the production config
   exactly" — production is `act_wrist_zone_v2_ft50k/006000`, a **6 000-step fine-tune of a
   two-camera 50 000-step model** (§1.2). Run 6 in §5 is the arm that actually reproduces it.

`chunk50` / `chunk30` / `base_seed2` / `frozen_backbone` agree with the ranking above and
should stay. `chunk150` is a legitimate opposite bet; note it drops the supervised fraction to
~0.51 and moves the loss scale again, so it must be judged by the proxy, never by loss.

> **CORRECTION, after the measurement in §4.6 — not deleted, per the AUTOPILOT rule.** Points 2
> and 3 above were written before I reconstructed the loss split. Measuring posterior collapse
> (`kld` → 4.4e-5 at the deployed checkpoint) **promotes `kl1` from folklore to the
> best-motivated arm in the queue**, and reclassifies `novae` from "known-bad" to "well-posed
> control with a clear null prediction". Both should stay, and both must be read by their `kld`
> and proxy values rather than by total loss. Point 1 (no holdout) is unaffected and is still
> the load-bearing objection. `lr3e5` remains unmotivated.

### 5.2 Second round — what to run after the nine

Ordered by expected value. The first entry is not optional: without it none of the nine in
flight can be ranked by anything except a rollout.

| # | id | change vs `base` | why |
|---|---|---|---|
| **R0** | `ft_from_v1` | `--policy.path=<rebased v1 50k>`, 6 k steps, on the split | **Now the top-ranked arm.** §4.7 measures the warm start generalising **2.3× → 1.4×** better than from-scratch, winning 7/7 paired on held-out grasps, at a 31% *worse* training loss. That is the single strongest signal in this document and it says the nine from-scratch runs in flight are exploring the weaker branch. Cheapest run in the table (6 k steps ≈ 5 min). |
| **R1** | `base_split`, `chunk50_split`, `chunk30_split`, `kl1_split` | add `--dataset.episodes="$TRAIN"` (§5) | **The whole point.** Re-run the four most informative arms on the 50-episode / 12-recording split so the proxy has 19 unseen windows to score. Four runs × 24 min = 1.6 h, ~3.3 GB. Everything else in this table is worthless without it. |
| **R1b** | `ft_from_v1_chunk50`, `ft_from_v1_20k` | warm start × the best chunk arm; and a longer fine-tune | Only after R0 and R1 confirm the warm start holds on the grouped split. Crosses the two effects that actually have evidence behind them. |
| **R2** | `base_seed3` | seed 3000, on the split | Two seeds bound the variance; three give it a usable spread. Cheap, and it sets the bar every treatment must clear. |
| **R3** | `randomcrop` | `dark_noise` + RandomResizedCrop(scale 0.90–1.0), on the split | The one axis with **no published ACT ablation anywhere** (§2.7) and the only lever that manufactures diversity from 15 scenes. Requires a `dark_noise_crop()` recipe in `tools/act_aug.py` — the only code change any run needs. |
| **R4** | `kl0p1` | `kl_weight=0.1`, on the split | Only if `kl1` shows a live latent at 20 k (`kld` ≫ 1e-3). Brackets the collapse threshold. Skip if `kl1` also collapses. |
| **R5** | `twocam` | top+wrist on the same 69 zone-filtered grasps, same 150-frame windows | Every published camera ablation ranks wrist-only last (§2.5) and the switch here was confounded four ways. **Gate on T0.4** (the clock-domain probe): if `camera_top` frames are being silently dropped, the export is corrupt and the run is wasted. Needs a fresh two-camera export + predecode (~1.5 GB) and runs ~4× slower. |
| **R6** | `chunk50_seed2` | confirmation arm | Only if `chunk50` clears the seed band from R2. Confirms a winner is not a seed. |

**Not recommended for round two**, with reasons already in §3.3: `lr` variants beyond `lr3e5`,
`weight_decay`, any LR schedule, `dim_model` / heads / layer counts, `latent_dim`, resnet34/50,
`n_action_steps` (deploy knob → T0.1), `temporal_ensemble_coeff` (deploy knob → T0.2).

Total for R1–R3 + R6: **6 runs, ~2.5 h, ~5 GB.** Comfortable.

---

### 5.3 `tools/rebase_checkpoint_cameras.py` — verified, and it should be committed

The tool is **untracked** (`git ls-files` → "Did you forget to 'git add'?"), which is how it
came to be mistaken for stray kinematics code. It is neither stray nor kinematics: it reads a
checkpoint's `config.json`, drops the `VISUAL` entries not in `--keep`, copies the directory,
and rewrites `config.json` (and `train_config.json` for provenance, stamping `_rebased_from` /
`_rebased_dropped`). It refuses to overwrite an existing destination. It touches no weights.

Its load-bearing claim — *"ACT has zero camera-keyed tensors"* — I verified independently
against the v1 50 k checkpoint: **234 tensors, none whose name mentions any camera.** There is
one shared `model.backbone.*`, one shared `model.encoder_img_feat_input_proj`, and
`model.encoder_1d_feature_pos_embed` is `(2, 512)` — latent + state, not cameras. Camera
position encoding is sinusoidal and computed, not stored. So camera count changes only the
*number of vision tokens* entering the encoder, never a weight shape, and the rebase is a
config edit rather than surgery. The docstring is accurate.

It is also the tool that produced `outputs/pretrained/act_50k_wristonly` (§1.2), so the
technique is already proven on this rig. **Recommend `git add` it** — the deployed checkpoint
cannot be reproduced without it, which is half of `AUDIT.md` §S2's complaint.

Regeneration command, for R0:

```bash
.venv/bin/python tools/rebase_checkpoint_cameras.py \
  --src outputs/train/act_grasp_dark_noise_20260728/checkpoints/050000/pretrained_model \
  --dst sweeps/pretrained/act_50k_wristonly \
  --keep observation.images.wrist
# then:  --policy.path=sweeps/pretrained/act_50k_wristonly
```

Note `--dst` under `sweeps/`, not `outputs/` — `outputs/` is the live production symlink, and
the tool refuses to overwrite, so pointing it at the existing path would simply fail.

---

## 6. Things found in passing that other streams should know

- **`AUDIT.md` §S1.3's hypothesis does not reach the ACT policy.** [measured] Zero of 69 (and
  zero of 77) exported episodes have a degenerate gripper channel; every episode has a full
  close-open swing (min per-episode range 0.827, median 1.000) and 40% of frames sit below the
  close threshold. The `normalize_width` corruption damages the **grasp corpus / witness**, not
  the ACT training tensors. The audit explicitly labelled this a hypothesis worth testing; it
  is now tested and negative.
- **`AUDIT.md` §S7.2 (time-axis compression from skipped stale frames) has not measurably
  fired.** [measured] The largest single-step joint delta anywhere is **0.074 rad** in
  `yam_grasp_v2_wrist` and **0.045 rad** in `yam_grasp_v1`; **zero** steps exceed 0.10 rad in
  either. A 0.5 s dropout would show as a 10–20× outlier. The mechanism is real; its incidence
  in the deployed datasets is nil.
- **The gripper action channel is min-max normalised per *recording*** (`export_lerobot.py:418-419`,
  over the whole episode before windowing). So the policy's index-6 output is not in a
  consistent physical unit across training episodes, and `act_runner.py` publishes it directly
  as a 0..1 bus command. This is a train/deploy unit question for Stream C, and it is why the
  proxy in §4.3 measures crossing *times* rather than widths.
- **`results/sort_runs/*.json` does not record the checkpoint.** `sort_server.py:638` writes it
  into `tl.meta` and the summary never carries it. ~380 logged cycles, none attributable. Two
  lines. Without it the proxy can never be validated against rollouts.
- **Inference costs 6.2 ms** (§2.4). Anything in the control loop that claims it cannot afford
  another observation is not being limited by the policy.

---

## 7. Provenance

| section | inherited from `ACT_RESEARCH_GUIDELINES.md` | new tonight |
|---|---|---|
| §1 four-run diff, lineage | — | all [measured] |
| §2.1 version audit | the claims | the 0.4.3 verification, the adopted/ignored table |
| §2.2 upstream defaults | partially | confirmed against installed source + upstream main |
| §2.3 action horizon | F1's *recommendation* (12–20) | ACT Fig 8a numbers, LeRobot PR #319 table — **which contradict F1** |
| §2.4 latency | F1 asked for it | measured |
| §2.5 cameras | F2 (Hsu et al.) | AV-ALOHA 7-way ablation, Unitree G1 study — **which contradict wrist-only** |
| §2.6 small data | F5 (Lin et al.) | ACT's own 50-demo scale, LeRobot's ≥50/≥10-per-location guidance |
| §2.7 aug / KL | F6 | confirmation that **no** ACT augmentation ablation exists; no `kl_weight` sweep exists |
| §2.8 offline metrics | F3 (robomimic) | robomimic's exact "50 to 100% worse" quote; CI-MSE r/ρ numbers; LeRobot issues #250, #2851 |
| §3 ranked plan, do-not-sweep list | ladder ordering | the justifications, all from measured corpus properties |
| §4 proxy + negative result | F3's "loss is not a selection signal" | the no-val-loss finding, the padding mis-scaling, the metric, the prototype, the saturation result |
| §5 run list | — | all |
| §6 | — | all [measured] |

Files written by this stream: `rl-teleop/sweeps/RESEARCH.md` (this file),
`rl-teleop/sweeps/grasp_proxy.py`, `rl-teleop/sweeps/split_group3.json`. Nothing else was
modified anywhere.

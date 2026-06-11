# Dubins-car World Model from the offline `(s_t, o_t, a_t)` dataset

These files train this repository's DreamerV3 latent **world model** (the latent
**transition function** + the **failure classifier**) directly from the image
observations in the offline dataset produced by `dubins_data_generation.py`.
They **reuse the repo's code unmodified** — nothing under `dreamerv3_torch/`,
`scripts/`, or `PyHJ/` is edited.

## Files

| File | What it does |
|------|--------------|
| `dubins_dataset.py` | Adapter: reads the flat replay-buffer `.npz` from `dubins_data_generation.py`, slices it back into episodes, and emits per-episode dicts with exactly the keys `models.WorldModel` expects (`image`, `obs_state=[cosθ,sinθ]`, `privileged_state=(x,y,θ)`, `failure`, `action`, `is_first/last/terminal`, …). Uses the dataset's `gt_failure` so the failure classifier learns **both** failure modes (center obstacle **and** the `|y|>0.8` sidewalks). |
| `train_dubins_wm.py` | Driver: builds the config from `configs.yaml`, sets workspace/obstacle/image-size to match the dataset, constructs the repo's `Dreamer` agent, and runs its `pretrain_model_only` loop. Saves `rssm_ckpt.pt`. |

## What gets trained (all from `models.WorldModel`)

- **encoder** (`MultiEncoder`): image + `cos/sin(θ)` → embedding
- **RSSM dynamics** (`RSSM`): `z_t, a_t → ẑ_{t+1}` — the latent **transition function**
- **decoder** (`MultiDecoder`): latent → reconstruct image + `obs_state`
- **cont head**: latent → episode continuation
- **margin head**: latent → safety margin `g(z)` — the **failure classifier**,
  hinge loss on the dataset failure labels (margin > 0 safe, < 0 unsafe)

Observation convention matches the repo: the vector input reveals only the
heading (`cos/sin θ`); planar position is inferred from the image — so the WM
architecture/config are unchanged.

## How to run

```bash
conda activate anysafe          # the repo's environment (has gym, torch+CUDA, etc.)

# 1) generate the offline dataset (attached generator)
python dubins_data_generation.py --num-steps 30000 --img-size 64 --out dubins_dataset.npz

# 2) train the world model
python dubins_world_model/train_dubins_wm.py \
    --dataset dubins_dataset.npz \
    --steps 10000 \
    --device cuda:0
```

Output: `rssm_ckpt.pt` (+ `best_rssm_ckpt_*.pt`) under
`logs/dreamer_dubins/<wm_name>/` — the same location/name the rest of the repo
(e.g. `train_failure_classifier_sem_dubins.py`, `run_training_ddpg_wm.py`)
expects. Override with `--logdir`.

Useful flags: `--batch-size`, `--batch-length`, `--eval-every`, `--val-frac`,
`--video-pred` (log open-loop reconstructions), `--compile` (torch.compile).

Inspect the converted dataset without training:

```bash
python dubins_world_model/dubins_dataset.py --dataset dubins_dataset.npz
```

## Notes / caveats

- The driver uses a tiny `PrintLogger` instead of the repo's wandb `Logger`, so
  no wandb account/login is needed. Swap in `tools.Logger` if you want the
  dashboards and video panels.
- Image side length is read from the dataset and pushed into `config.size`; the
  RSSM CNN supports any power-of-two size (64 → 4 conv stages, 128 → 5).
- Only the world-model **pre-training** stage is covered here. Downstream latent
  reachability RL (`run_training_ddpg_wm.py`) and conformal calibration
  (`dubins_cp.py`) are unchanged and out of scope.

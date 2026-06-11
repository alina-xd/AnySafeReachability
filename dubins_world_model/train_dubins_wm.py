"""
Train the DreamerV3 Dubins-car *world model* (latent dynamics / transition
function + failure classifier) directly from image observations, using the
offline dataset produced by `dubins_data_generation.py`.

What gets trained (all reusing this repo's code, nothing in it is modified)
---------------------------------------------------------------------------
We instantiate the repository's `Dreamer` agent (from `scripts/dreamer_offline.py`)
and run its `pretrain_model_only` loop. That single loop jointly optimizes the
components of `models.WorldModel`:

  * encoder            (networks.MultiEncoder)  image + cos/sin(theta) -> embedding
  * RSSM dynamics      (networks.RSSM)          the latent transition function
                                                z_t, a_t -> hat z_{t+1}  (the "world model")
  * decoder            (networks.MultiDecoder)  latent -> reconstruct image + obs_state
  * cont head          (networks.MLP)           latent -> episode continuation
  * margin head        (networks.MLP)           latent -> safety margin g(z)
                                                = the FAILURE CLASSIFIER, trained with a
                                                hinge loss on the dataset's failure labels
                                                (positive margin = safe, negative = unsafe)

The transition function is what "Dreamer with the Dubins car" needs first: a
model that, given a latent state and a yaw-rate action, predicts the next latent
(and can be decoded back to an image). The margin head is the failure / safety
classifier learned in the same latent space.

This script only does the world-model pre-training stage (`rssm_train_steps`);
the downstream latent-reachability RL (`run_training_ddpg_wm.py`) is out of scope
and unchanged.

Usage
-----
    python dubins_world_model/train_dubins_wm.py \
        --dataset dubins_dataset.npz \
        --steps 10000 \
        --device cuda:0

Outputs `rssm_ckpt.pt` (and `best_rssm_ckpt_*.pt`) under
`logs/dreamer_dubins/<wm_name>/`, the same location/name the rest of the repo
expects. Use `--logdir` to override.
"""

import argparse
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import ruamel.yaml as yaml

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
# make the repo's packages and the scripts/ modules importable
for p in (str(_REPO), str(_REPO / "dreamerv3_torch"), str(_REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import tools
from tqdm import trange

import dubins_dataset  # local adapter (same folder)

to_np = lambda x: x.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Minimal logger (same interface the Dreamer/WorldModel expect, no wandb needed)
# ----------------------------------------------------------------------------
class PrintLogger:
    def __init__(self, step=0):
        self.step = step

    def scalar(self, name, value):
        pass

    def image(self, name, value):
        pass

    def video(self, name, value):
        pass

    def write(self, fps=False, step=False, **kwargs):
        pass


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    return tools.from_generator(generator, config.batch_size)


# ----------------------------------------------------------------------------
# Config construction (mirrors the bottom of scripts/dreamer_offline.py)
# ----------------------------------------------------------------------------
def build_config(overrides, remaining_argv):
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load((_REPO / "configs.yaml").read_text())
    defaults = dict(configs["defaults"])
    defaults.update(overrides)

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining_argv)


def main():
    import gym

    pre = argparse.ArgumentParser(add_help=True)
    pre.add_argument("--dataset", required=True, help="path to the .npz dataset")
    pre.add_argument("--steps", type=int, default=None,
                     help="RSSM training steps (default: configs.yaml rssm_train_steps)")
    pre.add_argument("--device", type=str, default=None,
                     help="torch device, e.g. cuda:0 or cpu (default: configs.yaml)")
    pre.add_argument("--logdir", type=str, default=None,
                     help="override checkpoint/output directory")
    pre.add_argument("--batch-size", type=int, default=None)
    pre.add_argument("--batch-length", type=int, default=None)
    pre.add_argument("--eval-every", type=int, default=None)
    pre.add_argument("--val-frac", type=float, default=0.05)
    pre.add_argument("--compile", action="store_true",
                     help="torch.compile the model (off by default for portability)")
    pre.add_argument("--video-pred", action="store_true",
                     help="log open-loop video reconstructions at eval time")
    args, remaining = pre.parse_known_args()

    # image size must match the dataset the model is trained on
    img_size = dubins_dataset.dataset_image_size(args.dataset)

    # ---- overrides applied on top of configs.yaml defaults --------------------
    # workspace / obstacle constants match dubins_data_generation.py
    overrides = {
        "size": [img_size, img_size],
        "x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0,
        "obs_x": 0.0, "obs_y": 0.0, "obs_r": 0.5,
        "speed": 0.5, "turnRate": 1.5, "dt": 0.05,
        "compile": bool(args.compile),
        "video_pred_log": bool(args.video_pred),
    }
    if args.steps is not None:
        overrides["rssm_train_steps"] = args.steps
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.batch_length is not None:
        overrides["batch_length"] = args.batch_length
    if args.eval_every is not None:
        overrides["eval_every"] = args.eval_every

    config = build_config(overrides, remaining)

    tools.set_seed_everywhere(config.seed)
    config = tools.set_wm_name(config)  # sets config.logdir / rssm_ckpt_path / wm_name
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(args.logdir).expanduser() if args.logdir \
        else pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # ---- observation / action spaces (must match the WM encoder config) -------
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    image_space = gym.spaces.Box(
        low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8
    )
    obs_state_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    dict_obs_space = {}
    if "image" in config.encoder["cnn_keys"]:
        dict_obs_space["image"] = image_space
    if "obs_state" in config.encoder["mlp_keys"]:
        dict_obs_space["obs_state"] = obs_state_space
    observation_space = gym.spaces.Dict(dict_obs_space)

    config.num_actions = action_space.shape[0]
    print("Observation space:", observation_space)
    print("Action space:", action_space)

    # ---- datasets -------------------------------------------------------------
    import collections

    expert_eps = collections.OrderedDict()
    n_tr = dubins_dataset.fill_cache_from_npz(
        args.dataset, expert_eps, val_frac=args.val_frac, is_val_set=False, seed=config.seed
    )
    expert_val_eps = collections.OrderedDict()
    n_va = dubins_dataset.fill_cache_from_npz(
        args.dataset, expert_val_eps, val_frac=args.val_frac, is_val_set=True, seed=config.seed
    )
    print(f"Train episodes: {n_tr}   Val episodes: {n_va}")
    expert_dataset = make_dataset(expert_eps, config)
    eval_dataset = make_dataset(expert_val_eps if n_va > 0 else expert_eps, config)

    # ---- agent (reuses this repo's Dreamer / WorldModel) ----------------------
    from dreamer_offline import Dreamer

    logger = PrintLogger(step=0)
    agent = Dreamer(observation_space, action_space, config, logger, expert_dataset).to(
        config.device
    )
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        print("Resuming from", logdir / "latest.pt")
        ckpt = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        agent._should_pretrain._once = False

    # ---- lightweight obs-reconstruction eval (used as checkpoint score) -------
    def eval_obs_recon():
        recon_steps = 51
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
        eval_loss = []
        for i in range(recon_steps):
            if i % 10 == 0:
                eval_loss.append(
                    agent.pretrain_regress_obs(next(eval_dataset), obs_mlp, obs_opt, eval=True)
                )
            else:
                agent.pretrain_regress_obs(next(expert_dataset), obs_mlp, obs_opt)
        del obs_mlp, obs_opt
        return float(np.min(eval_loss))

    # ---- pre-training loop ----------------------------------------------------
    total_steps = config.rssm_train_steps
    assert total_steps > 0, "rssm_train_steps must be > 0"
    print(f"Pre-training the RSSM world model for {total_steps} steps ...")
    best = float("inf")
    for step in trange(total_steps, desc="RSSM pretrain", ncols=0):
        if ((step + 1) % config.eval_every == 0) or step == 1:
            score = eval_obs_recon()
            print(f"\n[step {step}] obs-recon eval loss: {score:.5f}")
            if config.video_pred_log:
                try:
                    openl = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(openl))
                except Exception as e:
                    print("video_pred skipped:", e)
            best = tools.save_checkpoint("rssm_ckpt", step, score, best, agent, logdir)
        agent.pretrain_model_only(next(expert_dataset), step)

    # final save
    tools.save_checkpoint("rssm_ckpt", total_steps - 1, None, best, agent, logdir)
    print("Done. Checkpoint saved to", logdir / "rssm_ckpt.pt")


if __name__ == "__main__":
    main()

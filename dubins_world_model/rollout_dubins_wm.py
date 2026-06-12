"""
Roll out and visualize trajectories from a *trained* DreamerV3 Dubins-car world
model (the checkpoint produced by `train_dubins_wm.py`).

What this does
--------------
Given a trained checkpoint and the offline dataset, for each evaluation episode
it performs an **open-loop latent rollout**:

  1. Feed the first `--context` real frames (image + cos/sin theta) through the
     encoder + RSSM `obs_step` to obtain the posterior latent at the end of the
     context window (this is the model "watching" the start of the episode).
  2. From that latent, **imagine forward** with the episode's recorded actions
     using only the RSSM transition function `img_step` -- no further image
     observations are used. This is the world model predicting the future.
  3. Decode every latent back to (a) an image (decoder CNN head), (b) the
     observed heading cos/sin theta (decoder MLP head), and (c) the full
     privileged state (x, y, theta) via a small latent->state probe regressor,
     and evaluate the learned safety margin g(z) (margin head).

It then writes several visualizations under `--outdir`:

  * `filmstrip_ep*.png`  : ground-truth vs decoded-imagined frames (+ abs error)
                           along the rollout, with the context/imagined split
                           marked.
  * `trajectory_grid.png`: predicted vs ground-truth (x, y) paths drawn on the
                           Dubins workspace (center obstacle + sidewalks), one
                           panel per episode, colored by the learned margin.
  * `state_error.png`    : open-loop position error vs rollout horizon, averaged
                           over the evaluation episodes (with per-step x/y/theta
                           traces for one example episode).
  * `rollout_ep*.gif`    : (optional, --gif) animated truth|prediction film.

Nothing in the repository is modified -- the world model, RSSM, decoder and
margin head are all loaded from the trained checkpoint and used through their
existing APIs (`encoder`, `dynamics.obs_step/img_step/get_feat`,
`heads['decoder']`, `heads['margin']`). The latent->state probe is a tiny MLP
trained here only so we can *plot* the imagined latents in world coordinates;
it does not touch the world model's weights.

Usage
-----
    conda activate anysafe
    python dubins_world_model/rollout_dubins_wm.py \
        --dataset dubins_dataset.npz \
        --device cuda:0 \
        --num-episodes 6 --context 5 --gif

By default the checkpoint is read from the same location `train_dubins_wm.py`
saves it (`logs/dreamer_dubins/<wm_name>/rssm_ckpt.pt`); override with --ckpt.
"""

import argparse
import collections
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
# repo root (for `import dreamerv3_torch`), scripts/ (for dreamer_offline) and
# this folder (for the local adapter / train driver helpers).
for p in (str(_REPO), str(_REPO / "scripts"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

# Same module-aliasing dance as train_dubins_wm.py: the world-model package uses
# relative imports, but scripts/dreamer_offline.py imports them by bare name.
from dreamerv3_torch import (
    models as _models,
    tools as tools,
    networks as _networks,
    exploration as _exploration,
)

for _name, _mod in (
    ("models", _models),
    ("tools", tools),
    ("networks", _networks),
    ("exploration", _exploration),
):
    sys.modules.setdefault(_name, _mod)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

import dubins_dataset  # local adapter (same folder)
from train_dubins_wm import build_config, make_dataset  # reuse the driver helpers

# environment geometry for drawing the background (same constants the generator used)
from dubins_data_generation import R_OBS, SIDEWALK_Y, WORKSPACE

to_np = lambda x: x.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Agent construction (mirrors train_dubins_wm.main, inference-only)
# ----------------------------------------------------------------------------
def build_agent(args, remaining):
    import gym
    from dreamer_offline import Dreamer

    img_size = dubins_dataset.dataset_image_size(args.dataset)

    overrides = {
        "size": [img_size, img_size],
        "x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0,
        "obs_x": 0.0, "obs_y": 0.0, "obs_r": 0.5,
        "speed": 0.5, "turnRate": 1.5, "dt": 0.05,
        "compile": False,
        "video_pred_log": False,
    }
    if args.device is not None:
        overrides["device"] = args.device

    config = build_config(overrides, remaining)
    tools.set_seed_everywhere(config.seed)
    config = tools.set_wm_name(config)

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

    # a (tiny) dataset is needed only to construct the agent; we re-make proper
    # generators below for the obs-probe training.
    episodes = dubins_dataset.episodes_from_npz(args.dataset)
    boot = collections.OrderedDict((f"boot_{i}", e) for i, e in enumerate(episodes[:4]))
    boot_dataset = make_dataset(boot, config)

    from train_dubins_wm import PrintLogger

    agent = Dreamer(observation_space, action_space, config, PrintLogger(0), boot_dataset)
    agent = agent.to(config.device)
    agent.requires_grad_(False)

    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else pathlib.Path(config.logdir) / "rssm_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Train first with "
            f"train_dubins_wm.py or pass --ckpt."
        )
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=config.device)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    agent.eval()
    return agent, config, img_size


# ----------------------------------------------------------------------------
# Latent -> privileged-state probe (for plotting imagined latents in world frame)
# ----------------------------------------------------------------------------
def train_obs_probe(agent, config, episodes, steps, val_frac=0.1, seed=0):
    """Reuse the repo's `pretrain_regress_obs` to fit a latent->(x,y,theta) MLP.

    This is purely a *read-out* so imagined latents can be drawn on the map; it
    leaves the world model untouched (the probe is a separate module).
    """
    n_eps = len(episodes)
    n_val = max(1, int(round(n_eps * val_frac)))
    train_eps = collections.OrderedDict(
        (f"tr_{i}", e) for i, e in enumerate(episodes[:-n_val])
    )
    val_eps = collections.OrderedDict(
        (f"va_{i}", e) for i, e in enumerate(episodes[-n_val:])
    )
    train_ds = make_dataset(train_eps, config)
    val_ds = make_dataset(val_eps, config)

    obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
    print(f"Training latent->state probe for {steps} steps ...")
    best = float("inf")
    for i in range(steps):
        agent.pretrain_regress_obs(next(train_ds), obs_mlp, obs_opt)
        if (i + 1) % max(1, steps // 10) == 0:
            vloss = agent.pretrain_regress_obs(next(val_ds), obs_mlp, obs_opt, eval=True)
            best = min(best, vloss)
            print(f"  probe step {i + 1:5d}/{steps}   val MSE {vloss:.5f}")
    print(f"Probe trained (best val MSE {best:.5f}).")
    obs_mlp.eval()
    return obs_mlp


# ----------------------------------------------------------------------------
# Open-loop rollout of a single episode
# ----------------------------------------------------------------------------
@torch.no_grad()
def rollout_episode(agent, config, episode, context, obs_mlp):
    """Watch `context` frames, then imagine the rest with recorded actions.

    Returns a dict of numpy arrays (length T) with ground-truth and predicted
    images / states / margin, plus the context length actually used.
    """
    wm = agent._wm
    dev = config.device

    T = episode["image"].shape[0]
    context = int(min(max(context, 1), T - 1))

    # build a (1, T, ...) batch and preprocess (handles /255, dtype, device)
    batch = {
        "image": episode["image"][None],            # (1, T, H, W, 3) uint8
        "obs_state": episode["obs_state"][None],     # (1, T, 2)
        "action": episode["action"][None],           # (1, T, 1)
        "is_first": episode["is_first"][None],        # (1, T)
        "is_terminal": episode["is_terminal"][None],  # (1, T)
    }
    data = wm.preprocess(batch)
    embed = wm.encoder(data)                          # (1, T, E)

    # --- posteriors over the context window (model conditioned on real frames) -
    post, _ = wm.dynamics.observe(
        embed[:, :context], data["action"][:, :context], data["is_first"][:, :context]
    )
    feat_ctx = wm.dynamics.get_feat(post)            # (1, context, F)

    # --- imagine forward from the last context latent using recorded actions ---
    state = {k: v[:, context - 1] for k, v in post.items()}  # (1, ...) at t=context-1
    imag_feats = []
    actions = data["action"]                         # (1, T, 1); action[t]: t -> t+1
    for t in range(context - 1, T - 1):
        state = wm.dynamics.img_step(state, actions[:, t], sample=False)
        imag_feats.append(wm.dynamics.get_feat(state))  # latent for step t+1
    if imag_feats:
        feat_imag = torch.stack(imag_feats, dim=1)   # (1, T-context, F)
        feat_full = torch.cat([feat_ctx, feat_imag], dim=1)  # (1, T, F)
    else:
        feat_full = feat_ctx

    # --- decode everything ----------------------------------------------------
    dec = wm.heads["decoder"](feat_full)
    img_pred = dec["image"].mode()[0]                # (T, H, W, 3) in ~[0,1]
    img_pred = torch.clamp(img_pred, 0.0, 1.0)
    obs_state_pred = dec["obs_state"].mode()[0]      # (T, 2) = [cos, sin]
    theta_dec = torch.atan2(obs_state_pred[:, 1], obs_state_pred[:, 0])

    state_pred = obs_mlp(feat_full)[0]               # (T, 3) = (x, y, theta)
    margin = wm.heads["margin"](feat_full)[0, :, 0]  # (T,) g(z): >0 safe, <0 unsafe

    img_truth = data["image"][0]                     # (T, H, W, 3) in [0,1]

    return {
        "context": context,
        "img_truth": to_np(img_truth),
        "img_pred": to_np(img_pred),
        "state_gt": episode["privileged_state"],     # (T, 3)
        "state_pred": to_np(state_pred),             # (T, 3)
        "theta_dec": to_np(theta_dec),               # (T,)
        "margin": to_np(margin),                     # (T,)
        "failure": episode["failure"],               # (T,)
    }


# ----------------------------------------------------------------------------
# Visualizations
# ----------------------------------------------------------------------------
def _draw_env(ax):
    lo, hi = WORKSPACE
    ax.add_patch(Rectangle((lo, SIDEWALK_Y), hi - lo, hi - SIDEWALK_Y,
                           color="0.72", zorder=0))
    ax.add_patch(Rectangle((lo, lo), hi - lo, hi - SIDEWALK_Y,
                           color="0.72", zorder=0))
    ax.add_patch(Circle((0, 0), R_OBS, color=(0.81, 0.33, 0.33), alpha=0.85, zorder=0))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def save_filmstrip(roll, path, max_frames=10):
    """Rows: ground truth / decoded imagined / abs error, columns = timesteps."""
    T = roll["img_truth"].shape[0]
    ctx = roll["context"]
    idx = np.linspace(0, T - 1, min(max_frames, T)).round().astype(int)
    idx = np.unique(idx)
    n = len(idx)

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 4.8))
    if n == 1:
        axes = axes[:, None]
    err = np.abs(roll["img_truth"] - roll["img_pred"])
    rows = [("truth", roll["img_truth"]), ("imagined", roll["img_pred"]), ("|error|", err)]
    for r, (label, frames) in enumerate(rows):
        for c, t in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(np.clip(frames[t], 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                phase = "ctx" if t < ctx else "imag"
                ax.set_title(f"t={t}\n{phase}", fontsize=8,
                             color="tab:green" if t < ctx else "tab:red")
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(
        f"Open-loop rollout: first {ctx} frames observed, remaining "
        f"{T - ctx} imagined from the latent transition model",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _colored_path(ax, xy, c, cmap, vmin, vmax, lw=2.0, **kw):
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=lw, **kw)
    lc.set_array(0.5 * (c[:-1] + c[1:]))
    ax.add_collection(lc)
    return lc


def save_trajectory_grid(rolls, path):
    n = len(rolls)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    mmax = max(1e-3, max(np.abs(r["margin"]).max() for r in rolls))
    lc = None
    for k, roll in enumerate(rolls):
        ax = axes[k // cols][k % cols]
        _draw_env(ax)
        gt = roll["state_gt"]
        pr = roll["state_pred"]
        ctx = roll["context"]
        # ground truth path
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.6, alpha=0.55,
                label="ground truth", zorder=2)
        # predicted path, colored by learned margin g(z)
        lc = _colored_path(ax, pr[:, :2], roll["margin"], "coolwarm_r",
                           -mmax, mmax, zorder=3)
        # markers: start, context->imagine handoff, end
        ax.plot(gt[0, 0], gt[0, 1], "o", color="black", ms=6, zorder=4)
        ax.plot(pr[ctx - 1, 0], pr[ctx - 1, 1], "s", color="tab:green", ms=7,
                label="imagine start", zorder=4)
        ax.plot(pr[-1, 0], pr[-1, 1], "*", color="tab:purple", ms=12,
                label="predicted end", zorder=4)
        perr = np.linalg.norm(pr[:, :2] - gt[:, :2], axis=1)
        ax.set_title(f"ep {k}: {ctx} obs + {len(gt) - ctx} imagined\n"
                     f"final pos err {perr[-1]:.3f}", fontsize=9)
        if k == 0:
            ax.legend(loc="upper left", fontsize=6, framealpha=0.85)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    if lc is not None:
        cb = fig.colorbar(lc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cb.set_label("learned margin g(z)  (>0 safe, <0 unsafe)", fontsize=9)
    fig.suptitle("Imagined trajectories (latent->world via probe) vs ground truth",
                 fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_state_error(rolls, path):
    """Position error vs horizon (aggregate) + per-step state traces (example)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # (a) aggregate open-loop error vs steps-since-imagine-start
    ax = axes[0]
    max_h = max(len(r["state_gt"]) - r["context"] for r in rolls)
    acc = [[] for _ in range(max_h + 1)]
    for r in rolls:
        ctx = r["context"]
        err = np.linalg.norm(r["state_pred"][:, :2] - r["state_gt"][:, :2], axis=1)
        for h in range(ctx - 1, len(err)):
            acc[h - (ctx - 1)].append(err[h])
    hs = [h for h, a in enumerate(acc) if a]
    mean = np.array([np.mean(acc[h]) for h in hs])
    std = np.array([np.std(acc[h]) for h in hs])
    ax.plot(hs, mean, color="tab:red", lw=2, label="mean position error")
    ax.fill_between(hs, mean - std, mean + std, color="tab:red", alpha=0.2, label="±1 std")
    ax.set_xlabel("imagination horizon (steps after context)")
    ax.set_ylabel("position error  ‖xy_pred - xy_gt‖")
    ax.set_title(f"Open-loop prediction error vs horizon (n={len(rolls)} episodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) per-step x / y / theta traces for the longest example episode
    r = max(rolls, key=lambda r: len(r["state_gt"]))
    ctx = r["context"]
    t = np.arange(len(r["state_gt"]))
    ax = axes[1]
    for j, (name, col) in enumerate([("x", "tab:blue"), ("y", "tab:orange")]):
        ax.plot(t, r["state_gt"][:, j], color=col, lw=1.6, label=f"{name} gt")
        ax.plot(t, r["state_pred"][:, j], "--", color=col, lw=1.6, label=f"{name} pred")
    ax.axvline(ctx - 1, color="tab:green", ls=":", label="imagine start")
    ax.set_xlabel("timestep")
    ax.set_ylabel("position")
    ax.set_title("Example episode: state prediction over time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_gif(roll, path, scale=4, fps=10):
    """Animated truth | imagined side-by-side strip."""
    import imageio

    T = roll["img_truth"].shape[0]
    ctx = roll["context"]
    frames = []
    H = roll["img_truth"].shape[1]
    sep = np.ones((H, 2, 3), dtype=np.uint8) * 255
    for t in range(T):
        truth = (np.clip(roll["img_truth"][t], 0, 1) * 255).astype(np.uint8)
        pred = (np.clip(roll["img_pred"][t], 0, 1) * 255).astype(np.uint8)
        # tint the predicted panel border green during context, red while imagined
        border = (60, 160, 60) if t < ctx else (200, 60, 60)
        pred = pred.copy()
        pred[:2, :, :] = border; pred[-2:, :, :] = border
        pred[:, :2, :] = border; pred[:, -2:, :] = border
        frame = np.concatenate([truth, sep, pred], axis=1)
        frame = np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8))
        frames.append(frame)
    imageio.mimsave(path, frames, fps=fps, loop=0)


# ----------------------------------------------------------------------------
def main():
    pre = argparse.ArgumentParser(add_help=True)
    pre.add_argument("--dataset", default=str(_REPO / "dubins_dataset.npz"),
                     help="path to the .npz dataset")
    pre.add_argument("--ckpt", default=None,
                     help="path to rssm_ckpt.pt (default: logs/dreamer_dubins/<wm_name>/rssm_ckpt.pt)")
    pre.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu")
    pre.add_argument("--num-episodes", type=int, default=6,
                     help="number of evaluation episodes to roll out / plot")
    pre.add_argument("--context", type=int, default=5,
                     help="number of real frames observed before imagination")
    pre.add_argument("--probe-steps", type=int, default=2000,
                     help="training steps for the latent->state read-out probe")
    pre.add_argument("--outdir", default=str(_HERE / "rollouts"),
                     help="directory for the output figures")
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--gif", action="store_true", help="also write animated GIFs")
    pre.add_argument("--filmstrips", type=int, default=3,
                     help="how many episodes to render as image filmstrips")
    args, remaining = pre.parse_known_args()

    agent, config, img_size = build_agent(args, remaining)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Output dir:", outdir)

    episodes = dubins_dataset.episodes_from_npz(args.dataset)
    print(f"Loaded {len(episodes)} episodes (img {img_size}x{img_size}).")

    # latent->state probe (for plotting imagined latents in world coordinates)
    obs_mlp = train_obs_probe(agent, config, episodes, args.probe_steps, seed=args.seed)

    # pick evaluation episodes from the held-out tail (not used to fit the probe),
    # preferring longer ones so the open-loop horizon is meaningful.
    rng = np.random.default_rng(args.seed)
    tail = episodes[int(0.9 * len(episodes)):]
    tail = sorted(tail, key=lambda e: -e["image"].shape[0])
    chosen = tail[: args.num_episodes]
    print(f"Rolling out {len(chosen)} episodes (context={args.context}) ...")

    rolls = []
    for i, ep in enumerate(chosen):
        roll = rollout_episode(agent, config, ep, args.context, obs_mlp)
        rolls.append(roll)
        perr = np.linalg.norm(roll["state_pred"][:, :2] - roll["state_gt"][:, :2], axis=1)
        print(f"  ep {i}: T={roll['img_truth'].shape[0]:3d}  "
              f"final pos err {perr[-1]:.3f}  "
              f"img MSE {np.mean((roll['img_truth'] - roll['img_pred'])**2):.4f}")

    # ---- write figures -------------------------------------------------------
    save_trajectory_grid(rolls, outdir / "trajectory_grid.png")
    save_state_error(rolls, outdir / "state_error.png")
    for i in range(min(args.filmstrips, len(rolls))):
        save_filmstrip(rolls[i], outdir / f"filmstrip_ep{i}.png")
    if args.gif:
        for i in range(min(args.filmstrips, len(rolls))):
            save_gif(rolls[i], outdir / f"rollout_ep{i}.gif")

    print("\nDone. Wrote:")
    for f in sorted(outdir.glob("*")):
        print("  ", f)


if __name__ == "__main__":
    main()

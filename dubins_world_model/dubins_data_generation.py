"""
Dubins-car dataset collection for latent-space reachability / world-model training.

Produces an offline dataset of (s_t, o_t, a_t) tuples where
    s_t = (x, y, theta)   privileged low-dimensional state
    o_t = RGB image       high-dimensional observation
    a_t = omega           yaw-rate action

Environment ("Dubins Car with Sidewalks" toy example)
------------------------------------------------------
    state     s = (x, y, theta),  workspace [-1, 1]^2
    action    a = omega  (yaw rate)
    dynamics  xdot     = v cos(theta)
              ydot     = v sin(theta)
              thetadot = omega
    failure   center obstacle:  x^2 + y^2 < r_obs^2   (r_obs = 0.5)
              sidewalks:        |y| > 0.8
    safe regions (outside both failure sets, by y-band):
              s1:  0.5 < y < 0.8   (upper near-sidewalk)
              s2: -0.8 < y < -0.5  (lower near-sidewalk)
              s3:  0   < y < 0.5   (upper near-center)
              s4: -0.5 < y < 0     (lower near-center)

Data-collection heuristic (answering the "simple heuristic" prompt)
-------------------------------------------------------------------
A world model has to predict the transition function everywhere it may later be
queried, so the dataset needs broad, roughly uniform coverage of (x, y, theta).
The heuristic used here:

  1. Sample each initial state uniformly over the whole workspace -- positions
     AND headings -- including inside failure regions. The dynamics are valid
     there and the model must learn them (a safety filter is queried near and
     inside the failure set).
  2. Drive with piecewise-constant random yaw rates ("action repeat"): sample an
     omega, hold it for a few steps, then resample. Constant-omega segments
     trace circular arcs, which sweep position/heading space far more
     efficiently than i.i.d. per-step noise (which mostly jitters in place).
  3. End an episode when the car leaves the workspace or the horizon is reached,
     then resample a fresh initial state. Transitions that exit the workspace
     are dropped (they cannot be rendered), so every stored frame is in-bounds.

Integration uses the closed-form solution of the unicycle under constant control
over each step (exact, not forward-Euler), so recorded transitions are accurate.

Stored dataset (np.savez_compressed), flat replay-buffer layout:
    obs        (N, H, W, 3) uint8     RGB observation o_t
    state      (N, 3)       float32   privileged state s_t = (x, y, theta)
    action     (N, 1)       float32   action a_t applied at s_t (0 on last step)
    is_first   (N,)         bool      start of an episode
    is_last    (N,)         bool      end of an episode
    gt_failure (N,)         bool      ground-truth failure label (privileged;
                                      for calibration/eval only -- NOT used as a
                                      training signal for the preference margin)
    region     (N,)         int8      region id (see REGION_NAMES)

Usage:
    python dubins_data_collection.py --num-steps 30000 --img-size 64 \
        --out dubins_dataset.npz --viz
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------- 
# Environment constants
# -----------------------------------------------------------------------------
WORKSPACE = (-1.0, 1.0)        # both x and y live in [-1, 1]
R_OBS = 0.5                    # center obstacle radius
SIDEWALK_Y = 0.8              # |y| > 0.8 is sidewalk failure

# region ids
R_CENTER, R_SIDEWALK, R_S1, R_S2, R_S3, R_S4 = 0, 1, 2, 3, 4, 5
REGION_NAMES = {
    R_CENTER: "center_fail",
    R_SIDEWALK: "sidewalk_fail",
    R_S1: "s1_upper_near_sidewalk",
    R_S2: "s2_lower_near_sidewalk",
    R_S3: "s3_upper_near_center",
    R_S4: "s4_lower_near_center",
}

# rendering colors
C_BG = (236, 236, 238)
C_SIDEWALK = (172, 172, 178)
C_OBSTACLE = (206, 84, 84)
C_CAR = (46, 96, 196)
C_CAR_OUTLINE = (20, 40, 90)


# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------
def wrap_to_pi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def step_state(s, omega, v, dt):
    """Exact integration of the Dubins/unicycle model under constant omega."""
    x, y, th = s
    if abs(omega) < 1e-6:
        nx = x + v * np.cos(th) * dt
        ny = y + v * np.sin(th) * dt
        nth = th
    else:
        nth = th + omega * dt
        nx = x + (v / omega) * (np.sin(nth) - np.sin(th))
        ny = y - (v / omega) * (np.cos(nth) - np.cos(th))
    return np.array([nx, ny, wrap_to_pi(nth)], dtype=np.float64)


def in_workspace(s, margin=0.0):
    lo, hi = WORKSPACE
    return (lo + margin <= s[0] <= hi - margin) and (lo + margin <= s[1] <= hi - margin)


def region_of(s):
    x, y = s[0], s[1]
    if x * x + y * y < R_OBS * R_OBS:
        return R_CENTER
    if abs(y) > SIDEWALK_Y:
        return R_SIDEWALK
    if 0.5 < y <= SIDEWALK_Y:
        return R_S1
    if -SIDEWALK_Y <= y < -0.5:
        return R_S2
    if 0.0 < y <= 0.5:
        return R_S3
    return R_S4  # -0.5 <= y <= 0


def is_failure(s):
    return region_of(s) in (R_CENTER, R_SIDEWALK)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def _w2p(x, y, img_size):
    """World (x, y) in [-1, 1]^2 -> pixel (col, row), with image y pointing down."""
    lo, hi = WORKSPACE
    col = (x - lo) / (hi - lo) * (img_size - 1)
    row = (hi - y) / (hi - lo) * (img_size - 1)
    return col, row


def render(s, img_size=64, draw_obstacles=True, car_len=0.20, car_half_w=0.10):
    """Render a top-down RGB observation of the car (and static environment)."""
    img = Image.new("RGB", (img_size, img_size), C_BG)
    d = ImageDraw.Draw(img)

    if draw_obstacles:
        # sidewalk bands: y in [0.8, 1] (top) and [-1, -0.8] (bottom)
        _, top_r = _w2p(0.0, 1.0, img_size)
        _, top_b = _w2p(0.0, SIDEWALK_Y, img_size)
        d.rectangle([0, top_r, img_size - 1, top_b], fill=C_SIDEWALK)
        _, bot_r = _w2p(0.0, -SIDEWALK_Y, img_size)
        _, bot_b = _w2p(0.0, -1.0, img_size)
        d.rectangle([0, bot_r, img_size - 1, bot_b], fill=C_SIDEWALK)
        # center obstacle circle
        cx0, cy0 = _w2p(-R_OBS, R_OBS, img_size)
        cx1, cy1 = _w2p(R_OBS, -R_OBS, img_size)
        d.ellipse([cx0, cy0, cx1, cy1], fill=C_OBSTACLE)

    # car as an oriented triangle
    x, y, th = s
    fwd = np.array([np.cos(th), np.sin(th)])
    left = np.array([-np.sin(th), np.cos(th)])
    p = np.array([x, y])
    tip = p + car_len * fwd
    bl = p - 0.5 * car_len * fwd + car_half_w * left
    br = p - 0.5 * car_len * fwd - car_half_w * left
    poly = [tuple(_w2p(*pt, img_size)) for pt in (tip, bl, br)]
    d.polygon(poly, fill=C_CAR, outline=C_CAR_OUTLINE)

    return np.asarray(img, dtype=np.uint8)


# -----------------------------------------------------------------------------
# Dataset collection
# -----------------------------------------------------------------------------
def collect_dataset(
    num_steps=30000,
    img_size=64,
    v=0.5,
    omega_max=1.5,
    dt=0.05,
    horizon=120,
    action_repeat_min=3,
    action_repeat_max=12,
    min_episode_len=5,
    draw_obstacles=True,
    seed=0,
    verbose=True,
):
    """Collect ~num_steps in-bounds transitions via random arc rollouts.

    Returns a dict of stacked arrays (see module docstring for the layout).
    """
    rng = np.random.default_rng(seed)
    lo, hi = WORKSPACE

    obs_buf, state_buf, act_buf = [], [], []
    first_buf, last_buf, fail_buf, region_buf = [], [], [], []

    total = 0
    n_episodes = 0
    while total < num_steps:
        # 1. uniform initial state over the whole workspace
        s = np.array(
            [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(-np.pi, np.pi)],
            dtype=np.float64,
        )

        ep_states, ep_actions = [], []
        omega = 0.0
        hold = 0
        for _ in range(horizon):
            # 2. piecewise-constant random yaw rate (action repeat)
            if hold <= 0:
                omega = float(rng.uniform(-omega_max, omega_max))
                hold = int(rng.integers(action_repeat_min, action_repeat_max + 1))
            hold -= 1

            s_next = step_state(s, omega, v, dt)
            if not in_workspace(s_next):
                break  # drop the exiting transition; end the episode here
            ep_states.append(s.copy())
            ep_actions.append(omega)
            s = s_next
        ep_states.append(s.copy())  # final in-bounds state, no outgoing action

        if len(ep_states) - 1 < min_episode_len:
            continue  # too short, discard and resample

        T = len(ep_states)  # number of stored frames (= transitions + 1)
        for t, st in enumerate(ep_states):
            obs_buf.append(render(st, img_size, draw_obstacles))
            state_buf.append(st.astype(np.float32))
            a = ep_actions[t] if t < T - 1 else 0.0
            act_buf.append([np.float32(a)])
            first_buf.append(t == 0)
            last_buf.append(t == T - 1)
            fail_buf.append(is_failure(st))
            region_buf.append(region_of(st))

        total += T
        n_episodes += 1
        if verbose and n_episodes % 25 == 0:
            print(f"  episodes={n_episodes:5d}  frames={total:7d}", flush=True)

    data = {
        "obs": np.stack(obs_buf).astype(np.uint8),
        "state": np.stack(state_buf).astype(np.float32),
        "action": np.stack(act_buf).astype(np.float32),
        "is_first": np.array(first_buf, dtype=bool),
        "is_last": np.array(last_buf, dtype=bool),
        "gt_failure": np.array(fail_buf, dtype=bool),
        "region": np.array(region_buf, dtype=np.int8),
    }
    if verbose:
        print(f"Done: {n_episodes} episodes, {data['obs'].shape[0]} frames.")
    return data


def save_dataset(data, out_path, meta):
    np.savez_compressed(out_path, **data)
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------------------------------------------------------------
# Optional sanity-check visualization
# -----------------------------------------------------------------------------
def visualize(data, out_dir=".", n_traj=8, n_obs=16, seed=0):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    rng = np.random.default_rng(seed)

    # episode index bookkeeping
    starts = np.where(data["is_first"])[0]
    ends = np.where(data["is_last"])[0]

    # (a) trajectories over the environment
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((-1, 0.8), 2, 0.2, color="0.7"))
    ax.add_patch(Rectangle((-1, -1.0), 2, 0.2, color="0.7"))
    ax.add_patch(Circle((0, 0), R_OBS, color=(0.81, 0.33, 0.33)))
    for k in rng.choice(len(starts), size=min(n_traj, len(starts)), replace=False):
        seg = slice(starts[k], ends[k] + 1)
        xy = data["state"][seg]
        ax.plot(xy[:, 0], xy[:, 1], lw=1.0, alpha=0.9)
        ax.plot(xy[0, 0], xy[0, 1], "k.", ms=4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title("sample trajectories (random arc rollouts)")
    fig.tight_layout()
    traj_path = os.path.join(out_dir, "dubins_trajectories.png")
    fig.savefig(traj_path, dpi=120)
    plt.close(fig)

    # (b) grid of sample observations
    idx = rng.choice(data["obs"].shape[0], size=n_obs, replace=False)
    cols = 4
    rows = int(np.ceil(n_obs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(data["obs"][i])
        ax.set_title(REGION_NAMES[int(data["region"][i])].split("_")[0], fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    obs_path = os.path.join(out_dir, "dubins_observations.png")
    fig.savefig(obs_path, dpi=120)
    plt.close(fig)
    return traj_path, obs_path


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Collect Dubins-car world-model dataset.")
    p.add_argument("--num-steps", type=int, default=30000)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--v", type=float, default=0.5)
    p.add_argument("--omega-max", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--action-repeat-min", type=int, default=3)
    p.add_argument("--action-repeat-max", type=int, default=12)
    p.add_argument("--no-obstacles", action="store_true",
                   help="do not render static failure regions into observations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="dubins_dataset.npz")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()

    data = collect_dataset(
        num_steps=args.num_steps,
        img_size=args.img_size,
        v=args.v,
        omega_max=args.omega_max,
        dt=args.dt,
        horizon=args.horizon,
        action_repeat_min=args.action_repeat_min,
        action_repeat_max=args.action_repeat_max,
        draw_obstacles=not args.no_obstacles,
        seed=args.seed,
    )

    meta = {
        "num_frames": int(data["obs"].shape[0]),
        "num_episodes": int(data["is_first"].sum()),
        "img_size": args.img_size,
        "v": args.v,
        "omega_max": args.omega_max,
        "dt": args.dt,
        "horizon": args.horizon,
        "workspace": WORKSPACE,
        "r_obs": R_OBS,
        "sidewalk_y": SIDEWALK_Y,
        "draw_obstacles": not args.no_obstacles,
        "region_names": REGION_NAMES,
        "action_space": "omega in [-omega_max, omega_max]",
        "seed": args.seed,
    }
    save_dataset(data, args.out, meta)
    print(f"Saved dataset -> {args.out}")
    print(f"  obs    {data['obs'].shape} {data['obs'].dtype}")
    print(f"  state  {data['state'].shape} {data['state'].dtype}")
    print(f"  action {data['action'].shape} {data['action'].dtype}")
    print(f"  failure frames: {int(data['gt_failure'].sum())} "
          f"({100*data['gt_failure'].mean():.1f}%)")

    if args.viz:
        tp, op = visualize(data, out_dir=os.path.dirname(args.out) or ".")
        print(f"  viz -> {tp}, {op}")


if __name__ == "__main__":
    main()
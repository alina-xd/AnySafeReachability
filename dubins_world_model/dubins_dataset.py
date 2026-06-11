"""
Adapter that turns the offline Dubins-car dataset produced by
`dubins_data_generation.py` into the in-memory episode `cache` that the
DreamerV3 world-model pipeline in this repository consumes.

Why an adapter (and not the repo's own loader)
-----------------------------------------------
The repo trains its world model from a *pickle* of trajectory dicts
(`scripts/generate_data_traj_cont.py` -> `tools.fill_expert_dataset_dubins`).
The new generator instead writes a flat replay-buffer `.npz`:

    obs        (N, H, W, 3) uint8     RGB observation o_t
    state      (N, 3)       float32   privileged state (x, y, theta)
    action     (N, 1)       float32   yaw-rate action omega applied at s_t
    is_first   (N,)         bool      start of an episode
    is_last    (N,)         bool      end of an episode
    gt_failure (N,)         bool      ground-truth failure (center obstacle OR sidewalk)
    region     (N,)         int8      region id

This module reads that `.npz`, slices it back into episodes (using
`is_first`/`is_last`), and produces episode dicts whose keys exactly match what
`models.WorldModel` expects after `tools.sample_episodes` /
`tools.from_generator` batch them:

    image            (T, H, W, 3) uint8   -> encoder/decoder cnn_keys
    obs_state        (T, 2)       float32 -> [cos(theta), sin(theta)]  (encoder mlp_keys)
    state            (T, 1)       float32 -> observed heading theta (book-keeping)
    privileged_state (T, 3)       float32 -> (x, y, theta), for obs-recon eval
    failure          (T,)         float32 -> margin/failure-classifier label (gt_failure)
    action           (T, 1)       float32 -> omega
    reward           (T,)         float32 -> zeros (unused by the WM here)
    discount         (T,)         float32 -> ones
    is_first/is_last/is_terminal (T,) bool

We deliberately keep the same observation convention as the original repo: the
vector input only reveals the heading (cos/sin theta); the planar position must
be inferred from the image. The architecture/config therefore stay unchanged.

Crucially we use the dataset's `gt_failure` for the failure-classifier label, so
*both* failure modes (center obstacle AND the |y|>0.8 sidewalks) are taught to
the margin head -- the repo's own loader only labels the center obstacle.
"""

import collections

import numpy as np


# keys the world model / dataset pipeline expects per transition
def _episode_slices(is_first, is_last, n):
    """Yield (start, end_inclusive) index ranges for each episode.

    Robust to a missing terminal flag: an episode also ends right before the
    next `is_first`, or at the final frame.
    """
    starts = list(np.where(is_first)[0])
    if len(starts) == 0:
        starts = [0]
    for i, s in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else n
        # default end is the frame just before the next episode begins
        e = next_start - 1
        # prefer an explicit is_last inside [s, next_start) if present
        last_idx = np.where(is_last[s:next_start])[0]
        if len(last_idx) > 0:
            e = s + int(last_idx[0])
        yield s, e


def episodes_from_npz(npz_path):
    """Load the .npz and return a list of per-episode dicts of numpy arrays."""
    data = np.load(npz_path)
    obs = data["obs"]                 # (N, H, W, 3) uint8
    state = data["state"].astype(np.float32)        # (N, 3)
    action = data["action"].astype(np.float32)      # (N, 1)
    is_first = data["is_first"].astype(bool)
    is_last = data["is_last"].astype(bool)
    gt_failure = data["gt_failure"].astype(np.float32)
    n = obs.shape[0]

    episodes = []
    for s, e in _episode_slices(is_first, is_last, n):
        T = e - s + 1
        if T < 2:
            continue  # need at least one transition
        theta = state[s : e + 1, 2:3]                      # (T, 1)
        obs_state = np.concatenate(
            [np.cos(theta), np.sin(theta)], axis=-1
        ).astype(np.float32)                                # (T, 2)
        ep = {
            "image": obs[s : e + 1].astype(np.uint8),       # (T, H, W, 3)
            "obs_state": obs_state,                         # (T, 2)
            "state": theta.astype(np.float32),              # (T, 1) observed heading
            "privileged_state": state[s : e + 1].astype(np.float32),  # (T, 3)
            "failure": gt_failure[s : e + 1].astype(np.float32),      # (T,)
            "action": action[s : e + 1].astype(np.float32),           # (T, 1)
            "reward": np.zeros((T,), dtype=np.float32),
            "discount": np.ones((T,), dtype=np.float32),
            "is_first": is_first[s : e + 1].copy(),
            "is_last": is_last[s : e + 1].copy(),
            "is_terminal": is_last[s : e + 1].copy(),
        }
        # guarantee episode-local first/last flags are consistent
        ep["is_first"][:] = False
        ep["is_first"][0] = True
        ep["is_last"][:] = False
        ep["is_last"][-1] = True
        ep["is_terminal"][:] = False
        ep["is_terminal"][-1] = True
        episodes.append(ep)
    return episodes


def fill_cache_from_npz(npz_path, cache, val_frac=0.05, is_val_set=False, seed=0):
    """Populate `cache` (an OrderedDict) with episodes from the .npz.

    Mirrors the role of `tools.fill_expert_dataset_dubins` but reads the flat
    replay-buffer .npz. The first `(1 - val_frac)` fraction of episodes are the
    training split, the remainder the validation split.

    Returns the number of episodes added.
    """
    episodes = episodes_from_npz(npz_path)
    n_eps = len(episodes)
    n_train = int(round(n_eps * (1.0 - val_frac)))
    sel = episodes[n_train:] if is_val_set else episodes[:n_train]
    for i, ep in enumerate(sel):
        key = f"val_traj_{i}" if is_val_set else f"exp_traj_{i}"
        cache[key] = ep
    return len(sel)


def dataset_image_size(npz_path):
    """Return the square image side length stored in the dataset."""
    data = np.load(npz_path)
    return int(data["obs"].shape[1])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect the converted Dubins dataset.")
    p.add_argument("--dataset", required=True, help="path to the .npz dataset")
    args = p.parse_args()

    eps = episodes_from_npz(args.dataset)
    n_frames = sum(len(e["image"]) for e in eps)
    n_fail = sum(int(e["failure"].sum()) for e in eps)
    print(f"episodes      : {len(eps)}")
    print(f"frames        : {n_frames}")
    print(f"image size    : {eps[0]['image'].shape[1]}x{eps[0]['image'].shape[2]}")
    print(f"failure frames: {n_fail} ({100.0 * n_fail / n_frames:.1f}%)")
    print(f"per-frame keys: {sorted(eps[0].keys())}")

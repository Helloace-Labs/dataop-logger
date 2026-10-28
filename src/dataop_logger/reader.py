"""DataopDataset: LeRobot frames joined with the sidecar, with standard targets and validity masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import adapter, events as ev, review as rv, store
from .validate import TICK_COLUMNS, validate

ACTION_COLS = ("base_action", "expert_action", "sent_action")
STRING_COLS = ("control_mode", "expert_source")


class DataopDataset:
    """Wraps a LeRobotDataset, including episode-filtered views. Keeps LeRobot's own sampling and indices.

    missing: "error" (default) refuses episodes in the view that have no committed run,
             "exclude" drops those episodes from iteration (original indices stay in each item),
             "mask" keeps them with every sidecar field marked invalid.
    """

    def __init__(self, lerobot_dataset, missing: str = "error", check: bool = True):
        if missing not in ("error", "exclude", "mask"):
            raise ValueError("missing must be 'error', 'exclude' or 'mask'")
        self.ds = lerobot_dataset
        self.missing = missing
        root = Path(lerobot_dataset.root)
        self.validation = validate(root) if check else None
        self.runs: dict[int, dict] = {}
        for run_path, m in store.iter_manifests(root, "committed"):
            if m["episode_index"] in self.runs:
                raise ValueError(f"episode {m['episode_index']} is bound to several runs; run dataop-logger validate")
            t = pq.read_table(run_path / "ticks.parquet")
            dim = len(m["action_schema"]["names"])
            cols = {}
            for c in TICK_COLUMNS:
                if c in ACTION_COLS:
                    cols[c] = np.asarray(t.column(c).to_pylist(), dtype=np.float32).reshape(t.num_rows, dim)
                elif c in STRING_COLS:
                    cols[c] = t.column(c).to_pylist()
                else:
                    cols[c] = t.column(c).to_numpy(zero_copy_only=False)
            review = rv.load(run_path)
            # moved drive-mode switch points count, so training sees the same run the reviewer saw
            mask = rv.train_mask(m, review, ev.effective_modes(m, review, cols["control_mode"], cols["expert_source"]))
            self.runs[m["episode_index"]] = {"run_id": m["run_id"], "dim": dim, "cols": cols, "train_mask": mask}
        feat = lerobot_dataset.meta.features.get("action", {})
        self.fallback_dim = int(np.prod(feat.get("shape", (0,)))) if feat else 0

        selected = adapter.selected_episodes(lerobot_dataset)
        eps = lerobot_dataset.meta.episodes
        spans = {int(e): (int(a), int(b)) for e, a, b in
                 zip(eps["episode_index"], eps["dataset_from_index"], eps["dataset_to_index"])}
        # Follow the wrapped view's own row order: relative position -> absolute row -> episode.
        rel = adapter.relative_index_map(lerobot_dataset)
        ep_of = {}
        for e, (a, b) in spans.items():
            for i in range(a, b):
                ep_of[i] = e
        if rel is not None:
            rows = sorted((r, a) for a, r in rel.items())
        else:
            rows = [(i, i) for i in range(len(lerobot_dataset))]
        self.view_episodes = sorted({ep_of[a] for _, a in rows if a in ep_of})
        self.unbound_episodes = [e for e in self.view_episodes if e not in self.runs]
        if self.unbound_episodes and missing == "error":
            raise ValueError(f"{len(self.unbound_episodes)} episodes have no committed dataop run "
                             f"(e.g. {self.unbound_episodes[:5]}); use missing='exclude' or 'mask'")
        self.index = np.asarray([r for r, a in rows if missing != "exclude" or ep_of.get(a) in self.runs],
                                dtype=np.int64)

    @property
    def bound_episodes(self) -> list[int]:
        """Bound episodes inside this (possibly filtered) view."""
        return sorted(e for e in self.view_episodes if e in self.runs)

    @property
    def all_bound_episodes(self) -> list[int]:
        return sorted(self.runs)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        item = self.ds[int(self.index[i])]
        ep = int(item["episode_index"])
        fi = int(item["frame_index"])
        run = self.runs.get(ep)
        if run is None:
            dim = self.fallback_dim
            side = {c: np.zeros(dim, np.float32) for c in ACTION_COLS}
            side.update({c: False for c in ("base_valid", "expert_valid", "safety_modified", "reward_valid",
                                            "terminated", "truncated")})
            side.update(control_mode="none", expert_source=None, blend_weight=0.0, inference_id=-1, chunk_offset=-1,
                        obs_ns=-1, obs_skew_ns=-1, sent_ns=-1, reward=0.0, run_bound=False, run_id=None,
                        train_mask=False)
        else:
            c = run["cols"]
            side = {k: c[k][fi] for k in TICK_COLUMNS if k != "frame_index"}
            side.update(run_bound=True, run_id=run["run_id"], train_mask=bool(run["train_mask"][fi]))
        bv, ev, keep = bool(side["base_valid"]), bool(side["expert_valid"]), bool(side["train_mask"])
        targets = {   # review edits (excluded spans, rejected runs) switch every target off via train_mask
            "dagger_label": side["expert_action"], "dagger_mask": ev and keep,
            "expert_residual": side["expert_action"] - side["base_action"], "expert_residual_mask": bv and ev and keep,
            "executed_residual": side["sent_action"] - side["base_action"],
            "executed_residual_mask": bv and not bool(side["safety_modified"]) and keep,
        }
        try:
            import torch

            def conv(v):
                if isinstance(v, np.ndarray):
                    return torch.as_tensor(v)
                if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
                    return torch.as_tensor(v)
                return v
        except ImportError:
            conv = lambda v: v
        item.update({f"dataop.{k}": conv(v) for k, v in {**side, **targets}.items()})
        return item

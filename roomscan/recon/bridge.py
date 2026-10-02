"""Join COLMAP pieces of a video through the fast turns that split them (fix loop; method from E5).

Each piece is first made metric on its own (DA3Metric depth vs that piece's sparse points). For the gap between two
consecutive pieces, DA3 chains every frame from the end of piece A through the turn into the start of piece B
(windows of the profile's size, linked on shared frames). The chain is mapped into A's metric frame with a similarity
fitted on the frames it shares with A; piece B is then mapped onto the chain with rotation + translation only (B is
already metric) using the frames they share.
"""

from __future__ import annotations

import numpy as np

from roomscan.recon.windows import WindowPrediction, run_windowed


def sim3_from_cams(src: list[np.ndarray], dst: list[np.ndarray], with_scale: bool = True):
    """(s, R, t): dst = s R src + t, from camera-to-world poses known in both frames."""
    R = sum(d[:3, :3] @ s_[:3, :3].T for s_, d in zip(src, dst))
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    cs = np.stack([x[:3, 3] for x in src])
    cd = np.stack([x[:3, 3] for x in dst])
    s = 1.0
    if with_scale:
        s = float(np.sqrt(((cd - cd.mean(0)) ** 2).sum() / max(((cs - cs.mean(0)) ** 2).sum(), 1e-12)))
    t = cd.mean(0) - s * R @ cs.mean(0)
    return s, R, t


def apply(T: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = s * R @ T[:3, 3] + t
    return out


class Bridger:
    def __init__(self, cfg):
        import torch

        from roomscan import models
        from roomscan.recon.da3_loader import load_da3_class, load_pretrained

        dev = cfg["runtime"]["device"]
        self.dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.torch = torch
        self.model = load_pretrained(load_da3_class(), models.get(cfg["recon"]["model"]).hf_repo).to(self.dev).eval()
        self.win = max(3, cfg["recon"]["max_views"])
        self.res = cfg["recon"]["image_long_side"]

    def chain(self, images: list[np.ndarray]) -> np.ndarray:
        """Camera-to-world poses for a run of consecutive frames, in the chain's own frame and scale."""

        def predict(idx):
            with self.torch.inference_mode():
                p = self.model.inference([images[i] for i in idx], process_res=self.res)
            E = np.tile(np.eye(4), (len(idx), 1, 1))
            E[:, :3, :4] = p.extrinsics
            return WindowPrediction(np.linalg.inv(E), p.depth, p.conf, p.intrinsics)

        return run_windowed(len(images), self.win, max(1, self.win - 1), predict).c2w

    def close(self):
        del self.model
        if self.dev == "mps":
            self.torch.mps.empty_cache()


def join_pieces(pieces_metric: list[dict[int, np.ndarray]], load_frame, cfg, anchor: int = 3,
                max_gap: int = 80) -> tuple[dict[int, np.ndarray], list[dict]]:
    """pieces_metric: per piece, frame index -> 4x4 c2w in that piece's own METRIC frame.
    The LARGEST piece is the anchor (global frame). Each remaining piece is bridged at its frames closest in time to an
    already placed frame, before, after or interleaved (fix loop after_v1: COLMAP pieces interleave in time, and
    requiring "B starts after A ends" dropped the 95-frame piece). Largest pieces are attached first.
    Returns frame index -> c2w in the anchor's frame, and a log per piece."""
    order = sorted(range(len(pieces_metric)), key=lambda i: -len(pieces_metric[i]))
    world = dict(pieces_metric[order[0]])
    log, pending = [], [pieces_metric[i] for i in order[1:]]
    bridger = None
    progress = True
    while pending and progress:
        progress = False
        for B in list(pending):
            placed = np.array(sorted(world))
            b = np.array(sorted(B))
            # closest (placed, B) frame pair in time
            d = np.abs(placed[:, None] - b[None, :])
            ia, ib = np.unravel_index(np.argmin(d), d.shape)
            lo, hi = sorted((int(placed[ia]), int(b[ib])))
            start, end = lo - anchor, hi + anchor
            if end - start > max_gap:
                continue
            idx = list(range(max(start, 0), end + 1))
            shared_a = [i for i in idx if i in world]
            shared_b = [i for i in idx if i in B]
            if len(shared_a) < 2 or len(shared_b) < 2:
                continue
            if bridger is None:
                bridger = Bridger(cfg)
            chain = bridger.chain([load_frame(i) for i in idx])
            C = dict(zip(idx, chain))
            sA = sim3_from_cams([C[i] for i in shared_a], [world[i] for i in shared_a], with_scale=True)
            Cw = {i: apply(T, *sA) for i, T in C.items()}  # chain in the anchor's metric frame
            sB = sim3_from_cams([B[i] for i in shared_b], [Cw[i] for i in shared_b], with_scale=False)
            for i, T in B.items():
                if i not in world:
                    world[i] = apply(T, *sB)
            log.append({"piece_frames": len(B), "joined": True, "bridge_frames": len(idx),
                        "chain_scale": round(float(sA[0]), 4)})
            pending.remove(B)
            progress = True
    for B in pending:
        log.append({"piece_frames": len(B), "joined": False, "reason": "no placed frame within the bridge limit"})
    if bridger is not None:
        bridger.close()
    return world, log

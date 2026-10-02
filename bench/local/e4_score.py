"""Pose scoring shared by E4/E5 (no pycolmap/torch imports)."""

import numpy as np


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def score(Tp, Tg):
    """Same definitions as bench/kaggle/model_eval_v2: per-view orientation error after one global rotation."""
    n = len(Tg)
    M = sum(Tg[i, :3, :3] @ Tp[i, :3, :3].T for i in range(n))
    U, _, Vt = np.linalg.svd(M)
    A = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    per = np.array([ang(Tg[i, :3, :3].T @ A @ Tp[i, :3, :3]) for i in range(n)])
    pair = [ang((Tg[i, :3, :3].T @ Tg[j, :3, :3]).T @ (Tp[i, :3, :3].T @ Tp[j, :3, :3]))
            for i in range(n) for j in range(i + 1, n)]
    s, R, t = umeyama(Tp[:, :3, 3], Tg[:, :3, 3])
    al = (s * (R @ Tp[:, :3, 3].T)).T + t
    return {"views_ok": int((per < 10).sum()), "rot_err_med": float(np.median(pair)),
            "ate_m": float(np.sqrt(((al - Tg[:, :3, 3]) ** 2).sum(1).mean()))}

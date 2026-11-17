"""E31 段 1：論文 2 の比較実験（GPU）。設計は `docs/researchvision.md` 3.8 節，実装の細部は `docs/exploratory-analyses.md` E31。

手法：P1〜P3（1 ソースの仮説転移．中心合わせ 2 × 罰則 C′ 4 × 切片 3），P4（全ソースの下書きの仮説転移．同じ 24 設定），
H3（帰納の中心合わせ＋合併），O1（重みつき合併．α 5 通り），O6（最近傍セントロイド．3 通り），O7（メタ初期値）。
P1〜P4 と O1 では K 本の交差検証の UAR も残す。テストと K 本は地図と同じ。
段 2（E31「段 2 の実装の細部」）：`--parts o2` は特徴の水増し O2（C ＝ ソースのみの C．交差検証なし），`--parts trans` は推移の中心合わせ
（P2・P3 の 12 設定と H3．cen = 2）。段 3：`--parts final` は最終候補の計算だけ（P3・P4 の C′ 4 × C_b {2, 8}，P1 の既定，
H3，O1．交差検証なし）。`--tag` を渡すと出力の名前に付ける（`<ターゲット>_<tag>.csv`）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/compare_b1.py --kind audio_wavlm [--target …] [--reps …] [--ks …]
        [--parts all|o2|trans] [--tag …]

出力：`results/compare_<kind>/<ターゲット>[_<tag>].csv`（method, source, cen, cp, cb, alpha, variant, rep, K, uar, cv_uar），
`.npz`（主な設定のテストの点数），`provenance[_<tag>].json`（最後の起動），`provenance_log.jsonl`（起動ごと）。
終わったターゲットは飛ばす（全部済みなら記録も書かない）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402
from split import speaker_folds  # noqa: E402

KS = ("1", "5", "10", "25")
CPS = (0.001, 0.01, 0.1, 1.0)
CBS = (2.0, 8.0, None)
ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEF_CP, DEF_CB = 0.01, 8.0
FINAL_CBS = (2.0, 8.0)   # 段 3 の P3・P4 の選ぶ範囲
MAX_ROWS = 6000          # 反復ごとの行の塊を縦に並べるときの上限（超えたら反復を分けて解く．解は変わらない）
O7_STEPS_IN, O7_LR_IN, O7_ITERS, O7_LR_OUT, O7_BATCH, O7_QUERY = 10, 0.5, 300, 1e-2, 4, 256
DEV = logreg.DEVICE


def cbname(cb):
    return "free" if cb is None else f"{cb:g}"


def uar(y, pred):
    return shift.uar(np.asarray(y), np.asarray(pred))


class Ctx:
    """ターゲットごとの分割（テスト，K 本，交差検証の袋，帰納の中心）。"""

    def __init__(self, tgt: M.Group, reps, ks, center: str = "ind"):
        self.tgt, self.reps, self.ks = tgt, list(reps), list(ks)
        ct_all = tgt.X.mean(0)
        self.te, self.idx, self.persons, self.ct, self.cvf = {}, {}, {}, {}, {}
        for r in self.reps:
            te = M.test_mask(tgt, r)
            order = M.kshot_order(tgt, np.flatnonzero(~te), r)
            self.te[r], self.persons[r] = te, np.unique(tgt.person[te])
            self.ct[r] = M.rows(tgt.X, np.flatnonzero(~te)).mean(0) if center == "ind" else ct_all   # 帰納／推移
            for K in self.ks:
                idx = M.take_k(order, K)
                self.idx[(r, K)] = idx
                f = speaker_folds(pd.Series(tgt.person[idx]), k=3, seed=r).to_numpy()
                ok = all(len(np.unique(tgt.y[idx][f != j])) == 4 for j in np.unique(f)) and len(np.unique(f)) >= 2
                self.cvf[(r, K)] = f if ok else None


def jobs_for(ctx: Ctx, rs, used_pos, n_total, with_cv=True):
    """（r, K, 袋 j または None）と，仮面（長さ n_total．used_pos：ターゲットの行番号 → 行列の中の位置）。"""
    jobs, masks = [], []
    for r in rs:
        for K in ctx.ks:
            idx = ctx.idx[(r, K)]
            m = np.zeros(n_total, bool)
            m[[used_pos[int(i)] for i in idx]] = True
            jobs.append((r, K, None)); masks.append(m)
            f = ctx.cvf[(r, K)]
            if with_cv and f is not None:
                for j in np.unique(f):
                    mj = np.zeros(n_total, bool)
                    mj[[used_pos[int(i)] for i in idx[f != j]]] = True
                    jobs.append((r, K, int(j))); masks.append(mj)
    return jobs, masks


def collect(ctx, jobs, fits, cs, X_of, rowsel, emit, extra):
    """テストの UAR と，交差検証の予測をまとめた UAR を emit(r, K, c, uar, cv_uar, logits) に渡す。"""
    tgt = ctx.tgt
    cvpred = defaultdict(dict)
    for (r, K, j), f in zip(jobs, fits):
        if j is None:
            continue
        idx = ctx.idx[(r, K)]
        hold = idx[ctx.cvf[(r, K)] == j]
        Xh = X_of(r, hold)
        for c in cs:
            cvpred[(r, K, c)].update(dict(zip(hold.tolist(), f.predict(Xh, c))))
    for (r, K, j), f in zip(jobs, fits):
        if j is not None:
            continue
        Xte = X_of(r, np.flatnonzero(ctx.te[r]))
        yte = tgt.y[ctx.te[r]]
        for c in cs:
            L = f.logits(Xte, c).cpu().numpy()
            cv = np.nan
            if (r, K, c) in cvpred:
                idx = ctx.idx[(r, K)]
                pr = np.array([cvpred[(r, K, c)][int(i)] for i in idx])
                cv = uar(tgt.y[idx], pr)
            emit(r, K, c, uar(yte, L.argmax(1)), cv, L, **extra)


def used_rows(ctx, r):
    return np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))


def chunks(ctx, rs):
    """反復を，K 本の行の和が MAX_ROWS を超えない組に分ける（1 組に少なくとも 1 つ）。"""
    out, cur, n = [], [], 0
    for r in rs:
        m = len(used_rows(ctx, r))
        if cur and n + m > MAX_ROWS:
            out.append(cur); cur, n = [], 0
        cur.append(r); n += m
    return out + [cur] if cur else out


def ht_block(ctx, rs, src, C_s, cen: int, add, cps=CPS, cbs=CBS, with_cv=True):
    """1 つの下書き（src．cen なら中心合わせ版）からの仮説転移．罰則 4 × 切片 3 の 12 設定（cen は固定）。

    cen：0 なし，1 帰納，2 推移（引く平均は ctx.ct）。中心合わせ版は反復ごとの行の塊を縦に並べるので，大きいときは分けて解く。
    """
    if cen:
        for part in chunks(ctx, rs):
            _ht_block(ctx, part, src, C_s, cen, add, cps, cbs, with_cv)
    else:
        _ht_block(ctx, rs, src, C_s, cen, add, cps, cbs, with_cv)


def _ht_block(ctx, rs, src, C_s, cen: int, add, cps, cbs, with_cv):
    tgt = ctx.tgt
    blocks, ys, jobs, masks = [], [], [], []
    if not cen:
        used = np.unique(np.concatenate([ctx.idx[(r, K)] for r in rs for K in ctx.ks]))
        blocks.append(M.rows(tgt.X, used)); ys.append(tgt.y[used])
        jobs, masks = jobs_for(ctx, rs, {int(i): k for k, i in enumerate(used)}, len(used), with_cv)
    else:                                                     # 反復ごとに中心が違うので，反復ごとの行の塊を縦に並べる
        layout, off = [], 0
        for r in rs:
            used_r = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
            blocks.append(M.rows(tgt.X, used_r) - ctx.ct[r]); ys.append(tgt.y[used_r])
            layout.append((r, used_r, off)); off += len(used_r)
        for r, used_r, o in layout:
            jj, mm = jobs_for(ctx, [r], {int(i): o + k for k, i in enumerate(used_r)}, off, with_cv)
            jobs += jj; masks += mm
    X, yX = torch.cat(blocks), M.labels(np.concatenate(ys))
    X_of = (lambda r, ii: M.rows(tgt.X, ii) - ctx.ct[r]) if cen else (lambda r, ii: M.rows(tgt.X, ii))
    fits = logreg.fit_batch(X, yX, masks * len(cbs), cps, anchor=src, anchor_c=C_s, stats=(src.mu, src.sd),
                            Cb=[cb for cb in cbs for _ in masks])     # 切片の罰則を 1 回の計算に束ねる
    for k, cb in enumerate(cbs):
        collect(ctx, jobs, fits[k * len(masks):(k + 1) * len(masks)], cps, X_of, None, add, {"cen": int(cen), "cb": cbname(cb)})


def h3_block(ctx, rs, Xs, ys_np, src_c, C_s, add, cen: int):
    """中心合わせ（ctx.ct．cen = 1 帰納，2 推移）＋合併。ソースはソースの平均を引く。反復の組ごとに解く。"""
    tgt = ctx.tgt
    cs = Xs.mean(0)
    for part in chunks(ctx, rs):
        blocks, ylist, jobs, masks, off = [], [], [], [], 0
        for r in part:
            used_r = used_rows(ctx, r)
            blocks.append(M.rows(tgt.X, used_r) - ctx.ct[r]); ylist.append(tgt.y[used_r])
            p = {int(i): off + k for k, i in enumerate(used_r)}
            for K in ctx.ks:
                jobs.append((r, K, None)); masks.append(np.array([p[int(i)] for i in ctx.idx[(r, K)]]))
            off += len(used_r)
        Xh = torch.cat(blocks + [Xs - cs])
        yh = M.labels(np.concatenate(ylist + [ys_np]))
        full = []
        for m in masks:
            f = np.zeros(len(yh), bool); f[off:] = True; f[m] = True; full.append(f)
        fits = logreg.fit_batch(Xh, yh, full, (C_s,), inits=[src_c] * len(full))   # 温め直し（解は同じ．速さのため）
        collect(ctx, jobs, fits, (C_s,), lambda r, ii: M.rows(tgt.X, ii) - ctx.ct[r], None, add, {"cen": cen})


def o2_block(ctx, rs, Xs, ys_np, src, C_s, add):
    """特徴の水増し（Daumé 2007）。ソースの統計で標準化した z から，ソースは [z, z, 0]，ターゲットは [z, 0, z]
    （`logreg.fit_batch` の aug．行列積は d 列で済ませる）。C ＝ ソースのみの C に固定し，交差検証はしない
    （C の候補 4 通りと交差検証は約 40 時間かかるため．2026-10-13 研究者の承認）。
    K の組（25 以下，50，全件）ごとに，使うターゲットの行だけを並べて解く。"""
    tgt = ctx.tgt
    mu, sd = src.mu, src.sd
    zs = (Xs - mu) / sd
    ns = len(ys_np)

    def aug_t(X):
        z = (X - mu) / sd
        return torch.cat([z, torch.zeros_like(z), z], 1)

    small = [K for K in ctx.ks if K not in ("50", "all")]
    for ks in [small] + [[K] for K in ctx.ks if K in ("50", "all")]:
        if not ks:
            continue
        sub = Ctx.__new__(Ctx)
        sub.__dict__.update(ctx.__dict__); sub.ks = ks
        used = np.unique(np.concatenate([ctx.idx[(r, K)] for r in rs for K in ks]))
        nu = len(used)
        Xa = torch.cat([(M.rows(tgt.X, used) - mu) / sd, zs])
        ya = M.labels(np.r_[tgt.y[used], ys_np])
        jobs, masks = jobs_for(sub, rs, {int(i): k for k, i in enumerate(used)}, nu, with_cv=False)
        masks = [np.r_[m, np.ones(ns, bool)] for m in masks]
        fits = logreg.fit_batch(Xa, ya, masks, (C_s,), aug=np.r_[np.zeros(nu, bool), np.ones(ns, bool)])
        collect(sub, jobs, fits, (C_s,), lambda r, ii: aug_t(M.rows(tgt.X, ii)), None, add, {"variant": "Cs"})
        del Xa, fits
        torch.cuda.empty_cache()


def run_target(kind, tgt, data, ref_all, reps, ks, out, tag="", parts="all", only=None):
    ctx = Ctx(tgt, reps, ks, center="trans" if parts == "trans" else "ind")
    rows, saved = [], {}
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    ref = map_rows(kind, tgt.name)
    nt = len(tgt.y)

    def adder(method, source):
        def add(r, K, c, u, cv, L, cen=0, cb="", alpha=np.nan, variant=""):
            rows.append({"target": tgt.name, "source": source, "method": method, "cen": cen, "cp": c, "cb": cb,
                         "alpha": alpha, "variant": variant, "rep": r, "K": K, "uar": u, "cv_uar": cv})
            key = None
            if method == "HT" and c == DEF_CP:
                key = {(1, f"{DEF_CB:g}"): "p3", (0, "free"): "p1", (1, "free"): "p2",
                       (2, f"{DEF_CB:g}"): "p3t", (2, "free"): "p2t"}.get((cen, cb))
            if method == "HT_ALL" and c == DEF_CP and cen == 1 and cb == f"{DEF_CB:g}":
                key = "p4"
            if method == "H3":
                key = "h3" if cen == 1 else "h3t"
            if method == "O2" and variant == "Cs":
                key = "o2"
            if key:
                saved[f"{key}_{source}_K{K}_r{r}"] = L.astype(np.float32)
        return add

    srcs = [s for s in data if s != tgt.name]
    for sname in [s for s in srcs if only is None or s in only]:   # only：1 ソースの部分を絞る（試走用）
        s = data[sname]
        t0 = time.perf_counter()
        tim = defaultdict(float)
        groups = {}
        for r in reps:
            keep = ~np.isin(s.person, ctx.persons[r])
            C_s = ref[("src_only", sname, "", r)][0]
            groups.setdefault((keep.tobytes(), C_s), []).append(r)
        for (kb, C_s), rs in groups.items():
            keep = np.frombuffer(kb, dtype=bool)
            Xs, ys_np = M.rows(s.X, keep), s.y[keep]
            ys = M.labels(ys_np)
            cs = Xs.mean(0)
            ns = len(ys_np)
            tb = time.perf_counter()
            if parts == "o2":
                src = logreg.fit(Xs, ys, (C_s,))
                o2_block(ctx, rs, Xs, ys_np, src, C_s, adder("O2", sname))
                tim["O2"] += time.perf_counter() - tb
                continue
            src_c = logreg.fit(Xs - cs, ys, (C_s,))
            if parts == "trans":
                ht_block(ctx, rs, src_c, C_s, 2, adder("HT", sname))
                tim["HT"] += time.perf_counter() - tb; tb = time.perf_counter()
                h3_block(ctx, rs, Xs, ys_np, src_c, C_s, adder("H3", sname), 2)
                tim["H3"] += time.perf_counter() - tb
                continue
            src = logreg.fit(Xs, ys, (C_s,))
            if parts == "final":                              # 段 3：P3 の選ぶ範囲（C_b ∈ {2, 8}）と P1 の既定だけ．交差検証なし
                ht_block(ctx, rs, src_c, C_s, 1, adder("HT", sname), cbs=FINAL_CBS, with_cv=False)
                ht_block(ctx, rs, src, C_s, 0, adder("HT", sname), cps=(DEF_CP,), cbs=(None,), with_cv=False)
            else:
                ht_block(ctx, rs, src, C_s, 0, adder("HT", sname))
                ht_block(ctx, rs, src_c, C_s, 1, adder("HT", sname))
            tim["HT"] += time.perf_counter() - tb; tb = time.perf_counter()
            # ---- H3（帰納）：反復ごとに中心を引いたターゲットの K 本 ＋ 中心を引いたソース
            h3_block(ctx, rs, Xs, ys_np, src_c, C_s, adder("H3", sname), 1)
            tim["H3"] += time.perf_counter() - tb; tb = time.perf_counter()
            # ---- O1：重みつき合併（中心合わせなし）
            used = np.unique(np.concatenate([ctx.idx[(r, K)] for r in rs for K in ctx.ks]))
            pos = {int(i): k for k, i in enumerate(used)}
            Xo = torch.cat([M.rows(tgt.X, used), Xs])
            yo = M.labels(np.r_[tgt.y[used], ys_np])
            nu = len(used)
            base_jobs, base_masks = jobs_for(ctx, rs, pos, nu, parts != "final")
            jobs, masks, weights = [], [], []
            ys_t = ys
            ws_src = torch.zeros(len(yo), dtype=logreg.DTYPE, device=DEV)
            cnt_s = torch.bincount(ys_t, minlength=4).to(logreg.DTYPE)
            ws_src[nu:] = 1.0 / (4 * cnt_s[ys_t])                          # ソースの中で感情を均等にし，和を 1 にする
            for alpha in ALPHAS:
                for job, m in zip(base_jobs, base_masks):
                    mt = torch.as_tensor(m, device=DEV)
                    yt_m = yo[:nu][mt]
                    cnt_t = torch.bincount(yt_m, minlength=4).to(logreg.DTYPE)
                    wt = torch.zeros(len(yo), dtype=logreg.DTYPE, device=DEV)
                    wt[:nu][mt] = 1.0 / (4 * cnt_t[yt_m].clamp_min(1))
                    S = float(m.sum() + ns)
                    w = (alpha * wt / wt.sum() + (1 - alpha) * ws_src) * S
                    fm = np.r_[m, np.ones(ns, bool)]
                    jobs.append(job + (alpha,)); masks.append(fm); weights.append(w)
            fits = logreg.fit_batch(Xo, yo, masks, (C_s,), weights=weights, inits=[src] * len(masks))   # 温め直し
            for alpha in ALPHAS:
                sel = [k for k, jb in enumerate(jobs) if jb[3] == alpha]
                collect(ctx, [jobs[k][:3] for k in sel], [fits[k] for k in sel], (C_s,), lambda r, ii: M.rows(tgt.X, ii), None,
                        adder("O1", sname), {"alpha": alpha})
            tim["O1"] += time.perf_counter() - tb
        print(f"  {kind} {tgt.name} ← {sname}：{time.perf_counter() - t0:.1f} 秒（束 {len(groups)}．"
              + "，".join(f"{k} {v:.0f}" for k, v in tim.items()) + " 秒）", flush=True)

    if parts in ("o2", "trans"):
        save(out, tgt.name, tag, saved, rows)
        return

    # ---- P4：全ソースの下書き（同じコーパスの群を除く）
    t0 = time.perf_counter()
    pool = [s for s in srcs if GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    Xp = torch.cat([data[s].X for s in pool]); yp_np = np.concatenate([data[s].y for s in pool]); yp = M.labels(yp_np)
    msf = ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv"
    if msf.exists():
        C_p = float(pd.read_csv(msf, dtype={"K": str}).C.iloc[0])
    else:                                                     # E26 と同じ手順（混ぜたソースのソースのみの交差検証）
        pp = np.concatenate([data[s].person for s in pool])
        C_p = float(M.train(Xp, yp_np, pp, np.ones(len(yp_np), bool), M.SRC_CV_SEED).C)
    cp_mean = Xp.mean(0)
    src_p = logreg.fit(Xp, yp, (C_p,))
    src_pc = logreg.fit(Xp - cp_mean, yp, (C_p,))
    if parts == "final":
        ht_block(ctx, reps, src_pc, C_p, 1, adder("HT_ALL", "ALL"), cbs=FINAL_CBS, with_cv=False)
        print(f"  {kind} {tgt.name} P4：{time.perf_counter() - t0:.1f} 秒", flush=True)
        save(out, tgt.name, tag, saved, rows)
        return
    ht_block(ctx, reps, src_p, C_p, 0, adder("HT_ALL", "ALL"))
    ht_block(ctx, reps, src_pc, C_p, 1, adder("HT_ALL", "ALL"))
    print(f"  {kind} {tgt.name} P4：{time.perf_counter() - t0:.1f} 秒", flush=True)

    # ---- O6：最近傍セントロイド（K 本だけ．余弦）
    for r in reps:
        te = np.flatnonzero(ctx.te[r]); yte = tgt.y[te]
        for variant, m in (("none", torch.zeros_like(cp_mean)), ("src_mean", cp_mean), ("tgt_mean", ctx.ct[r])):
            Zte = M.rows(tgt.X, te) - m
            Zte = Zte / Zte.norm(dim=1, keepdim=True).clamp_min(1e-12)
            for K in ks:
                idx = ctx.idx[(r, K)]
                Zk = M.rows(tgt.X, idx) - m
                Zk = Zk / Zk.norm(dim=1, keepdim=True).clamp_min(1e-12)
                cen = torch.stack([Zk[torch.as_tensor(tgt.y[idx] == k, device=DEV)].mean(0) for k in range(4)])
                cen = cen / cen.norm(dim=1, keepdim=True)
                pred = (Zte @ cen.T).argmax(1).cpu().numpy()
                rows.append({"target": tgt.name, "source": "ALL", "method": "O6", "cen": int(variant == "tgt_mean"), "cp": np.nan,
                             "cb": "", "alpha": np.nan, "variant": variant, "rep": r, "K": K, "uar": uar(yte, pred), "cv_uar": np.nan})

    # ---- O7：メタ初期値（MAML．線形の頭．ソースを混ぜた統計で標準化．float32）
    t0 = time.perf_counter()
    mu_p, sd_p = Xp.mean(0).float(), Xp.std(0, correction=0).float()
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    corp = {s: ((data[s].X.float() - mu_p) / sd_p, torch.as_tensor(data[s].y, device=DEV)) for s in pool}
    Zt_all = (tgt.X.float() - mu_p) / sd_p
    d = Zt_all.shape[1]

    def inner(W, b, Z, y):
        for _ in range(O7_STEPS_IN):
            loss = torch.nn.functional.cross_entropy(Z @ W.T + b, y)
            gW, gb = torch.autograd.grad(loss, (W, b), create_graph=True)
            W, b = W - O7_LR_IN * gW, b - O7_LR_IN * gb
        return W, b

    for K in [k_ for k_ in ks if k_ != "all"]:                    # 全件は疑似課題の支えの本数が決まらないので外す
        kk = int(K)
        g = torch.Generator(device="cpu").manual_seed(zlib.crc32(f"O7|{kind}|{tgt.name}|{K}".encode()))
        W0 = torch.zeros(4, d, device=DEV, requires_grad=True)
        b0 = torch.zeros(4, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([W0, b0], lr=O7_LR_OUT)
        names = list(corp)
        for it in range(O7_ITERS):
            opt.zero_grad()
            tot = 0.0
            for _ in range(O7_BATCH):
                sname = names[int(torch.randint(len(names), (1,), generator=g))]
                Z, y = corp[sname]
                sup, qry = [], []
                for k in range(4):
                    ii = torch.nonzero(y == k).squeeze(1)
                    perm = ii[torch.randperm(len(ii), generator=g).to(DEV)]
                    sup.append(perm[:kk]); qry.append(perm[kk:kk + O7_QUERY // 4])
                sup, qry = torch.cat(sup), torch.cat(qry)
                W, b = inner(W0, b0, Z[sup], y[sup])
                tot = tot + torch.nn.functional.cross_entropy(Z[qry] @ W.T + b, y[qry])
            (tot / O7_BATCH).backward()
            opt.step()
        for r in reps:
            idx = ctx.idx[(r, K)]
            yk = torch.as_tensor(tgt.y[idx], device=DEV)
            W, b = inner(W0, b0, Zt_all[torch.as_tensor(idx, device=DEV)], yk)
            te = np.flatnonzero(ctx.te[r])
            pred = (Zt_all[torch.as_tensor(te, device=DEV)] @ W.T + b).argmax(1).detach().cpu().numpy()
            rows.append({"target": tgt.name, "source": "ALL", "method": "O7", "cen": 0, "cp": np.nan, "cb": "", "alpha": np.nan,
                         "variant": "maml", "rep": r, "K": K, "uar": uar(tgt.y[te], pred), "cv_uar": np.nan})
    print(f"  {kind} {tgt.name} O6・O7：{time.perf_counter() - t0:.1f} 秒", flush=True)
    save(out, tgt.name, tag, saved, rows)


def stem(name, tag):
    return f"{name}_{tag}" if tag else name


def save(out, name, tag, saved, rows):
    np.savez_compressed(out / f"{stem(name, tag)}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{stem(name, tag)}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--parts", choices=["all", "o2", "trans", "final"], default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", nargs="+", help="1 ソースの部分のソースを絞る（試走用．全ソースの部分は変えない）")
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"compare_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{stem(t, args.tag)}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み（{stem('', args.tag) or '段 1'}）", flush=True)
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {
        "script": "exp/compare_b1.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
        "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "CPS": CPS, "CBS": [cbname(c) for c in CBS],
        "ALPHAS": ALPHAS, "O7": [O7_STEPS_IN, O7_LR_IN, O7_ITERS, O7_LR_OUT, O7_BATCH, O7_QUERY],
        "parts": args.parts, "tag": args.tag, "O2_C": "src_only", "MAX_ROWS": MAX_ROWS}
    (out / f"{stem('provenance', args.tag)}.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:                  # 起動ごとの記録（途中から続けたときも残す）
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in args.target or groups:
        if (out / f"{stem(t, args.tag)}.npz").exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, None, args.reps, args.ks, out, args.tag, args.parts, args.source)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()

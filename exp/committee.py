"""【探索的】E44（GPU）：ソースごとの分類器の票を，仮想ターゲットで測った癖でまとめると，下書きの予測 p より候補の感情を当てられるか（設定 1 の見込み q の段 1）。

規則は `docs/exploratory-analyses.md` E44（設計と予言 S1〜S3 は計算の前に書いた）。ターゲット T のソースは，T と同じコーパスの群を除く群
（下書きと同じ）。各ソース s の分類器 f_s は s の全件（s の平均を引く）で学ぶ（C は C を固定した地図の src_only の値）。ソースどうしの組
（s → t．t は T のソースで，s と同じコーパスでない群）で f_s を t の全件（t の平均を引く）に当て，s の癖の表を作る：
  H_s[y, ŷ]  ：本当の感情が y の発話に f_s が ŷ と言い切った割合（t ごとに数えて t の平均）
  Hs_s[y, ŷ] ：同じく f_s の確率 p_s(ŷ) の平均（確率のままの癖）
T の候補（反復ごとのテスト以外．候補の平均を引く）での q のまとめ方：
  mv  ：多数決（同点は確率の和で決める）
  tab ：癖の表を掛け合わせる．log q(y) = Σ_s log H_s[y, ŷ_s]（感情の割合は一様とする）
  A   ：当たる割合で重みを付けた多数決．重みは H_s から求めた「s が k と言ったときに本当に k である割合」（感情の割合は一様）
  B   ：確率のまま癖で直す．log q(y) = Σ_s log Σ_ŷ p_s(ŷ) Hs_s[y, ŷ]
  C   ：まとめ方を仮想ターゲットで学ぶ．logit(y) = b_y + (使えるソースの平均) Σ_ŷ W_s[y, ŷ] log p_s(ŷ)．T のソースの発話で学ぶ
        （その発話の群と同じコーパスのソースの票は使わない）．L2 の強さは，仮想ターゲットを 1 つずつ外す交差検証で選ぶ
  D   ：癖をターゲットの上でラベル無しで直す（Dawid–Skene の繰り返し）．tab から始め，癖の表を「仮想ターゲットの表を κ·n/4 件ぶんの
        事前の数とした」数え直しで更新する．κ は仮想ターゲット（T のソースを 1 つずつ外し，残りで同じ手順）の UAR で選ぶ
比べる相手は下書きの確率 p（E41〜E43 と同じ）。p を言い切った q は p と同じ当たり方なので，段 1 では p だけを出す。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/committee.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/committee.py --analyze

出力：`results/committee_<kind>/<ターゲット>.csv`（rep, variant, uar, acc, 食い違い，自信・票の一致，自信の上位の当たり方），
`<ターゲット>_src.csv`（各 f_s の候補の上の UAR），`<ターゲット>_meta.csv`（仮想ターゲットでの UAR と選んだつまみ），
`<ターゲット>_habit.csv`（癖の持ち越し：仮想ターゲットごとの表と平均の表の差，T での表と平均の表の差），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e44_targets.csv`・`e44_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

VARIANTS = ["p", "mv", "tab", "A", "B", "C", "D"]
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)            # C の L2 の強さ（W を単位行列へ引く）
KAPPAS = (0.01, 0.1, 1.0, 10.0)                # D の事前の数（候補 1 感情あたりの件数 n/4 の倍数）
EPS = 1e-3                                      # 癖の表の 0 を避ける足し込み
EM_ITERS = 30
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"
corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731


# ---------------------------------------------------------------- 癖とまとめ方

def src_C(kind: str) -> dict:
    """C を固定した地図の src_only の C（ソースごとの最頻値）。"""
    d = map_dir(kind)
    a = pd.concat([pd.read_csv(f, comment="#", usecols=["source", "method", "C"]) for f in sorted(d.glob("*.csv"))])
    a = a[a.method == "src_only"]
    return a.groupby("source").C.agg(lambda s: s.value_counts().idxmax()).to_dict()


def habits(P: dict, Y: dict, members: list, groups: list) -> tuple[dict, dict]:
    """members の各 s の癖の表（4×4．行 = 本当の感情）。数えるのは groups のうち s と同じコーパスでない群。"""
    H, Hs = {}, {}
    for s in members:
        hh, hs = [], []
        for t in groups:
            if t == s or corp(t) == corp(s):
                continue
            pt, yt = P[(s, t)], Y[t]
            pred = pt.argmax(1)
            h = np.zeros((4, 4))
            g = np.zeros((4, 4))
            for y in range(4):
                m = yt == y
                h[y] = np.bincount(pred[m], minlength=4) / max(m.sum(), 1)
                g[y] = pt[m].mean(0) if m.any() else 0.25
            hh.append(h)
            hs.append(g)
        H[s] = (np.mean(hh, 0) + EPS) / (1 + 4 * EPS)
        Hs[s] = (np.mean(hs, 0) + EPS) / (1 + 4 * EPS)
    return H, Hs


def softmax_np(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def agg_mv(Ps: list):
    V = sum(np.eye(4)[p.argmax(1)] for p in Ps)
    return (V + 1e-3 * sum(Ps) / len(Ps)) / len(Ps)


def agg_tab(Ps, Hl):
    return softmax_np(sum(np.log(H[:, p.argmax(1)].T) for p, H in zip(Ps, Hl)))


def agg_A(Ps, Hl):
    S = np.zeros_like(Ps[0])
    for p, H in zip(Ps, Hl):
        prec = np.diag(H) / H.sum(0)                                                   # s が k と言ったときに本当に k である割合（一様な割合で）
        k = p.argmax(1)
        S[np.arange(len(k)), k] += prec[k]
    S = S + 1e-3 * sum(Ps) / len(Ps)
    return S / S.sum(1, keepdims=True)


def agg_B(Ps, Hsl):
    return softmax_np(sum(np.log(p @ Hs.T) for p, Hs in zip(Ps, Hsl)))


def agg_D(Ps, Hl, kappa: float):
    """tab から始め，癖の表を候補の上で数え直す（事前の数 κ·n/4，感情の割合は一様のまま）。"""
    preds = [p.argmax(1) for p in Ps]
    q = agg_tab(Ps, Hl)
    n0 = kappa * len(q) / 4
    for _ in range(EM_ITERS):
        HT = []
        for k, H in zip(preds, Hl):
            cnt = q.T @ np.eye(4)[k]                                                   # (y, ŷ) の重み付きの数
            HT.append((n0 * H + cnt) / (n0 + q.sum(0)[:, None]))
        q = softmax_np(sum(np.log(Ht[:, k].T) for k, Ht in zip(preds, HT)))
    return q


class Pool:
    """C：使えるソースの票の log 確率を W_s で写して平均する多クラスのロジスティック回帰。"""

    def __init__(self, S: list):
        self.S = S

    def _logits(self, F, A, W, b):
        num = torch.einsum("nsk,syk->ny", F * A[:, :, None], W)
        return b + num / A.sum(1, keepdim=True).clamp(min=1.0)

    def fit(self, F: torch.Tensor, A: torch.Tensor, y: torch.Tensor, w: torch.Tensor, lam: float):
        S = len(self.S)
        W = torch.eye(4, dtype=F.dtype, device=F.device).repeat(S, 1, 1).requires_grad_(True)
        b = torch.zeros(4, dtype=F.dtype, device=F.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
        I = torch.eye(4, dtype=F.dtype, device=F.device)

        def closure():
            opt.zero_grad()
            ll = torch.nn.functional.cross_entropy(self._logits(F, A, W, b), y, reduction="none")
            loss = (w * ll).sum() + lam * ((W - I) ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict(self, F, A):
        return torch.softmax(self._logits(F, A, self.W, self.b), 1).cpu().numpy()


def pool_data(P, Y, S, groups):
    """C の学習データ：groups の発話．使えるソースは S のうちその群と同じコーパスでないもの。重みは群と感情ごとにそろえる。"""
    F, A, ys, ws = [], [], [], []
    for t in groups:
        n = len(Y[t])
        F.append(np.stack([np.log(np.clip(P[(s, t)], 1e-12, None)) if (s, t) in P else np.zeros((n, 4)) for s in S], 1))
        A.append(np.array([[float(s != t and corp(s) != corp(t))] * 1 for s in S]).T.repeat(n, 0))
        ys.append(Y[t])
        cnt = np.bincount(Y[t], minlength=4)
        ws.append(1.0 / (len(groups) * 4 * cnt[Y[t]]))
    tt = lambda a, dt=logreg.DTYPE: torch.as_tensor(np.concatenate(a), dtype=dt, device=logreg.DEVICE)  # noqa: E731
    return tt(F), tt(A), tt(ys, torch.long), tt(ws)


def uar(y, q):
    return CB.uar(y, q.argmax(1))


# ---------------------------------------------------------------- ターゲットごと

def run_target(kind, T, data, f, Cs, P, Y, reps, out):
    S = [s for s in data if corp(s) != corp(T)]
    groups_S = S
    H, Hs = habits(P, Y, S, groups_S)
    # 仮想ターゲット：T のソースを 1 つずつ外し，残りのソースで同じ手順（D の κ と C の λ を選ぶ．記述として各まとめ方の UAR も出す）
    mrows = []
    for t in S:
        St = [s for s in S if s != t and corp(s) != corp(t)]
        Gt = [g for g in S if g != t]
        Ht, Hst = habits(P, Y, St, Gt)
        Ps = [P[(s, t)] for s in St]
        yt = Y[t]
        mrows += [{"pseudo": t, "variant": "mv", "knob": "", "uar": uar(yt, agg_mv(Ps))},
                  {"pseudo": t, "variant": "tab", "knob": "", "uar": uar(yt, agg_tab(Ps, [Ht[s] for s in St]))},
                  {"pseudo": t, "variant": "A", "knob": "", "uar": uar(yt, agg_A(Ps, [Ht[s] for s in St]))},
                  {"pseudo": t, "variant": "B", "knob": "", "uar": uar(yt, agg_B(Ps, [Hst[s] for s in St]))}]
        for kp in KAPPAS:
            mrows.append({"pseudo": t, "variant": "D", "knob": kp, "uar": uar(yt, agg_D(Ps, [Ht[s] for s in St], kp))})
        Ftr, Atr, ytr, wtr = pool_data(P, Y, S, Gt)
        Fte, Ate, _, _ = pool_data(P, Y, S, [t])
        for lam in LAMS:
            mrows.append({"pseudo": t, "variant": "C", "knob": lam, "uar": uar(yt, Pool(S).fit(Ftr, Atr, ytr, wtr, lam).predict(Fte, Ate))})
    mt = pd.DataFrame(mrows)
    kappa = float(mt[mt.variant == "D"].groupby("knob").uar.mean().idxmax())
    lam = float(mt[mt.variant == "C"].groupby("knob").uar.mean().idxmax())
    mt["chosen"] = [(v == "D" and k == kappa) or (v == "C" and k == lam) or v in ("mv", "tab", "A", "B") for v, k in zip(mt.variant, mt.knob)]
    mt.to_csv(out / f"{T}_meta.csv", index=False)
    Fall, Aall, yall, wall = pool_data(P, Y, S, S)
    pool_C = Pool(S).fit(Fall, Aall, yall, wall, lam)

    # 下書き（E41〜E43 と同じ）
    pool_T = S
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in pool_T])
    yp = M.labels(np.concatenate([data[s].y for s in pool_T]))
    draft = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp

    ev = data[T]
    ctx = CB.Ctx(ev, reps, ["5"], center="ind")
    rows, srows, hrows = [], [], []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        p = torch.softmax(draft.logits(Xc, C_T), 1).cpu().numpy()
        Ps = {s: torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S}
        for s in S:
            srows.append({"target": T, "rep": r, "source": s, "uar": CB.uar(yq, Ps[s].argmax(1))})
            if r == reps[0]:                                                           # 癖の持ち越し（記述．T の正解を使う）
                h = np.stack([np.bincount(Ps[s].argmax(1)[yq == y], minlength=4) / max((yq == y).sum(), 1) for y in range(4)])
                devs = []
                for t in S:
                    if t == s or corp(t) == corp(s):
                        continue
                    pt = P[(s, t)].argmax(1)
                    ht = np.stack([np.bincount(pt[Y[t] == y], minlength=4) / max((Y[t] == y).sum(), 1) for y in range(4)])
                    devs.append(0.5 * np.abs(ht - H[s]).sum(1).mean())
                hrows.append({"target": T, "source": s, "tv_pseudo_mean": float(np.mean(devs)), "tv_pseudo_max": float(np.max(devs)),
                              "tv_target": float(0.5 * np.abs(h - H[s]).sum(1).mean())})
        Pl = [Ps[s] for s in S]
        Fq = torch.as_tensor(np.stack([np.log(np.clip(Ps[s], 1e-12, None)) for s in S], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
        Aq = torch.ones(len(yq), len(S), dtype=logreg.DTYPE, device=logreg.DEVICE)
        Q = {"p": p, "mv": agg_mv(Pl), "tab": agg_tab(Pl, [H[s] for s in S]), "A": agg_A(Pl, [H[s] for s in S]),
             "B": agg_B(Pl, [Hs[s] for s in S]), "C": pool_C.predict(Fq, Aq), "D": agg_D(Pl, [H[s] for s in S], kappa)}
        votes = np.stack([q.argmax(1) for q in Pl], 1)
        pa = p.argmax(1)
        for v, q in Q.items():
            a = q.argmax(1)
            conf = q.max(1)
            agree = (votes == a[:, None]).mean(1)
            row = {"target": T, "rep": r, "variant": v, "n": len(yq), "uar": CB.uar(yq, a), "acc": float((a == yq).mean()),
                   "n_flip": int((a != pa).sum()), "n_q_only": int(((a != pa) & (a == yq)).sum()),
                   "n_p_only": int(((a != pa) & (pa == yq)).sum()), "conf_mean": float(conf.mean()), "agree_mean": float(agree.mean()),
                   "kappa": kappa if v == "D" else np.nan, "lam": lam if v == "C" else np.nan}
            for frac in (0.5, 0.25):                                                   # 自信の上位の当たり方
                top = conf >= np.quantile(conf, 1 - frac)
                row[f"uar_top{int(frac * 100)}"] = CB.uar(yq[top], a[top]) if len(np.unique(yq[top])) == 4 else np.nan
                row[f"acc_top{int(frac * 100)}"] = float((a[top] == yq[top]).mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_src.csv", index=False)
    pd.DataFrame(hrows).to_csv(out / f"{T}_habit.csv", index=False)


def prepare(kind, data):
    """各群の分類器 f_s と，群どうしの確率 P[(s, t)]（t の全件．t の平均を引く）。"""
    Cs = src_C(kind)
    f, P, Y = {}, {}, {}
    Xc = {g: data[g].X - data[g].X.mean(0) for g in data}
    for s in data:
        f[s] = logreg.fit(Xc[s], M.labels(data[s].y), (float(Cs[s]),))
        Y[s] = np.asarray(data[s].y)
    for s in data:
        for t in data:
            if s != t:
                P[(s, t)] = torch.softmax(f[s].logits(Xc[t], float(Cs[s])), 1).cpu().numpy()
    return f, {s: float(c) for s, c in Cs.items()}, P, Y


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"committee_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith(("_src", "_meta", "_habit"))]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(x) for x in fs])
        a = a[a.rep.isin(EC.REPS)]
        g = a.groupby(["target", "variant"])[["uar", "acc", "conf_mean", "agree_mean", "uar_top50", "acc_top50", "uar_top25", "acc_top25"]].mean()
        fl = a.groupby(["target", "variant"])[["n_flip", "n_q_only", "n_p_only", "n"]].sum()
        src = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_src.csv"))])
        src = src[src.rep.isin(EC.REPS)].groupby(["target", "source"]).uar.mean()
        hb = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_habit.csv"))]).groupby("target")[["tv_pseudo_mean", "tv_target"]].mean()
        mt = pd.concat([pd.read_csv(x).assign(target=x.stem[:-5]) for x in sorted(d.glob("*_meta.csv"))])
        mt = mt[mt.chosen].groupby(["target", "variant"]).uar.mean()
        for T in sorted(a.target.unique()):
            row = {"kind": kind, "target": T}
            for v in VARIANTS:
                row[v] = float(g.loc[(T, v), "uar"]) * 100
                row[f"{v}:acc"] = float(g.loc[(T, v), "acc"]) * 100
                row[f"{v}:conf"] = float(g.loc[(T, v), "conf_mean"])
                row[f"{v}:agree"] = float(g.loc[(T, v), "agree_mean"])
                for c in ("uar_top50", "acc_top50", "uar_top25", "acc_top25"):
                    row[f"{v}:{c}"] = float(g.loc[(T, v), c]) * 100
                fr = fl.loc[(T, v)]
                row[f"{v}:flip"] = fr.n_flip / fr.n
                row[f"{v}:q_only|flip"] = fr.n_q_only / max(fr.n_flip, 1)
                row[f"{v}:p_only|flip"] = fr.n_p_only / max(fr.n_flip, 1)
                if v not in ("p",) and (T, v) in mt.index:
                    row[f"{v}:pseudo"] = float(mt.loc[(T, v)]) * 100
            row["src_best"] = float(src.loc[T].max()) * 100
            row["src_mean"] = float(src.loc[T].mean()) * 100
            row["tv_pseudo_mean"] = float(hb.loc[T, "tv_pseudo_mean"])
            row["tv_target"] = float(hb.loc[T, "tv_target"])
            tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for kind, g in D.groupby("kind", sort=False):
        row = {"kind": kind, "targets": len(g), "p": float(g.p.mean()), "src_mean": float(g.src_mean.mean()), "src_best": float(g.src_best.mean())}
        for v in VARIANTS[1:]:
            row[v] = float(g[v].mean())
            row[f"{v}-p"] = float((g[v] - g.p).mean())
            row[f"{v}>p"] = int((g[v] > g.p).sum())
            row[f"{v}:pseudo"] = float(g[f"{v}:pseudo"].mean()) if f"{v}:pseudo" in g else np.nan
        row["tab-mv"] = float((g.tab - g.mv).mean())
        row["C-tab"] = float((g.C - g.tab).mean())
        row["tv_pseudo_mean"] = float(g.tv_pseudo_mean.mean())
        row["tv_target"] = float(g.tv_target.mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        k = R.kind == kind
        if k.any():
            r = R[k].iloc[0]
            R.loc[k, "S1"] = bool(r["tab-mv"] > 0)
            R.loc[k, "S2"] = bool(max(r[f"{v}-p"] for v in VARIANTS[1:]) < 5)
            R.loc[k, "S3"] = bool(r["C-tab"] > 0)
    D.to_csv(OUT_REL / "e44_targets.csv", index=False)
    R.to_csv(OUT_REL / "e44_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[["kind", "targets", "p", "src_mean", "src_best"] + [f"{v}-p" for v in VARIANTS[1:]] + [f"{v}>p" for v in VARIANTS[1:]]
            + ["tab-mv", "C-tab", "tv_pseudo_mean", "tv_target"]].round(2).to_string(index=False))
    print(R[["kind"] + [f"{v}:pseudo" for v in VARIANTS[1:]]].round(2).to_string(index=False))
    vc = [c for c in ("S1", "S2", "S3") if c in R]
    if vc:
        print(R[["kind"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"committee_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    t0 = time.perf_counter()
    f, Cs, P, Y = prepare(args.kind, data)
    print(f"{args.kind}：ソースごとの分類器 {time.perf_counter() - t0:.0f} 秒（C {Cs}）", flush=True)
    prov = {"script": "exp/committee.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "reps": args.reps, "variants": VARIANTS, "src_C": Cs, "lams": LAMS,
            "kappas": KAPPAS, "eps": EPS, "em_iters": EM_ITERS, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

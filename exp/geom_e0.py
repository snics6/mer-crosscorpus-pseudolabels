"""【探索的】E58（GPU）：出発点の向きの誤り E_0 = W* − W_0 の幾何を測る（命題 1〜4 の診断．候補全体のラベルは解析にだけ使う）。

規則は `docs/exploratory-analyses.md` E58（設計と判断の決まりは計算の前に書いた）。生徒は E39 の学習器（座標 z は E39 の計量で直したもの）．
W_0 は E39 の出発点，W* は候補全体のラベルで出発点から学び直した分類器（C′ は候補を話者で 2 つに分けて選ぶ）．各ターゲット・各反復で次を測る：
  (1) E_0 のエネルギーが候補の特徴の主成分（分散の大きい順）のどこにあるか（上位 4・20・100 方向の割合）
  (2) 部分空間 S が E_0 を捕らえる割合 r(S) = ‖P_S E_0ᵀ‖² / ‖E_0‖²．S は 20 本の発話が張る空間（random_true，random_pool，LLL（条件 1 の教える選び方），
      margin，herding_pred，coreset（標準化しただけの座標），typiclust，白色化した座標での k-center と，白色化したノルムの上位）と，
      ソース間の向きの散らばり U_A（全体と感情ごと），仮のターゲットの直し向き U_B，その和．比べる相手は等方な k 次元の部分空間の期待値 k／d
  (3) その 20 本で生徒（E39 の設定）が実際に動く量：g(S) = ⟨W_S − W_0, E_0⟩／‖E_0‖²（E_0 の向きにどれだけ進むか），cos，動いた長さの比
  (4) 既存の実験（E49・E50・E53）の K=5 の UAR の得と r(S) の順位相関

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/geom_e0.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/geom_e0.py --analyze

出力：`results/geom_e0_<kind>/<ターゲット>.csv`（rep ごとの量），`provenance.json`，`provenance_log.jsonl`．判定は `results/explore_relation/e58_targets.csv`・`e58_tests.csv`・`e58_corr.csv`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import combine_drafts as CD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_check import knn  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_baselines import K_TYP, order_coreset, order_margin, pick_typiclust  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import fill, present_weights  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_student39 import Student  # noqa: E402

KINDS = ["audio_wavlm", "video", "text_roberta"]
N_SEL = 20                                              # K=5 の本数
CS_STAR = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)           # W* の罰則の候補（候補を話者で 2 つに分けて選ぶ）
CB_STAR = 8.0
C_PSEUDO = 0.01
EPS_WHITE = 0.01                                        # 白色化で小さい固有値に足す分（平均の割合）
SETS = ["random_true", "random_pool", "LLL", "margin", "herding_pred", "coreset_raw", "typiclust", "kcenter_white", "norm_white"]
OUT_REL = ROOT / "results" / "explore_relation"


E40_KINDS = ("audio_wavlm", "audio_hubert", "video", "fuse-audio-video")


KINDS_2ND = ("video_clip", "text_deberta", "fuse2-audio-video-text")   # 第 2 エンコーダ（E62）．audio_hubert は E40 にある


def settings(kind, T):
    """(γ, ρ_s, ρ_M, C_b, C′, 歩幅 c)．E40 のある種類（A・A(HuBERT)・V・AV）は E40 の random の組，言葉を含む種類は E39 の言葉の値と E37 の言葉の歩幅．
    第 2 エンコーダの種類（E62）は E39 の格子で仮のターゲットだけから選んだ値（`e62_targets.csv`．歩幅は無いので NaN）。"""
    if kind in KINDS_2ND:
        s = pd.read_csv(OUT_REL / "e62_targets.csv", dtype={"K": str, "cb": str}).query("kind == @kind").drop_duplicates("target").set_index("target").loc[T]
        cb = None if s.cb == "free" else float(s.cb)
        return float(s.gamma), float(s.rho_s), float(s.rho_M), cb, float(s.cp), float("nan")
    if kind not in E40_KINDS:
        s = pd.read_csv(OUT_REL / "e39_text_targets.csv", dtype={"K": str}).query("kind == @kind").drop_duplicates("target").set_index("target").loc[T]
        c = pd.read_csv(OUT_REL / "e37_targets_text.csv", dtype={"K": str}).query("kind == @kind").drop_duplicates("target").set_index("target").loc[T, "policy_star"]
        return float(s.gamma), float(s.rho_s), float(s.rho_M), float(s.cb), float(s.cp), float(c.split("_c")[1])
    s = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).query("kind == @kind").drop_duplicates("target").set_index("target").loc[T]
    return float(s.gamma), float(s.rho_s), float(s.r_rho_M), float(s.r_cb), float(s.r_cp), float(s.t_c)


def fit_star(Z, y, speaker, anchor, C_anchor, seed):
    """座標 Z の上で，出発点 anchor から全ラベルで学び直す．C′ は話者で 2 つに分けた片方で学び，他方の UAR で選ぶ．(W, b, C′, 選ぶときの UAR) を返す。"""
    yt = M.labels(y)
    zero = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)
    one = torch.ones_like(zero)
    fold = M.speaker_folds(pd.Series(np.asarray(speaker)), k=2, seed=seed).to_numpy()
    m0 = fold == 0
    w0 = torch.zeros(len(y), dtype=logreg.DTYPE, device=Z.device)
    w0[torch.as_tensor(m0, device=Z.device)] = logreg.balanced_weights(yt[torch.as_tensor(m0, device=Z.device)])
    half = logreg.fit_batch(Z, yt, [m0], CS_STAR, anchor=anchor, anchor_c=C_anchor, stats=(zero, one), Cb=[CB_STAR], weights=[w0])[0]
    Z1, y1 = Z[torch.as_tensor(~m0, device=Z.device)], np.asarray(y)[~m0]
    sc = {c: CB.uar(y1, (Z1 @ half.W[c].T + half.b[c]).argmax(1).cpu().numpy()) for c in CS_STAR}
    c_star = max(sc, key=sc.get)
    w = logreg.balanced_weights(yt)
    fit = logreg.fit_batch(Z, yt, [np.ones(len(y), bool)], (c_star,), anchor=anchor, anchor_c=C_anchor, stats=(zero, one),
                           Cb=[CB_STAR], weights=[w])[0]
    return fit.W[c_star], fit.b[c_star], c_star, sc[c_star]


def capture(E0, vecs):
    """E_0（4×d）の行を vecs（k×d）の張る空間へ射影したエネルギーの割合。"""
    if vecs is None or len(vecs) == 0:
        return float("nan")
    Q, _ = torch.linalg.qr(vecs.T)                      # d×k の正規直交基底
    return float(((E0 @ Q) ** 2).sum() / (E0 ** 2).sum())


def source_dirs_z(st, lin, S, rho_s, Xu, Mh):
    """ソース群ごとの向き（尺度をそろえたもの）を，ターゲットの z 座標で並べる（|S|×4×d）。"""
    out = []
    for s in S:
        A, a0 = lin[(s, rho_s)]
        sc = CD.pair_scale(Xu @ A.T + a0)
        out.append(((A / sc) * st.src.sd) @ Mh)
    return torch.stack(out)


def pseudo_corrections_x(data, S, gamma, rho_s, cache):
    """仮のターゲット c ごとの直し向き E^{(c)} = W*_c − W_0^{(c)} を，生の特徴の座標で並べる（|S|·4 × d）。"""
    rows = []
    for c in S:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        src_c, lin_c, n_c = M5.pool_info(data, pool_c, C_PSEUDO, cache)
        Xu = data[c].X - data[c].X.mean(0)
        st_c = Student(data[c], Xu, src_c, lin_c, n_c, pool_c, C_PSEUDO, gamma, rho_s, [1.0])      # ρ_M = 1 → z = 標準化した座標
        Wst = fit_star(st_c.Xstd, np.asarray(data[c].y), data[c].speaker, st_c.anchor(1.0), C_PSEUDO, 7)[0]
        rows.append((Wst - st_c.W_std) / src_c.sd)                                                  # 生の座標の向き（4×d）
    return torch.cat(rows)


def run_target(kind, T, data, reps, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, c_t = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    E_pseudo_x = pseudo_corrections_x(data, S, gamma, rho_s, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ["1", "5"], center="ind")
    rows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        pos_of = {int(i): j for j, i in enumerate(pool)}
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, gamma, rho_s, [rho_m])
        Am, Mh = st.mats[rho_m]
        Z = st.Xstd @ Am                                                                             # 生徒の座標
        W0 = st.W_std @ Mh
        Wst, bst, c_star, uar_half = fit_star(Z, yq, ev.speaker.iloc[pool], st.anchor(rho_m), C_T, 1000 + r)
        E0 = Wst - W0
        te = np.flatnonzero(ctx.te[r])
        Zte = ((M.rows(ev.X, te) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am
        uar_star = CB.uar(ev.y[te], (Zte @ Wst.T + bst).argmax(1).cpu().numpy())
        uar_w0 = CB.uar(ev.y[te], (Zte @ W0.T + st.b_std).argmax(1).cpu().numpy())
        d = Z.shape[1]
        Zc = Z - Z.mean(0)
        lam, V = torch.linalg.eigh(Zc.T @ Zc / len(Zc))
        lam, V = lam.flip(0).clamp(min=0), V.flip(1)
        prof = ((E0 @ V) ** 2).sum(0)
        prof = (prof / prof.sum()).cumsum(0).cpu().numpy()
        Zw = Zc @ (V * (lam + EPS_WHITE * lam.mean()).rsqrt()) @ V.T                                  # 白色化した座標
        Zf, Xf = Z.float(), st.Xstd.float()
        P = torch.softmax(st.L0, 1).float()
        pred = st.L0.argmax(1).cpu().numpy()
        sel = {"random_true": np.array([pos_of[int(i)] for i in ctx.idx[(r, "5")]]),
               "random_pool": np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool))[:N_SEL]}
        Km = Zf @ Zf.T
        Km.fill_diagonal_(0.0)
        sel["LLL"] = np.array(ablate_order(Km, yq, st.L0, c_t, N_SEL, "LLL")[0])
        del Km
        sel["margin"] = np.array(order_margin(P)[:N_SEL])
        groups = [np.flatnonzero(pred == g) for g in range(4)]
        hord = [list(G[herd(st.Xstd[torch.as_tensor(G, device=CB.DEV)], min(len(G), N_SEL))]) if len(G) else [] for G in groups]
        sel["herding_pred"] = np.array(fill(groups, [o[:5] for o in hord], 5, lambda j, ps, hord=hord: hord[j][len(ps)]))
        sel["coreset_raw"] = np.array(order_coreset(Xf, N_SEL))
        Xn = Xf / Xf.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(Xf, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)
        sel["typiclust"] = np.array(pick_typiclust(Xf, typ, N_SEL, torch.Generator().manual_seed(zlib.crc32(f"e46|typiclust5|{T}|{r}".encode()))))
        Zwf = Zw.float()
        sel["kcenter_white"] = np.array(order_coreset(Zwf, N_SEL))
        sel["norm_white"] = torch.argsort(Zwf.norm(dim=1), descending=True)[:N_SEL].cpu().numpy()
        row = {"target": T, "rep": r, "d": d, "n_pool": len(pool), "d_eff": float(lam.sum() ** 2 / (lam ** 2).sum()),
               "c_star": c_star, "uar_half": uar_half, "uar_star_test": uar_star, "uar_w0_test": uar_w0,
               "normE0": float(E0.norm()), "normW0": float(W0.norm()), "normBeta0": float((bst - st.b_std).norm()),
               "frac_top4": float(prof[3]), "frac_top20": float(prof[19]), "frac_top100": float(prof[min(99, d - 1)]),
               "null20": N_SEL / d}
        masks, weights = [], []
        for name, idx in sel.items():
            assert len(np.unique(idx)) == N_SEL, (name, len(np.unique(idx)))
            row[f"r_{name}"] = capture(E0, Z[torch.as_tensor(idx, device=Z.device)])
            mk = np.zeros(len(pool), bool)
            mk[idx] = True
            w = torch.zeros(len(pool), dtype=logreg.DTYPE, device=Z.device)
            mt = torch.as_tensor(mk, device=Z.device)
            w[mt] = present_weights(M.labels(yq)[mt])
            masks.append(mk)
            weights.append(w)
        zero = torch.zeros(d, dtype=Z.dtype, device=Z.device)
        fits = logreg.fit_batch(Z, M.labels(yq), masks, (cp,), anchor=st.anchor(rho_m), anchor_c=C_T, stats=(zero, torch.ones_like(zero)),
                                Cb=[cb] * len(masks), weights=weights)                                 # 生徒がその 20 本で実際に動く量
        for name, fit in zip(sel, fits):
            D_ = fit.W[cp] - W0
            row[f"g_{name}"] = float((D_ * E0).sum() / (E0 ** 2).sum())
            row[f"cos_{name}"] = float((D_ * E0).sum() / (D_.norm() * E0.norm()).clamp(min=1e-12))
            row[f"mov_{name}"] = float(D_.norm() / E0.norm())
        Ws = source_dirs_z(st, lin, S, rho_s, Xc, Mh)                                               # |S|×4×d
        Dif = (Ws - Ws.mean(0)).reshape(-1, d)
        row["r_UA"] = capture(E0, Dif)
        row["dim_UA"] = int(torch.linalg.matrix_rank(Dif))
        row["r_UA_cls"] = float(sum(((E0[k:k + 1] @ torch.linalg.qr((Ws[:, k, :] - Ws[:, k, :].mean(0)).T)[0]) ** 2).sum()
                                    for k in range(4)) / (E0 ** 2).sum())
        UB = (E_pseudo_x * st.src.sd) @ Mh
        row["r_UB"] = capture(E0, UB)
        row["dim_UB"] = int(torch.linalg.matrix_rank(UB))
        row["r_UAB"] = capture(E0, torch.cat([Dif, UB]))
        row["null_UA"] = row["dim_UA"] / d
        row["null_UB"] = row["dim_UB"] / d
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=KINDS) -> int:
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"geom_e0_{kind}"
        for x in sorted(d.glob("*.csv")):
            a = pd.read_csv(x)
            a = a[a.rep.isin(EC.REPS)]
            tabs.append(a.mean(numeric_only=True).to_frame().T.assign(kind=kind, target=x.stem))
    D = pd.concat(tabs)
    cols = ["d", "d_eff", "c_star", "uar_half", "uar_star_test", "uar_w0_test", "frac_top4", "frac_top20", "frac_top100", "null20"] + \
           [f"r_{s}" for s in SETS] + ["r_UA", "r_UA_cls", "dim_UA", "null_UA", "r_UB", "dim_UB", "null_UB", "r_UAB", "normE0", "normW0", "normBeta0"] + \
           [f"{q}_{s}" for q in ("g", "cos", "mov") for s in SETS]
    R = D.groupby("kind")[cols].mean()
    for s in SETS[1:]:
        R[f"r_{s}/r_random_true"] = R[f"r_{s}"] / R["r_random_true"]
    # 既存の UAR の得との順位相関（K=5．生徒は E39．A・V は E49・E50，T は E53）
    pairs = {"random_pool": "random_pool", "LLL": "LLL", "margin": "margin", "herding_pred": "herding_pred", "coreset_raw": "coreset", "typiclust": "typiclust"}
    e49 = pd.read_csv(OUT_REL / "e49_targets.csv", dtype={"K": str}).query("K == '5'").set_index(["kind", "target"])
    e50 = pd.read_csv(OUT_REL / "e50_targets.csv", dtype={"K": str}).query("K == '5'").set_index(["kind", "target"])
    e53 = pd.read_csv(OUT_REL / "e53_targets.csv", dtype={"K": str}).query("K == '5'").set_index("target")
    crows = []
    for _, row in D.iterrows():
        kind, T = row.kind, row.target
        for s, col in pairs.items():
            try:
                if kind == "text_roberta":
                    u, base = float(e53.loc[T, col]), float(e53.loc[T, "random_pool"])
                else:
                    src = e49 if col in ("random_pool", "LLL", "herding_pred") else e50
                    u, base = float(src.loc[(kind, T), col]), float(e49.loc[(kind, T), "random_pool"])
            except KeyError:
                continue
            crows.append({"kind": kind, "target": T, "set": s, "r": float(row[f"r_{s}"]), "gain": u - base})
    Cr = pd.DataFrame(crows)
    corr = Cr.groupby("kind").apply(lambda g: float(spearmanr(g.r, g.gain).statistic)).rename("spearman_r_gain")
    R = R.join(corr)
    for kind in R.index:
        g = R.loc[kind]
        R.loc[kind, "G1"] = bool(g["r_LLL"] >= 3 * g["r_random_true"])
        R.loc[kind, "G2"] = bool((1 - g["frac_top20"]) >= 0.7 and g["r_kcenter_white"] >= 2 * g["r_random_true"])
        R.loc[kind, "G3"] = bool(g["r_UA"] >= 0.5)
        R.loc[kind, "G4"] = bool(g["r_UB"] >= 0.5)
    D.to_csv(OUT_REL / "e58_targets.csv", index=False)
    R.to_csv(OUT_REL / "e58_tests.csv")
    Cr.to_csv(OUT_REL / "e58_corr.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["G1", "G2", "G3", "G4"]).round(3).T.to_string())
    print("\n== ターゲットごと（r の主なもの．反復の平均）")
    print(D.set_index(["kind", "target"])[["d_eff", "frac_top20", "r_random_true", "r_LLL", "r_kcenter_white", "r_UA", "r_UB", "g_random_true", "g_LLL"]].round(3).to_string())
    print(R[["G1", "G2", "G3", "G4"]].to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"geom_e0_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    prov = {"script": "exp/geom_e0.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "reps": args.reps, "n_sel": N_SEL, "cs_star": CS_STAR, "cb_star": CB_STAR,
            "c_pseudo": C_PSEUDO, "eps_white": EPS_WHITE, "sets": SETS, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

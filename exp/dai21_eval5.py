"""実験 21（DAI-21）：提案の評価の回の続き．(1) 提案の答えを疑似ラベルにして融合の分類器を学び直す（5 コーパス），(2) CREMA-D を評価のターゲットにする．
設計：docs/設計/実験21_設計.md．

段：
  selftest   人工の数での検算
  check3     まとめ方の計算（combine_all）を，実験 16 の下調べの保存（3 つの分類器）に当てて保存の判定と照らす（読むだけ）
  check18    CREMA-D の単位の計算を，実験 18 の単位（RAVDESS を除いた回）と同じ設定で流して保存の判定と照らす
  freeze     固定のファイル frozen/dai21_frozen.json を作る（実験 20 の計算の後．コミットしてから計算する）
  students4  4 コーパス：実験 20 の単位の出力から疑似ラベルを作り，融合の分類器を学び直す
  cremad     CREMA-D：土台（K 本だけの分類器，微調整した声・顔）→ まとめ方 → 学び直し
  summary    UAR と区間，差と区間

実験 20 の計算のあいだは exp に置かない（tools/ に置き，計算の後に exp へ移す）．
使い方：.venv/bin/python <この台本> --stage cremad
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai18_probe as P18  # noqa: E402
import dai20_eval as P20  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT18 = ROOT / "results" / "dai18_probe"
OUT20 = ROOT / "results" / "dai20_eval"
OUT_PROD = ROOT / "results" / "dai21_eval5"
DESIGN = ROOT / "docs" / "設計" / "実験21_設計.md"
FROZEN = ROOT / "frozen" / "dai21_frozen.json"
FROZEN20 = ROOT / "frozen" / "dai20_frozen.json"
C4 = tuple(DD.CORPORA)
CD, HOLD = "CREMA-D", "RAVDESS"          # CREMA-D の単位は，実験 18 の単位の作り（Unit6）で RAVDESS を除く側に置く．ソースは 4 コーパスになる
KS = (5, 10, 25)
NF = P16.NF
GRID = P16.GRID
NG = len(GRID)
SGRID = P18.SGRID                        # 学び直しの設定の格子：(ρ, a) の 12 点（実験 1 と同じ）
NS = len(SGRID)
SEED = 20261114 + 21
WF_MS = P20.WF_MS
N_WF = P20.N_WF
AGREE_MIN = P18.NONE_AGREE               # 保存の確率（float32）から作り直した判定と保存の判定の一致の下限（0.995）
SEED_SETS = {"all": None, "0-1": (0, 1), "2-9": tuple(range(2, 10))}
EXTRA_T = ("noprof", "noint", "levelfree", "konly", "tcal")     # 主の設定の点だけで学び直す疑似ラベル（提案の部品を外した表 3 つ，K 本だけで測った表，分類器ごとの温度の較正）
EX1 = ("kd", "agree", "mix")             # 実験 1 の手法のうち，積の疑似ラベルを受け取る 3 つ（4 コーパスで，疑似ラベルだけを提案に替えた版を足す）
FULL_K = 25                              # CREMA-D の上限（候補の全ラベルで学ぶ分類器）に使う引き戻しの強さは，K = 25 の値
N_SEL_CD = 40                            # CREMA-D の設定を選んだ単位の数（実験 18．CREMA-D を触らずに残し，分類器が 2 つ以上の仮のターゲットの単位）
TEMP_MODES = P20.TEMP_MODES
B_VERS = P20.B_VERS
A2 = ("prod", "avg", "conf", "M_F2_K", "M_F6_K", "cf_D_l05", "tcal", "temp", "tempb") + tuple(f"wf{m}" for m in WF_MS)     # CREMA-D で比べる相手（別のコーパスで選んだ指数の積は無い）
SCRIPTS = ("dai20_eval.py", "dai18_probe.py", "dai17_probe.py", "dai16_probe.py", "dai15.py", "dai14.py", "dai12_probe.py", "dai11_probe.py", "dai3.py", "dai1.py",
           "dai1_stats.py", "dai_solver.py", "dai_data.py")
RUN = {"commit": "", "dirty": True}
WATCH = tuple(f"exp/{n}" for n in SCRIPTS) + ("exp/dai21_eval5.py", "frozen/dai21_frozen.json", "frozen/dai20_frozen.json", "splits", "measure", "features",
                                                  "data/metadata", "requirements.txt", "docs/設計/実験21_設計.md")
FIXED = ("frozen/dai21_frozen.json", "frozen/dai20_frozen.json", "exp/dai16_probe.py", "exp/dai18_probe.py")   # 集計の段が，単位を作った後に変わっていないことを確かめるもの（選んだ点と格子）


def same_code(commit: str, other: str = "HEAD", paths=WATCH) -> bool:
    """2 つのコミットのあいだで，計算に関わるファイルに差が無いか（実験 20 の same_code と同じ形）．"""
    if str(commit) == str(other) or (other == "HEAD" and commit == RUN["commit"]):
        return True
    return subprocess.run(["git", "diff", "--quiet", str(commit), str(other), "--", *paths], cwd=ROOT).returncode == 0


def tables_of(nm: int) -> np.ndarray:
    """重みの表を探す版の表（実験 20 と同じ作り方を，分類器の数 nm に合わせたもの）．3 つなら実験 20 の表そのもの．"""
    if nm == 3:
        return P20.WS
    return np.concatenate([np.ones((1, nm, 4)), nm * np.random.default_rng([SEED, 1]).dirichlet(np.ones(nm), size=(N_WF, 4)).transpose(0, 2, 1)])


def seed_rng(*key) -> np.random.Generator:
    return P16.row_rng("dai21", *key)


# ------------------------------------------------------------------ まとめ方（分類器の数に依らない形）

def wf_search(tables: np.ndarray, LMh: np.ndarray, yk: np.ndarray, fold: np.ndarray, LMpj: list) -> tuple[np.ndarray, int]:
    """実験 20 の wf_search と同じ計算（表を引数で受け取る）：表ごとに，K 本の上の UAR（%）を切片をずらした答えで測る．"""
    dv = P20.dev()
    W = torch.tensor(tables, dtype=torch.float64, device=dv)
    pred = torch.empty((len(tables), len(yk)), dtype=torch.long, device=dv)
    ncap = 0
    for j in range(NF):
        te = np.flatnonzero(fold == j)
        Lj = torch.tensor(LMpj[j], dtype=torch.float64, device=dv)
        tgt = torch.softmax(Lj.sum(0), -1).mean(0)
        step = max(1, int(2.5e7 // (Lj.shape[1] * 4)))
        Lh = torch.tensor(LMh[:, te], dtype=torch.float64, device=dv)
        for a in range(0, len(tables), step):
            Sp = torch.einsum("rky,kny->rny", W[a:a + step], Lj)
            Sp = Sp - torch.logsumexp(Sp, -1, keepdim=True)
            d, nc = P20.match_shift_batch(Sp, tgt)
            ncap += nc
            Sh = torch.einsum("rky,kny->rny", W[a:a + step], Lh)
            Sh = Sh - torch.logsumexp(Sh, -1, keepdim=True)
            pred[a:a + step, te] = (Sh + d[:, None, :]).argmax(-1)
            del Sp
    y = torch.tensor(yk, device=dv)
    hit = (pred == y[None]).double()
    ua = torch.stack([hit[:, y == c].mean(1) for c in range(4) if bool((y == c).any())], 1).mean(1)
    return (100.0 * ua).cpu().numpy(), ncap


def wf_pick(tables: np.ndarray, ua: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    u = ua[:m + 1]
    tied = np.flatnonzero(u >= u.max() - 1e-9)
    return tables[tied].mean(0), tied


def combine_all(arr: dict, rows: list, K: int, Pt: list, Pp: list, Ph: list, P2: list, PJ: list, kpos, fold, yk, yt, yp,
                DK, DO, Dext, Dhome, tables: np.ndarray) -> None:
    """分類器の答え（Pt：テスト，Pp：候補，Ph：K 本の抜いた確率，P2：2 組を抜いた確率，PJ：組 j を抜いた分類器の候補の上の確率）から，
    提案・比べる相手・部品を外した版の判定を作る．式は実験 16 の run_unit と実験 20 の add_extra と同じ（分類器の数だけを一般にした）．"""
    nm = len(Pt)
    nk = np.setdiff1d(np.arange(len(yp)), kpos)
    pairs = list(itertools.combinations(range(NF), 2))
    pidx = {p: n for n, p in enumerate(pairs)}
    LMt, LMp, LMh = (np.stack([P16.centered(p) for p in Ps]) for Ps in (Pt, Pp, Ph))
    LMpj = [np.stack([P16.centered(PJ[k][j]) for k in range(nm)]) for j in range(NF)]
    target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)

    def put(v: str, st: np.ndarray, sp: np.ndarray, **extra) -> None:
        St, Sp = P16.norm_log(st), P16.norm_log(sp)
        if not (np.isfinite(St).all() and np.isfinite(Sp).all()):
            raise SystemExit(f"まとめた点が有限でない（{v} K={K}）")
        dl, rs = P16.match_shift(Sp, target)
        pr, pj, pp = St.argmax(1), (St + dl).argmax(1), (Sp + dl)[nk].argmax(1)
        arr[f"pred_raw_K{K}_{v}"], arr[f"pred_adj_K{K}_{v}"], arr[f"pred_pool_K{K}_{v}"] = pr.astype(np.int8), pj.astype(np.int8), pp.astype(np.int8)
        rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pj), uar_pool=100 * S.uar(yp[nk], pp), adj_resid=rs, **extra))

    def f6(W: np.ndarray, LM: np.ndarray) -> np.ndarray:
        return np.einsum("ky,kny->ny", W, LM)

    put("prod", LMt.sum(0), LMp.sum(0))
    Dm = {"K": DK, "ext": Dext, "orc": DO, **{f"mix{i}": P16.mix_rel(DK, Dext, 4 * K, n0, b_, pr) for i, (n0, b_, pr) in enumerate(GRID)}}
    for r_, D in Dm.items():
        W = P16.to_w(D)
        put(f"M_F6_{r_}", f6(W, LMt), f6(W, LMp))
        if r_ == "orc" or r_.startswith("mix"):
            Wn = P16.noint(W)
            put(f"M_F6n_{r_}", f6(Wn, LMt), f6(Wn, LMp))
    w2 = P16.to_w(DK).mean(1)
    put("M_F2_K", np.tensordot(w2, LMt, axes=1), np.tensordot(w2, LMp, axes=1))
    tabs = [P16.shaped(dai15.table_from(Ph[k], yk), "D", 0.5) for k in range(nm)]
    put("cf_D_l05", sum(P16.lg(P16.reread(Pt[k], tabs[k][0])) for k in range(nm)), sum(P16.lg(P16.reread(Pp[k], tabs[k][0])) for k in range(nm)),
        n_unused=int(sum(t_[1] for t_ in tabs)))
    put("avg", P16.lg(np.mean(Pt, 0)), P16.lg(np.mean(Pp, 0)))
    ct, cp = np.stack([p.max(1) for p in Pt]), np.stack([p.max(1) for p in Pp])
    put("conf", P16.lg((ct[:, :, None] * np.stack(Pt)).sum(0) / ct.sum(0)[:, None]), P16.lg((cp[:, :, None] * np.stack(Pp)).sum(0) / cp.sum(0)[:, None]))
    for v, mode in TEMP_MODES.items():
        w = P20.temp_w(LMh, yk, mode)
        est = np.empty((len(kpos), 4))
        for j in range(NF):
            te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
            Ltr = np.stack([P16.centered(np.stack([P2[k][pidx[tuple(sorted((int(fold[a]), j)))]][a] for a in tr])) for k in range(nm)])
            wj = P20.temp_w(Ltr, yk[tr], mode)
            dl, _ = P16.match_shift(P16.norm_log(np.tensordot(wj, LMpj[j], axes=1)), np.exp(P16.norm_log(LMpj[j].sum(0))).mean(0))
            est[te] = P16.norm_log(np.tensordot(wj, LMh[:, te], axes=1) + dl)
        put(v, np.tensordot(w, LMt, axes=1), np.tensordot(w, LMp, axes=1), w_bound=float(((w <= 1e-9) | (w >= P16.W_MAX - 1e-9)).mean()),
            est_uar=100 * S.uar(yk, est.argmax(1)), est_ll=P16.bal_ll(est, yk), **{f"w_{k}": float(w[k]) for k in range(nm)})
        arr[f"W_{v}_K{K}"] = w
    ua, ncap = wf_search(tables, LMh, yk, fold, LMpj)
    arr[f"wf_uar_K{K}"] = ua.astype(np.float32)
    for m in WF_MS:
        W, tied = wf_pick(tables, ua, m)
        ex = dict(wf_index=int(tied[0]), wf_n_tied=len(tied), wf_uar_k=float(ua[tied[0]]), wf_uar_k_eq=float(ua[0]), wf_n_capped=ncap)
        arr[f"wf_tied{m}_K{K}"] = tied.astype(np.int16)
        if len(tied) == 1 and tied[0] == 0:                                      # 等しい重みの表だけが最大：積の判定をそのまま使う
            for w_ in ("raw", "adj"):
                arr[f"pred_{w_}_K{K}_wf{m}"] = arr[f"pred_raw_K{K}_prod"]
            arr[f"pred_pool_K{K}_wf{m}"] = arr[f"pred_pool_K{K}_prod"]
            u_ = 100 * S.uar(yt, arr[f"pred_raw_K{K}_prod"].astype(int))
            rows.append(dict(K=K, version=f"wf{m}", uar_raw=u_, uar_adj=u_, uar_pool=100 * S.uar(yp[nk], arr[f"pred_pool_K{K}_prod"].astype(int)), adj_resid=0.0, **ex))
        else:
            put(f"wf{m}", f6(W, LMt), f6(W, LMp), **ex)
        arr[f"W_wf{m}_K{K}"] = W
    for i, (n0, beta, prior) in enumerate(GRID):                              # 相性をソースの中の値（残り方を掛けない）から取る版
        if beta == 1.0:
            lev = P16.mix_rel(DK, Dext, 4 * K, n0, 0.0, prior)
            Wh = P16.to_w(np.maximum(0.0, lev + (Dhome - Dhome.mean(1, keepdims=True))))
            put(f"M_F6h_mix{i}", f6(Wh, LMt), f6(Wh, LMp))


def teacher_probs(LMp: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """疑似ラベル（候補の上の確率）．W が None なら積，そうでなければ提案（重み W で点を足し，切片をずらす）．実験 18 の teacher_pool と同じ式．"""
    if W is None:
        return np.exp(P16.norm_log(LMp.sum(0)))
    Sp = P16.norm_log(np.einsum("ky,kny->ny", W, LMp))
    dl, _ = P16.match_shift(Sp, np.exp(P16.norm_log(LMp.sum(0))).mean(0))
    return np.exp(P16.norm_log(Sp + dl))


def teachers_of(LMp: np.ndarray, DK: np.ndarray, Dext: np.ndarray, K: int, i: int, w_tcal: np.ndarray) -> dict:
    """学び直しに使う疑似ラベル（候補の上の確率）．どれも，提案と同じ切片のずらしを当てた確率．
    prod：積．prop：提案（格子の点 i）．noprof：提案から相性を外した表（同じ n0・戻し先で β = 0）．noint：提案の重みから分類器 × 感情の成分を消した表．
    levelfree：分類器どうしの水準の違いを外した表（格子の点 14）．konly：K 本だけで測った表．tcal：分類器ごとの温度の較正（K 本で合わせた重み）．"""
    n0, b_, pr = GRID[i]
    Wp = P16.to_w(P16.mix_rel(DK, Dext, 4 * K, n0, b_, pr))
    return {"prod": teacher_probs(LMp, None), "prop": teacher_probs(LMp, Wp),
            "noprof": teacher_probs(LMp, P16.to_w(P16.mix_rel(DK, Dext, 4 * K, n0, 0.0, pr))), "noint": teacher_probs(LMp, P16.noint(Wp)),
            "levelfree": teacher_probs(LMp, P16.to_w(P16.mix_rel(DK, Dext, 4 * K, float("inf"), 1.0, "eq"))), "konly": teacher_probs(LMp, P16.to_w(DK)),
            "tcal": teacher_probs(LMp, np.repeat(np.asarray(w_tcal, float)[:, None], 4, 1))}


def student_specs(gm: int, ex1: dict | None = None) -> list:
    """学び直す問の一覧 [(名前, 手法, 疑似ラベル, 設定)]．
    今の提案の式（joint）：積と提案の疑似ラベルは設定の格子の 12 点すべて，ほかの疑似ラベルは主の点 gm だけ．
    ex1 を渡すと（4 コーパスだけ），実験 1 の 3 手法（温度つき蒸留 kd，一致した発話だけ agree，混ぜる手 mix）に，実験 1 が固定した設定のまま，
    積と提案の疑似ラベルを渡した問を足す（積を渡した問は実験 1 の保存の再現の検算に使う）．"""
    sp = [(f"stud_{t_}_g{gi}", "joint", t_, dict(rho=SGRID[gi][0], a=SGRID[gi][1])) for t_ in ("prod", "prop") for gi in range(NS)]
    sp += [(f"stud_{t_}_g{gm}", "joint", t_, dict(rho=SGRID[gm][0], a=SGRID[gm][1])) for t_ in EXTRA_T]
    if ex1:
        sp += [(f"{m_}_{t_}", m_, t_, dict(ex1[m_])) for m_ in EX1 for t_ in ("prod", "prop")]
    return sp


def fit_students(X, Xt, W0, b0, cw, cb, gk, n: int, own, teachers: dict, specs: list, label: str):
    """実験 1 の段 3 と同じ式と同じ解き方（dai1.make_G で重みを作り，まとめて解く）．own：その分類器の K 本だけの版の候補の上の確率（agree・mix が使う）．
    戻り：{名前: (テストの判定, 候補の判定)}，重み W・b（float32），勾配の最大，止まりの数．"""
    un = types.SimpleNamespace(n=n)
    G, G2, T2 = zip(*[dai1.make_G(un, m_, S.t(teachers[t_]), own, gk, dict(it)) for _, m_, t_, it in specs])
    res = S.fit(X, torch.stack(G), W0, b0, Cw=cw, Cb=cb, G2=torch.stack(G2), T2=np.array(T2, float))
    if float(res["grad_max"]) > P16.GRAD_TOL or not res["finite_p"].all():
        raise SystemExit(f"学び直しが解けていない（{label}．勾配の最大 {float(res['grad_max']):.1e}）")
    Zt, Zp = S.logits(Xt, res["W"], res["b"]).argmax(-1).cpu().numpy(), S.logits(X, res["W"], res["b"]).argmax(-1).cpu().numpy()
    return ({sp[0]: (Zt[q], Zp[q]) for q, sp in enumerate(specs)}, res["W"].to(torch.float32).cpu().numpy(), res["b"].to(torch.float32).cpu().numpy(),
            float(res["grad_max"]), int(res["stalled"]))


def save_students(arr: dict, rows: list, K: int, out: dict, Wb: tuple, specs: list, teach: dict, yt, yp, nk) -> None:
    arr[f"studW_K{K}"], arr[f"studb_K{K}"] = Wb                              # 学び直した融合の分類器の重み（specs の並び）
    arr[f"stud_items_K{K}"] = np.array([sp[0] for sp in specs])
    for name, m_, t_, it in specs:
        pt, pp = out[name]
        arr[f"pred_K{K}_{name}"], arr[f"predpool_K{K}_{name}"] = pt.astype(np.int8), pp.astype(np.int8)     # テストの判定と，候補の上の判定（K 本の位置を含む全部）
        gi = int(name.rsplit("_g", 1)[1]) if m_ == "joint" else -1
        rows.append(dict(K=K, version=(f"stud_{t_}" if m_ == "joint" else name), method=m_, teacher=t_, grid=gi, rho=it.get("rho", np.nan), a=it.get("a", np.nan),
                         uar=100 * S.uar(yt, pt), uar_pool=100 * S.uar(yp[nk], pp[nk]), mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean()),
                         teacher_uar=100 * S.uar(yp[nk], teach[t_][nk].argmax(1))))


# ------------------------------------------------------------------ 固定のファイル

def sgrid_index(rho, a) -> int:
    return SGRID.index((float(rho), int(float(a))))


def stage_freeze() -> None:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} が既にある（作り直すなら消してから）")
    fz1, _ = dai3.load_fz1(OUT1)
    f20 = json.load(open(FROZEN20))
    ssel = pd.read_csv(OUT18 / "summary" / "students_sel.csv")
    fsel = json.load(open(OUT18 / "frozen_sel.json"))["choice"]
    teth18 = json.load(open(OUT18 / "tether.json"))["tether"]

    def sel18(c, K, student, teacher):
        q = ssel[(ssel.c == c) & (ssel.K == K) & (ssel.student == student) & (ssel.teacher == teacher)]
        assert len(q) == 1, (c, K, student, teacher, len(q))
        return sgrid_index(q.iloc[0].rho, q.iloc[0].a)

    four = {}
    for T in C4:
        for K in KS:
            j = fz1[T]["select"][f"AVT|{K}|joint|one"]
            assert fz1[T]["main"][str(K)] == "one"
            four.setdefault(T, {})[str(K)] = dict(prop_grid=int(f20["choice"][T][str(K)]["grid"]), student_main=sgrid_index(j["rho"], j["a"]),
                                                  student_prop_sel=sel18(T, K, "AVT", "M_F6_mixsel"), student_prod_sel=sel18(T, K, "AVT", "prod"))
    # CREMA-D：実験 18 が，CREMA-D を触らずに残した単位（T = CREMA-D）だけで選んだ値
    ud = OUT18 / "units"
    cd = {}
    for K in KS:
        v = fsel[f"{CD}|{K}|M_F6_mixsel"]
        i = int(v.replace("M_F6_mix", ""))
        rows, seen, n = [], set(), 0
        for p in sorted(ud.glob(f"{CD}_*.csv")):
            d = pd.read_csv(p, low_memory=False)
            d = d[(d.K == K) & d.version.astype(str).str.fullmatch(r"M_F6_mix\d+")]
            if not len(d):
                continue                                                     # 分類器が 1 つの仮のターゲット（RAVDESS）の単位には提案の版が無い
            z = np.load(p.with_suffix(".npz"), allow_pickle=False)
            seen |= {str(x) for k_ in z.files if k_.startswith("srcs_") for x in z[k_]} | set(d.c.astype(str))
            assert (d["T"] == CD).all()
            rows.append(d[["version", "uar_adj"]]); n += 1
        g = pd.concat(rows).groupby("version").uar_adj.mean()
        vals = [float(g[f"M_F6_mix{q}"]) for q in range(NG)]
        v0 = [vals[q] if GRID[q][1] == 0.0 else -np.inf for q in range(NG)]
        if int(np.argmax(vals)) != i or CD in seen or n != N_SEL_CD:
            raise SystemExit(f"CREMA-D の選び直しが合わない：K={K} 保存 {i} 選び直し {int(np.argmax(vals))} 単位 {n} 現れたコーパス {sorted(seen)}")
        cd[str(K)] = dict(prop_grid=i, noprof_sel=int(np.argmax(v0)), n_units=n, corpora_in_units=sorted(seen),
                          student_main=sel18(CD, K, "AV", "prod"), student_prop_sel=sel18(CD, K, "AV", "M_F6_mixsel"),
                          tether={k: teth18[CD][f"{k}|{K}"] for k in ("A", "V", "AV")})
    srcs = [OUT18 / "summary" / "students_sel.csv", OUT18 / "frozen_sel.json", OUT18 / "tether.json", FROZEN20, OUT1 / "select" / "frozen.json"]
    pts = [q[k_] for d_ in list(four.values()) + [cd] for q in (d_.values() if d_ is not cd else cd.values()) for k_ in q if k_.startswith("student_")]
    json.dump(dict(four=four, cremad=cd, sgrid=[[r, a] for r, a in SGRID], n_student_points=len(pts),
                   n_student_points_at_rho_edge=int(sum(SGRID[g_][0] in (min(r for r, _ in SGRID), max(r for r, _ in SGRID)) for g_ in pts)), source_sha256={str(p.relative_to(ROOT)): dai15.sha(p) for p in srcs if p.exists()},
                   wf_tables2_sha256=hashlib.sha256(tables_of(2).tobytes()).hexdigest(),
                   rule="学び直しの主の設定は，積を疑似ラベルにした学び直しのために選ばれた値（4 コーパスは実験 1，CREMA-D は実験 18 が CREMA-D を含まない単位で選んだ値）．"
                        "提案の設定と引き戻しの強さも，評価するコーパスを含まない単位で選んだ値",
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=RUN["commit"]), open(FROZEN, "w"), ensure_ascii=False, indent=1)
    print(f"freeze：{FROZEN}", flush=True)


def load_frozen() -> dict:
    fz = json.load(open(FROZEN))
    for p, h in fz["source_sha256"].items():
        p = p.removeprefix("DAI/")                                       # 整理（2026-11-15）の前の鍵は MER-DA の直下から書いてある
        if (ROOT / p).exists() and dai15.sha(ROOT / p) != h:
            raise SystemExit(f"固定のファイルの元（{p}）が変わっている")
    if fz["sgrid"] != [[r, a] for r, a in SGRID] or fz["wf_tables2_sha256"] != hashlib.sha256(tables_of(2).tobytes()).hexdigest():
        raise SystemExit("固定のファイルの格子か乱数の表が台本と違う")
    return fz


# ------------------------------------------------------------------ 4 コーパスの学び直し

def unit_students4(T: str, f: int, r: int, fz: dict, fz1T: dict) -> tuple[list, dict]:
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)
    u = sv.u
    z = np.load(OUT20 / "units" / f"{T}_ALL_f{f}_r{r}.npz", allow_pickle=False)
    yt, yp = u.ytest, u.ypool
    if not ((z["y_test"].astype(int) == yt).all() and (z["sid_test"].astype(str) == sv.sid).all()):
        raise SystemExit(f"{T} f{f} r{r}：実験 20 の単位のテストが実験 1 の保存と違う")
    rows, arr = [], dict(y_test=yt.astype(np.int8), cl_test=sv.cl, sid_test=sv.sid)
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)                              # 実験 1 の段 1・2 の保存から作り直した確率（K 本だけの融合の確率を agree・mix が使う）
    for K in KS:
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all()
        nk = np.setdiff1d(np.arange(u.n), kpos)
        i = int(fz["four"][T][str(K)]["prop_grid"])
        LMp = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in "AVT"])
        W = None
        del W
        teach = teachers_of(LMp, z[f"D_M_K_K{K}"], z["SR_M_ext"], K, i, z[f"W_tcal_K{K}"])
        chk = (("prod", "prod"), ("prop", f"M_F6_mix{i}"), ("noprof", f"M_F6_mix{P20.same_but(i, beta=0.0)}"), ("noint", f"M_F6n_mix{i}"),
               ("levelfree", f"M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}"), ("konly", "M_F6_K"), ("tcal", "tcal"))
        ag = {t_: float((teach[t_][nk].argmax(1) == z[f"pred_pool_K{K}_{v}"].astype(int)).mean()) for t_, v in chk}
        if min(ag.values()) < AGREE_MIN:
            raise SystemExit(f"{T} f{f} r{r} K={K}：疑似ラベルの作り直しが実験 20 の保存と違う（{ag}）")
        cw, cb = sv.teth[("AVT", K)]
        W0, b0 = u.start["AVT"]
        gm = int(fz["four"][T][str(K)]["student_main"])
        ex1 = {m_: {k_: v_ for k_, v_ in fz1T["select"][f"AVT|{K}|{m_}|one"].items() if k_ != "pseudo_uar"} for m_ in EX1}
        specs = student_specs(gm, ex1)
        out, Wst, bst, gmax, stalled = fit_students(u.Xp["AVT"], u.Xt["AVT"], W0, b0, cw, cb, S.balanced_onehot(yp, kpos, u.n), u.n, TK[K]["none_AVT"][0], teach, specs,
                                                    f"{T} f{f} r{r} K={K}")
        save_students(arr, rows, K, out, (Wst, bst), specs, teach, yt, yp, nk)
        saved = {"joint": z[f"pred_raw_K{K}_joint_AVT"].astype(int), **{m_: z[f"pred_raw_K{K}_b_{m_}"].astype(int) for m_ in EX1}}
        rows.append(dict(K=K, version="_check", grad=gmax, stalled=stalled, agree_prod=ag["prod"], agree_prop=ag["prop"], agree_min=min(ag.values()),
                         joint_repro=float((out[f"stud_prod_g{gm}"][0] == saved["joint"]).mean()),   # 積を疑似ラベルにして学び直した判定が，実験 1 の保存の「今の提案」と一致する割合
                         **{f"{m_}_repro": float((out[f"{m_}_prod"][0] == saved[m_]).mean()) for m_ in EX1},   # 同じく，実験 1 の 3 手法の保存の判定と一致する割合
                         Cw=float(cw), Cb=float(cb), prop_grid=i, student_main=gm, dai20_npz_sha256=dai15.sha(OUT20 / "units" / f"{T}_ALL_f{f}_r{r}.npz")))
    return rows, arr


# ------------------------------------------------------------------ CREMA-D

def unit_cremad(u, teth: dict, sel: dict) -> tuple[list, dict]:
    """CREMA-D の 1 単位．teth：{"種類|K": (C′, C_b)}．sel：{K: (提案の格子の点, 学び直しの主の点)}．土台は実験 18 の関数（fit_none，ret_models，ext_rel）をそのまま呼ぶ．"""
    rows, arr = [], {}
    mods, nm = u.mods, len(u.mods)
    yt, yp = u.ytest, u.ypool
    arr["y_test"], arr["cl_test"], arr["sid_test"], arr["y_pool"], arr["cl_pool"] = yt.astype(np.int8), u.cl, u.sid, yp.astype(np.int8), u.cl_pool
    arr["sid_pool"] = u.cp.df["sample_id"].to_numpy().astype(str)[u.pool]
    for kind in u.kinds:
        arr[f"srcs_{kind}"] = np.array(u.srcs[kind])
    ext = P18.ext_rel(u.srcs, mods)
    Dext, Dhome = np.stack([ext[m][0] for m in mods]), np.stack([ext[m][2] for m in mods])
    arr["D_M_ext"], arr["D_M_home"], arr["ret_M"] = Dext, Dhome, np.array([ext[m][1] for m in mods])
    pairs = list(itertools.combinations(range(NF), 2))
    tables = tables_of(nm)
    for kind in u.kinds:                                                        # 出発点（ソースの分類器の平均．K = 0）と，上限（候補の全ラベルで学んだ分類器．参照）の判定
        p0 = S.logits(u.Xt[kind], *u.start[kind]).argmax(1).cpu().numpy()
        arr[f"pred_raw_K0_start_{kind}"] = p0.astype(np.int8)
        rows.append(dict(K=0, version=f"start_{kind}", uar_raw=100 * S.uar(yt, p0)))
        rf = P18.fit_none(u, kind, FULL_K, teth, [np.arange(u.n)])               # 候補の全ラベル．出発点への引き戻しの強さは K = 25 の値を使う
        pf = S.logits(u.Xt[kind], rf["W"][0], rf["b"][0]).argmax(1).cpu().numpy()
        arr[f"pred_raw_K0_full_{kind}"] = pf.astype(np.int8)
        rows.append(dict(K=0, version=f"full_{kind}", uar_raw=100 * S.uar(yt, pf)))
    for K in KS:
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        fold = dai15.kfold_of(yk)
        assert all(np.bincount(fold[yk == i], minlength=NF).min() >= 1 for i in range(4)), "組に感情が欠ける"
        sets = [kpos] + [kpos[fold != j] for j in range(NF)] + [kpos[(fold != i) & (fold != j)] for i, j in pairs]
        Rp, Rt = P18.ret_models(u, K, teth, sets)
        Pp, Pt = [Rp[m][0] for m in mods], [Rt[m][0] for m in mods]
        Ph = [np.stack([Rp[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in mods]
        P2 = [np.stack([Rp[m][1 + NF + n][kpos] for n in range(len(pairs))]) for m in mods]
        PJ = [Rp[m][1:1 + NF] for m in mods]
        for k, m in enumerate(mods):
            arr[f"p_pool_ret_{m}_K{K}"], arr[f"p_test_ret_{m}_K{K}"] = Pp[k].astype(np.float32), Pt[k].astype(np.float32)
            arr[f"pcf_{m}_K{K}"], arr[f"pcf2_{m}_K{K}"], arr[f"ppj_{m}_K{K}"] = Ph[k].astype(np.float32), P2[k].astype(np.float32), PJ[k].astype(np.float32)
            arr[f"pred_raw_K{K}_ret_{m}"] = Pt[k].argmax(1).astype(np.int8)
            rows.append(dict(K=K, version=f"ret_{m}", uar_raw=100 * S.uar(yt, Pt[k].argmax(1))))
        for kind in u.kinds:                                                    # K 本だけの分類器
            rn = P18.fit_none(u, kind, K, teth)
            pt = S.logits(u.Xt[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
            ppn = S.logits(u.Xp[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
            arr[f"pred_raw_K{K}_none_{kind}"], arr[f"predpool_K{K}_none_{kind}"] = pt.astype(np.int8), ppn.astype(np.int8)
            arr[f"W_none_{kind}_K{K}"], arr[f"b_none_{kind}_K{K}"] = rn["W"][0].to(torch.float32).cpu().numpy(), rn["b"][0].to(torch.float32).cpu().numpy()
            rows.append(dict(K=K, version=f"none_{kind}", uar_raw=100 * S.uar(yt, pt), uar_pool=100 * S.uar(yp[nk], ppn[nk])))
        DK = np.stack([P16.dprime(Ph[k], yk)[0] for k in range(nm)])
        DO = np.stack([P16.dprime(Pp[k][nk], yp[nk])[0] for k in range(nm)])
        arr[f"D_M_K_K{K}"], arr[f"D_M_orc_K{K}"], arr[f"kpos_K{K}"], arr[f"kfold_K{K}"] = DK, DO, kpos.astype(np.int32), fold.astype(np.int8)
        combine_all(arr, rows, K, Pt, Pp, Ph, P2, PJ, kpos, fold, yk, yt, yp, DK, DO, Dext, Dhome, tables)
        # 学び直し（融合の分類器）
        LMp = np.stack([P16.centered(p) for p in Pp])
        i, gm = sel[K]
        teach = teachers_of(LMp, DK, Dext, K, i, arr[f"W_tcal_K{K}"])
        cw, cb = teth[f"{u.fus}|{K}"]
        W0, b0 = u.start[u.fus]
        specs = student_specs(gm)                                               # CREMA-D には実験 1 の手法の固定の設定が無いので，今の提案の式だけ
        out, Wst, bst, gmax, stalled = fit_students(u.Xp[u.fus], u.Xt[u.fus], W0, b0, cw, cb, u.gK(K), u.n, None, teach, specs, f"{u.name()} K={K}")
        save_students(arr, rows, K, out, (Wst, bst), specs, teach, yt, yp, nk)
        rows.append(dict(K=K, version="_check", grad=gmax, stalled=stalled, Cw=float(cw), Cb=float(cb), prop_grid=i, student_main=gm,
                         gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
    return rows, arr


def cremad_teth(fz: dict) -> dict:
    return {f"{k}|{K}": tuple(v) for K in KS for k, v in fz["cremad"][str(K)]["tether"].items()}


def cremad_sel(fz: dict) -> dict:
    return {K: (int(fz["cremad"][str(K)]["prop_grid"]), int(fz["cremad"][str(K)]["student_main"])) for K in KS}


# ------------------------------------------------------------------ 段の回し方

def loop(out: Path, sub: str, todo: list, fn) -> None:
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for name, args in todo:
        if (d / f"{name}.csv").exists():
            if all(same_code(x) for x in pd.read_csv(d / f"{name}.csv", low_memory=False).commit.unique()):
                print(f"{sub} {name}：済み（飛ばす）", flush=True)
                continue
            raise SystemExit(f"{sub} {name} が，計算に関わるファイルの違うコミットで既にある")
        t0 = time.time()
        rows, arr = fn(*args)
        np.savez_compressed(d / f"{name}.npz", **arr)                       # npz を先に書き，csv を済みの印にする
        df = pd.DataFrame(rows).assign(name=name, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0, host=platform.node(), gpu=gpu)
        df.to_csv(d / f"{name}.csv", index=False)
        ck = df[df.version == "_check"]
        print(f"{sub} {name}：{time.time() - t0:.0f} 秒．行 {len(df)}．勾配の最大 {ck.grad.max():.1e}" +
              (f"．今の提案の再現 {ck.joint_repro.min():.4f}．疑似ラベルの一致 {ck.agree_min.min():.4f}" if "joint_repro" in ck else ""), flush=True)
        del rows, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def todo_students4(targets, folds, seeds) -> list:
    fz1, _ = dai3.load_fz1(OUT1)
    return [(f"{T}_ALL_f{f}_r{r}", (T, f, r)) for T in (targets or C4) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))
            if (folds is None or f in folds) and (seeds is None or r in seeds)]


def todo_cremad(folds, seeds) -> list:
    return [(f"{CD}_ALL_f{f}_r{r}", (f, r)) for f in range(DD.N_FOLDS) for r in range(10) if (folds is None or f in folds) and (seeds is None or r in seeds)]


# ------------------------------------------------------------------ 集計

def role_of(v: str) -> str:
    """表の行の役．proposal：提案（推定時の答えと，学び直した融合）．current：今の提案（積の疑似ラベルで学び直した融合）．konly_fusion：K 本だけの融合．
    teacher_ablation：疑似ラベルを替えた学び直し．selection：設定を別の規則で選んだ版（記述）．B：実験 1 の比較手法．"""
    b = v.split("@")[0]
    if b in ("prop", "stud_prop"):
        return "proposal"
    if b in ("stud_prod", "joint_AVT"):
        return "current"
    if b in ("none_AVT", "none_AV"):
        return "konly_fusion"
    if b in tuple(f"stud_{t_}" for t_ in EXTRA_T):
        return "teacher_ablation"
    if b in tuple(f"{m_}_prop" for m_ in EX1):
        return "B_new_teacher"                                              # 実験 1 の手法に提案の疑似ラベルを渡した版
    if b in tuple(f"{m_}_prod" for m_ in EX1):
        return "B_repro"                                                    # 実験 1 の手法をこの台本で作り直した版（保存の再現の検算）
    if b.startswith("full_"):
        return "upper"
    if b.startswith("start_"):
        return "start"
    if b in A2:
        return "A"
    if b in P20.ABL:
        return "ablation"
    if b.endswith("_sel"):
        return "selection"
    if b == "M_F6_orc":
        return "oracle"
    if b == "M_F6_ext":
        return "reference"
    if b.startswith(("ret_", "none_")):
        return "single"
    return "B"


def table_block(cells: dict, con: list, geff: dict, part: str, psig: dict) -> tuple[list, list, list]:
    """cells：{(コーパス, K, 版): {(分割, 種): 塊ごとの数}}．水準と差を，実験 20 の集計と同じ区間の作り方で出す．
    psig：{(コーパス, K, 版): テストの判定の並びの sha256}．same_as は正解の数が同じ版の印，same_pred は判定そのものが同じ版の印である．"""
    def eff(A, B, c, k_):
        if set(A) != set(B):
            raise SystemExit(f"差の単位がそろわない：{k_}")
        es = ST.effect_stats(A, B, geff[c], seed_rng(*k_))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"], n_folds=es["n_folds"],
                    folds_dropped=es.get("folds_dropped", ""), v_cluster=es.get("v_jk", np.nan), v_seed=es.get("v_seed", np.nan), v_seed_all=es.get("v_seed_all", np.nan),
                    mark={"上回る": "↑", "下回る": "↓"}.get(P16.verdict(es["lo"], es["hi"], es["effect"]), ""))

    def zero(A):
        return {k_: (x[0], np.zeros_like(x[1]), x[2]) for k_, x in A.items()}

    def sub(A, seeds):
        return A if seeds is None else {k_: x for k_, x in A.items() if k_[1] in seeds}

    def only(A, i):                                                          # 感情 i の列だけ残した数（再現率とその区間は 4 倍する．実験 20 と同じ）
        m = np.zeros(4); m[i] = 1.0
        return {k_: (x[0], x[1] * m, x[2]) for k_, x in A.items()}

    same_as = {}
    for c, K in sorted({(c, K) for c, K, _ in cells}):
        vs = [v for (c2, K2, v) in cells if c2 == c and K2 == K]
        first, size, sig = {}, {}, {}
        for v in vs:
            dd = cells[(c, K, v)]
            sig[v] = hashlib.sha256(b"".join(repr(k_).encode() + np.ascontiguousarray(dd[k_][1]).tobytes() for k_ in sorted(dd))).hexdigest()
            first.setdefault(sig[v], v)
            size[sig[v]] = size.get(sig[v], 0) + 1
        for v in vs:
            same_as[(c, K, v)] = first[sig[v]] if size[sig[v]] > 1 else ""
    same_pred = {}                                                           # 判定の並びが全部の単位で同じ版（正解の数が同じでも，誤りの行き先が違えばこの印は付かない）
    for c, K in sorted({(c, K) for c, K, _ in cells}):
        vs = [v for (c2, K2, v) in cells if c2 == c and K2 == K]
        dg = {v: psig[(c, K, v)].hexdigest() for v in vs}
        first, size = {}, {}
        for v in vs:
            first.setdefault(dg[v], v)
            size[dg[v]] = size.get(dg[v], 0) + 1
        for v in vs:
            same_pred[(c, K, v)] = first[dg[v]] if size[dg[v]] > 1 else ""
    lv, ef, pe = [], [], []
    for (c, K, v), dd in sorted(cells.items()):
        for sn, ss in SEED_SETS.items():
            ds = sub(dd, ss)
            if not ds or (sn != "all" and len(ds) == len(dd)):
                continue
            e_ = eff(ds, zero(ds), c, ("L", part, c, K, v, sn))
            lv.append(dict(part=part, c=c, K=K, version=v, role=role_of(v), seeds=sn, uar=e_["est"], lo=e_["lo"], hi=e_["hi"], se=e_["se"], df=e_["df"], method=e_["method"],
                           n_fr=e_["n_fr"], same_as=same_as[(c, K, v)], same_pred=same_pred[(c, K, v)]))
        cor, tot = sum(x[1].sum(0) for x in dd.values()), sum(x[2].sum(0) for x in dd.values())
        full = all((x[2].sum(0) > 0).all() for x in dd.values())
        for i in range(4):
            r_ = eff(only(dd, i), zero(dd), c, ("E", part, c, K, v, i)) if full else None
            pe.append(dict(part=part, c=c, K=K, version=v, role=role_of(v), emotion=i, recall=(4 * r_["est"] if full else np.nan), lo=(4 * r_["lo"] if full else np.nan),
                           hi=(4 * r_["hi"] if full else np.nan), recall_pooled=100 * cor[i] / max(tot[i], 1), n=float(tot[i]), n_units=len(dd),
                           n_per_seed=float(tot[i]) * len({k_[0] for k_ in dd}) / len(dd)))
    for a, b in con:
        for c, K in sorted({(c, K) for c, K, _ in cells}):
            A, B = cells.get((c, K, a)), cells.get((c, K, b))
            if A is None or B is None:
                raise SystemExit(f"差の相手が無い：{(part, a, b, c, K)}")
            for sn, ss in SEED_SETS.items():
                As, Bs = sub(A, ss), sub(B, ss)
                if not As or (sn != "all" and len(As) == len(A)):
                    continue
                ef.append(dict(part=part, a=a, b=b, c=c, K=K, seeds=sn, role_a=role_of(a), role_b=role_of(b), a_same_as=same_as[(c, K, a)], b_same_as=same_as[(c, K, b)],
                               same_pred=bool(psig[(c, K, a)].hexdigest() == psig[(c, K, b)].hexdigest()), **eff(As, Bs, c, ("A", part, a, b, c, K, sn))))
    return lv, ef, pe


def units1_digest() -> str:
    """実験 1 の評価の回の単位の保存（段 1〜3 の npz）の名前と sha256 を並べた文字列の sha256．学び直しの段の入力の記録に使う．"""
    fs = sorted((OUT1 / "eval" / "units").glob("*_ALL_f*_r*_p[123].npz"))
    return f"{len(fs)}:" + hashlib.sha256("".join(f"{q.name} {dai15.sha(q)}\n" for q in fs).encode()).hexdigest()


def stage_summary(out: Path, allow_partial: bool) -> None:
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    fz = load_frozen()
    f20 = P20.load_frozen()
    lv, ef, pe, grid_rows, chk, conf, wrows = [], [], [], [], {}, {}, []
    ST_EX = tuple(f"stud_{t_}" for t_ in EXTRA_T)

    def check_commits(cs, sub_):
        """単位どうしで計算に関わるファイルに差が無いことと，単位を作った後に固定のファイルと格子の台本が変わっていないことを確かめる．"""
        ucom = sorted(map(str, cs.commit.unique()))
        if not allow_partial and not all(same_code(x, ucom[0]) for x in ucom):
            raise SystemExit(f"{sub_} の中に，計算に関わるファイルの違うコミットで作った単位が混ざっている")
        if not allow_partial and not all(same_code(x, "HEAD", FIXED) for x in ucom):
            raise SystemExit(f"{sub_} の単位を作った後に，固定のファイルか格子の台本が変わっている")
        if cs.duplicated(["name", "K", "version", "grid"]).any():
            raise SystemExit(f"{sub_} の行が重なっている")
        return ucom

    def add_conf(c, K, v, yt, pred):
        np.add.at(conf.setdefault((c, K, v), np.zeros((4, 4))), (yt, pred.astype(int)), 1)

    # ---- 4 コーパス
    files = sorted((out / "students4").glob("*.csv"))
    if len(files) != len(todo_students4(None, None, None)) and not allow_partial:
        raise SystemExit(f"students4 の出力が {len(files)} 個")
    if files:
        cs = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
        ucom4 = check_commits(cs, "students4")
        cells, psig = {}, {}
        for p in files:
            T, f, r = p.stem.split("_ALL_")[0], int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
            z, z20 = np.load(p.with_suffix(".npz"), allow_pickle=False), np.load(OUT20 / "units" / f"{p.stem}.npz", allow_pickle=False)
            yt = z["y_test"].astype(int)
            assert (z20["y_test"].astype(int) == yt).all() and (z20["cl_test"] == z["cl_test"]).all()
            for K in KS:
                q = fz["four"][T][str(K)]
                i, gm, gs, gps = int(f20[T][K][0]), int(q["student_main"]), int(q["student_prop_sel"]), int(q["student_prod_sel"])
                assert i == int(q["prop_grid"])
                names = {"stud_prop": (z, f"pred_K{K}_stud_prop_g{gm}"), "stud_prod": (z, f"pred_K{K}_stud_prod_g{gm}"),
                         "stud_prop_sel": (z, f"pred_K{K}_stud_prop_g{gs}"), "stud_prod_sel": (z, f"pred_K{K}_stud_prod_g{gps}"),
                         **{f"stud_{t_}": (z, f"pred_K{K}_stud_{t_}_g{gm}") for t_ in EXTRA_T},
                         **{f"{m_}_{t_}": (z, f"pred_K{K}_{m_}_{t_}") for m_ in EX1 for t_ in ("prod", "prop")},
                         "prop": (z20, f"pred_adj_K{K}_M_F6_mix{i}"), "prod": (z20, f"pred_adj_K{K}_prod"),
                         **{v: (z20, f"pred_raw_K{K}_{v}") for v in ("none_AVT", "joint_AVT") + B_VERS}}
                for v, (src, kk) in names.items():
                    P12._add(cells.setdefault((T, K, v), {}), (f, r), ST.counts(yt, src[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
                    psig.setdefault((T, K, v), hashlib.sha256()).update(np.ascontiguousarray(src[kk].astype(np.int8)).tobytes())
                    if v in ("stud_prop", "stud_prod", "joint_AVT", "none_AVT", "prop", "prod"):
                        add_conf(T, K, v, yt, src[kk])
        P12._finish(cells)
        con = ([("stud_prop", b) for b in ("stud_prod", "joint_AVT", "none_AVT", "prop", "prod") + ST_EX + B_VERS]
               + [("stud_prop_sel", "stud_prod_sel"), ("stud_prod", "joint_AVT"), ("joint_AVT", "none_AVT"), ("prop", "joint_AVT")]
               + [(s_, "stud_prod") for s_ in ST_EX] + [(b, "joint_AVT") for b in B_VERS]
               + [("stud_prop", f"{m_}_prop") for m_ in EX1] + [(f"{m_}_prop", f"b_{m_}") for m_ in EX1] + [(f"{m_}_prod", f"b_{m_}") for m_ in EX1]
               + [(f"{m_}_prop", f"{m_}_prod") for m_ in EX1])                 # 同じ手法・同じ設定で，疑似ラベルだけをこの実験の中で替えた差
        a, b, e = table_block(cells, con, {c: ST.g_eff_of(c)[0] for c in C4}, "4corpora", psig)
        lv += a; ef += b; pe += e
        st = cs[cs.version.astype(str).str.startswith("stud_")].assign(T=lambda d: d.name.str.split("_ALL_").str[0])
        grid_rows.append(st.groupby(["T", "K", "version", "grid", "rho", "a"])[["uar", "mimic", "teacher_uar"]].agg(["mean", "count"]).reset_index().rename(columns={"T": "c"}).assign(part="4corpora"))
        ck = cs[cs.version == "_check"]
        chk.update(students4=dict(n_units=len(files), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()), joint_repro_min=float(ck.joint_repro.min()),
                                  joint_repro_mean=float(ck.joint_repro.mean()), joint_repro_ok=bool(ck.joint_repro.min() >= AGREE_MIN),
                                  n_joint_repro_below=int((ck.joint_repro < AGREE_MIN).sum()), agree_min=float(ck.agree_min.min()),
                                  **{f"{m_}_repro_min": float(ck[f"{m_}_repro"].min()) for m_ in EX1}, **{f"n_{m_}_repro_below": int((ck[f"{m_}_repro"] < AGREE_MIN).sum()) for m_ in EX1},
                                  commits=ucom4))
    # ---- CREMA-D
    files = sorted((out / "cremad").glob("*.csv"))
    if len(files) != len(todo_cremad(None, None)) and not allow_partial:
        raise SystemExit(f"cremad の出力が {len(files)} 個")
    if files:
        cs = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
        ucomc = check_commits(cs, "cremad")
        cells, psig = {}, {}
        fk = P18.fus(CD)
        for p in files:
            f, r = int(p.stem.split("_f")[1].split("_r")[0]), int(p.stem.rsplit("_r", 1)[1])
            z = np.load(p.with_suffix(".npz"), allow_pickle=False)
            yt = z["y_test"].astype(int)
            for K in KS:
                q = fz["cremad"][str(K)]
                i, gm, gs = int(q["prop_grid"]), int(q["student_main"]), int(q["student_prop_sel"])
                names = {v: f"pred_adj_K{K}_{v}" for v in A2 + ("M_F6_orc", "M_F6_ext")}
                names.update({f"{v}@raw": f"pred_raw_K{K}_{v}" for v in A2})
                names.update(prop=f"pred_adj_K{K}_M_F6_mix{i}", **{"prop@raw": f"pred_raw_K{K}_M_F6_mix{i}"}, prop_noprof=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, beta=0.0)}",
                             prop_noprof_sel=f"pred_adj_K{K}_M_F6_mix{int(q['noprof_sel'])}", prop_noint=f"pred_adj_K{K}_M_F6n_mix{i}",
                             prop_n0zero=f"pred_adj_K{K}_M_F6_mix{P20.same_but(i, n0=0.0)}", prop_home=f"pred_adj_K{K}_M_F6h_mix{i}",
                             prop_levelfree=f"pred_adj_K{K}_M_F6_mix{GRID.index((float('inf'), 1.0, 'eq'))}",
                             stud_prod=f"pred_K{K}_stud_prod_g{gm}", stud_prop=f"pred_K{K}_stud_prop_g{gm}", stud_prop_sel=f"pred_K{K}_stud_prop_g{gs}",
                             **{f"stud_{t_}": f"pred_K{K}_stud_{t_}_g{gm}" for t_ in EXTRA_T},
                             **{f"none_{k}": f"pred_raw_K{K}_none_{k}" for k in ("A", "V", fk)}, ret_A=f"pred_raw_K{K}_ret_A", ret_V=f"pred_raw_K{K}_ret_V",
                             **{f"full_{k}": f"pred_raw_K0_full_{k}" for k in ("A", "V", fk)}, **{f"start_{k}": f"pred_raw_K0_start_{k}" for k in ("A", "V", fk)})
                for v, kk in names.items():
                    P12._add(cells.setdefault((CD, K, v), {}), (f, r), ST.counts(yt, z[kk].astype(int), z["cl_test"]), f"{p.stem} {v}")
                    psig.setdefault((CD, K, v), hashlib.sha256()).update(np.ascontiguousarray(z[kk].astype(np.int8)).tobytes())
                    if v in ("stud_prop", "stud_prod", f"none_{fk}", "prop", "prod"):
                        add_conf(CD, K, v, yt, z[kk])
                Dk, Do, De = z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["D_M_ext"]
                wrows.append((K, np.stack([P16.to_w(P16.mix_rel(Dk, De, 4 * K, *GRID[i])), P16.to_w(Do), P16.to_w(Dk), P16.to_w(De)])))
        P12._finish(cells)
        nf = f"none_{fk}"
        con = ([("prop", b) for b in A2 + P20.ABL + (nf, "stud_prod", "M_F6_ext")] + [(a, "prod") for a in A2[1:] + ("prop_noprof", "prop_levelfree", "M_F6_orc", "M_F6_ext")]
               + [("prop_noint", "prop_noprof"), ("prop", "prop@raw")] + [("prop@raw", f"{b}@raw") for b in A2] + [("prop", f"{b}@raw") for b in A2[1:]]
               + [("stud_prop", b) for b in ("stud_prod", nf, "prop", "prod") + ST_EX] + [("stud_prop_sel", "stud_prod"), ("stud_prod", nf), ("stud_prod", "prod")]
               + [(s_, "stud_prod") for s_ in ST_EX])
        a, b, e = table_block(cells, con, {CD: P18.g_eff_of(CD)}, "cremad", psig)
        lv += a; ef += b; pe += e
        mods = P18.MODS_OF[CD]
        pd.DataFrame([dict(c=CD, K=K, mod=m, emotion=e_, w_prop=float(W[0, j, e_]), w_orc=float(W[1, j, e_]), w_K=float(W[2, j, e_]), w_ext=float(W[3, j, e_]),
                           n_units=sum(1 for k_, _ in wrows if k_ == K))
                      for K in KS for W in [np.mean([w_ for k_, w_ in wrows if k_ == K], 0)] for j, m in enumerate(mods) for e_ in range(4)]).to_csv(sdir / "weights_cremad.csv", index=False)
        st = cs[cs.version.astype(str).str.startswith("stud_")]
        grid_rows.append(st.groupby(["K", "version", "grid", "rho", "a"])[["uar", "mimic", "teacher_uar"]].agg(["mean", "count"]).reset_index().assign(c=CD, part="cremad"))
        ck = cs[cs.version == "_check"]
        chk.update(cremad=dict(n_units=len(files), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()), adj_resid_max=float(cs.adj_resid.max()),
                               wf_n_capped=int(cs[cs.version == f"wf{N_WF}"].wf_n_capped.fillna(0).sum()), commits=ucomc))
        gq = cs[cs.version.astype(str).str.fullmatch(r"M_F6_mix\d+")].copy()        # 提案の格子の 16 点の UAR（切片をずらした答え．CREMA-D）
        gq["grid"] = gq.version.str.replace("M_F6_mix", "").astype(int)
        gl = gq.groupby(["K", "grid"]).uar_adj.mean().reset_index()
        gl["frozen"] = [int(fz["cremad"][str(int(K))]["prop_grid"]) == int(g_) for K, g_ in zip(gl.K, gl.grid)]
        gl.assign(c=CD, n0=[GRID[int(g_)][0] for g_ in gl.grid], beta=[GRID[int(g_)][1] for g_ in gl.grid], prior=[GRID[int(g_)][2] for g_ in gl.grid]).to_csv(
            sdir / "grid_levels_cremad.csv", index=False)
    pd.DataFrame(lv).to_csv(sdir / "levels.csv", index=False)
    pd.DataFrame(ef).to_csv(sdir / "effects.csv", index=False)
    pd.DataFrame(pe).to_csv(sdir / "per_emotion.csv", index=False)
    pd.DataFrame([dict(c=c, K=K, version=v, true=a_, pred=b_, n=float(M[a_, b_])) for (c, K, v), M in sorted(conf.items()) for a_ in range(4) for b_ in range(4)]).to_csv(
        sdir / "confusion.csv", index=False)                                # n は全部の単位（分割 × 種）の合計
    if grid_rows:                                                           # 設定の格子の点ごとの平均（区間は付けない．テストで最良の点の値は上に偏る）
        g_ = pd.concat(grid_rows, ignore_index=True)
        g_.columns = ["_".join([str(x) for x in col if x != ""]) if isinstance(col, tuple) else col for col in g_.columns]
        g_.to_csv(sdir / "student_grid.csv", index=False)
    json.dump(dict(chk, summary_commit=RUN["commit"]), open(sdir / "convergence.json", "w"), indent=1)
    print(f"summary：水準の行 {len(lv)}，差の行 {len(ef)}．{json.dumps(chk, ensure_ascii=False)[:400]}", flush=True)


# ------------------------------------------------------------------ 検算

def stage_check3() -> None:
    """combine_all を実験 16 の下調べの保存（3 つの分類器）に当て，保存の判定と一致するかを見る（確率は float32 の保存から読むので，同点に近い発話は食い違いうる）．"""
    out = []
    for name in ("AFEW_IEMOCAP_f0_r0", "MELD_AFEW_f1_r1"):
        z = np.load(ROOT / "results" / "dai16_probe" / "units" / f"{name}.npz", allow_pickle=False)
        yt, yp = z["y_test"].astype(int), z["y_pool"].astype(int)
        for K in (5, 25):
            kpos, fold = z[f"kpos_K{K}"], z[f"kfold_K{K}"].astype(int)
            g = lambda pre: [z[f"{pre}_{m}_K{K}"].astype(np.float64) for m in "AVT"]  # noqa: E731
            arr, rows = {}, []
            combine_all(arr, rows, K, g("p_test_ret"), g("p_pool_ret"), g("pcf"), g("pcf2"), g("ppj"), kpos, fold, yp[kpos], yt, yp,
                        z[f"D_M_K_K{K}"], z[f"D_M_orc_K{K}"], z["SR_M_ext"], z["SR_M_home"], tables_of(3))
            vs = ["prod", "M_F6_K", "M_F6_ext", "M_F6_orc", "M_F2_K", "cf_D_l05"] + [f"M_F6_mix{i}" for i in range(NG)] + [f"M_F6n_mix{i}" for i in range(NG)]
            ag = {v: min(float((arr[f"pred_{w}_K{K}_{v}"] == z[f"pred_{w}_K{K}_{v}"]).mean()) for w in ("raw", "adj", "pool")) for v in vs}
            out.append(dict(unit=name, K=K, n_versions=len(vs), agree_min=min(ag.values()), worst=min(ag, key=ag.get), n_exact=sum(v == 1.0 for v in ag.values())))
    d = pd.DataFrame(out)
    print(d.to_string(index=False))
    if d.agree_min.min() < AGREE_MIN:
        raise SystemExit("check3：保存の判定と合わない")
    print("check3：通った")


def stage_check18(out: Path) -> None:
    """CREMA-D の単位の計算を，実験 18 の単位（RAVDESS を除いた回．ソースは 4 コーパス）と同じ引き戻しの強さで流し，保存の判定と照らす．"""
    P18.OUTDIR["out"] = OUT18
    teth = json.load(open(OUT18 / "tether.json"))["tether"][HOLD]
    u = P18.Unit6(HOLD, CD, 0, 0)
    assert all(sorted(u.srcs[k]) == sorted(C4) for k in u.kinds), u.srcs
    t0 = time.time()
    rows, arr = unit_cremad(u, teth, {K: (2, sgrid_index(1.0, 1)) for K in KS})
    z = np.load(OUT18 / "units" / f"{HOLD}_{CD}_f0_r0.npz", allow_pickle=False)
    res = []
    for K in KS:
        vs = ["prod", "M_F6_K", "M_F6_ext", "M_F6_orc"] + [f"M_F6_mix{i}" for i in range(NG)] + [f"M_F6n_mix{i}" for i in range(NG)]
        ag = {v: min(float((arr[f"pred_{w}_K{K}_{v}"] == z[f"pred_{w}_K{K}_{v}"]).mean()) for w in ("raw", "adj", "pool")) for v in vs}
        ag.update({f"none_{k}": float((arr[f"pred_raw_K{K}_none_{k}"] == z[f"pred_raw_K{K}_none_{k}"]).mean()) for k in u.kinds})
        pdev = max(float(np.abs(arr[f"p_test_ret_{m}_K{K}"] - z[f"p_test_ret_{m}_K{K}"]).max()) for m in u.mods)
        res.append(dict(K=K, n_versions=len(ag), agree_min=min(ag.values()), worst=min(ag, key=ag.get), n_exact=sum(v == 1.0 for v in ag.values()), prob_dev_max=pdev))
    d = pd.DataFrame(res)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "check18.csv", index=False)
    print(d.to_string(index=False))
    print(f"check18：{time.time() - t0:.0f} 秒．行 {len(rows)}．配列 {len(arr)}")
    if d.agree_min.min() < AGREE_MIN:
        raise SystemExit("check18：実験 18 の保存と合わない")
    print("check18：通った")


def selftest() -> None:
    assert len(SGRID) == 12 and SGRID[sgrid_index(1.0, 1)] == (1.0, 1) and GRID[14] == (float("inf"), 1.0, "eq")
    assert tables_of(3) is P20.WS
    W2 = tables_of(2)
    assert W2.shape == (N_WF + 1, 2, 4) and np.allclose(W2.sum(1), 2.0) and np.allclose(W2[0], 1.0) and W2.min() >= 0 and np.allclose(W2, tables_of(2))
    rng = np.random.default_rng(0)
    for nm in (2, 3):                                                         # 人工の単位に combine_all を通す
        n, nt, K = 400, 300, 5
        yp, yt = rng.integers(0, 4, n), rng.integers(0, 4, nt)
        kpos = np.sort(np.concatenate([np.flatnonzero(yp == c)[:K] for c in range(4)]))
        yk = yp[kpos]
        fold = dai15.kfold_of(yk)

        def mk(y, m):
            P = rng.dirichlet(np.ones(4), size=(m, len(y)))
            P[:, np.arange(len(y)), y] += 0.5
            return list(P / P.sum(-1, keepdims=True))

        Pt, Pp = mk(yt, nm), mk(yp, nm)
        Ph = [p[kpos] for p in mk(yp, nm)]
        P2 = [np.stack([p[kpos] for p in mk(yp, 10)]) for _ in range(nm)]
        PJ = [np.stack(mk(yp, NF)) for _ in range(nm)]
        DK = np.stack([P16.dprime(Ph[k], yk)[0] for k in range(nm)])
        DO = np.stack([P16.dprime(Pp[k], yp)[0] for k in range(nm)])
        De = rng.uniform(0.3, 1.5, size=(nm, 4))
        arr, rows = {}, []
        combine_all(arr, rows, K, Pt, Pp, Ph, P2, PJ, kpos, fold, yk, yt, yp, DK, DO, De, De * 1.3, tables_of(nm))
        d = pd.DataFrame(rows)
        need = set(A2) | {f"M_F6_mix{i}" for i in range(NG)} | {f"M_F6n_mix{i}" for i in range(NG)} | {"M_F6_orc", "M_F6_ext"} | {f"M_F6h_mix{i}" for i in range(NG) if GRID[i][1] == 1.0}
        assert need <= set(d.version) and not d.duplicated(["K", "version"]).any(), need - set(d.version)
        assert (arr[f"pred_raw_K{K}_prod"] == np.sum([np.log(p) for p in Pt], 0).argmax(1)).all()
        assert (arr[f"pred_adj_K{K}_prod"] == arr[f"pred_raw_K{K}_prod"]).all()            # 積は切片をずらしても同じ判定
        i0 = GRID.index((float("inf"), 0.0, "eq"))                                      # 水準を全部同じにして相性を使わない点は積と同じ
        assert (arr[f"pred_adj_K{K}_M_F6_mix{i0}"] == arr[f"pred_adj_K{K}_prod"]).all()
        assert (arr[f"pred_adj_K{K}_M_F6_mix0"] == arr[f"pred_adj_K{K}_M_F2_K"]).all()     # n0 = 0 で相性を使わない点は水準だけの重みと同じ
        Q0 = teacher_probs(np.stack([P16.centered(p) for p in Pp]), None)
        Q1 = teacher_probs(np.stack([P16.centered(p) for p in Pp]), np.ones((nm, 4)))
        assert np.allclose(Q0.sum(1), 1.0) and np.abs(Q0 - Q1).max() < 1e-9             # 重みが全部 1 の提案の疑似ラベルは積と同じ
    assert role_of("stud_prop") == "proposal" and role_of("wf10@raw") == "A" and role_of("b_kd") == "B" and role_of("none_AV") == "konly_fusion" and role_of("ret_A") == "single"
    assert role_of("prop_noprof_sel") == "ablation" and role_of("stud_prop_sel") == "selection" and role_of("stud_tcal") == "teacher_ablation" and role_of("joint_AVT") == "current"
    sp = student_specs(6)
    assert len(sp) == 2 * NS + len(EXTRA_T) and len({s_[0] for s_ in sp}) == len(sp) and sp[-1][:3] == ("stud_tcal_g6", "joint", "tcal") and sp[6][3] == dict(rho=1.0, a=1)
    sp2 = student_specs(6, {"kd": dict(temp=2.0, rho=1.0), "agree": dict(tau=0.7, f=1.0), "mix": dict(alpha=0.2, f=1.0)})
    assert len(sp2) == len(sp) + 6 and sp2[-1][:3] == ("mix_prop", "mix", "prop")
    assert role_of("kd_prop") == "B_new_teacher" and role_of("mix_prod") == "B_repro" and role_of("full_AV") == "upper" and role_of("stud_noint") == "teacher_ablation"
    LMq = np.stack([P16.centered(p) for p in Pp])
    tq = teachers_of(LMq, DK, De, K, 2, np.ones(nm))                             # 温度の重みが全部 1 なら積と同じ疑似ラベル．n0 = 0 の点 2 から相性を外すと点 0
    assert np.abs(tq["tcal"] - tq["prod"]).max() < 1e-9 and all(np.allclose(q_.sum(1), 1.0) for q_ in tq.values()) and set(tq) == {"prod", "prop"} | set(EXTRA_T)
    assert np.abs(tq["noprof"] - teacher_probs(LMq, P16.to_w(P16.mix_rel(DK, De, 4 * K, *GRID[0])))).max() < 1e-12
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "check3", "check18", "freeze", "students4", "cremad", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--frozen", default=None, help="検査用：固定のファイルの置き場を替える（本番の置き場では使えない）")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    dai1.ENC["enc"] = DD.ENC1
    if args.stage == "check3":
        stage_check3()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    if args.frozen:
        if prod:
            sys.exit("本番の置き場では固定のファイルの置き場を替えない")
        global FROZEN
        FROZEN = Path(args.frozen).resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if args.stage == "check18":
        if prod:
            sys.exit("check18 は本番の置き場に書かない")
        stage_check18(out)
        return
    if args.stage == "freeze":
        stage_freeze()
        return
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある")
    if prod and not (dai1.committed_clean(DESIGN) and dai1.committed_clean(FROZEN) and dai1.committed_clean(Path(__file__).resolve())):
        sys.exit("設計の文書・固定のファイル・この台本のどれかがコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.targets is not None or args.allow_partial):
        sys.exit("本番の置き場では単位を絞らない")
    if prod and not (ROOT / "logs" / "dai20_eval.done").exists():
        sys.exit("実験 20 の計算が終わっていない")
    P18.OUTDIR["out"] = OUT18
    out.mkdir(parents=True, exist_ok=True)
    fz = load_frozen()
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, python=platform.python_version(), this_script_sha256=dai15.sha(Path(__file__).resolve()),
                scripts={n: dai15.sha(ROOT / "exp" / n) for n in SCRIPTS}, design_sha256=(dai15.sha(DESIGN) if DESIGN.exists() else ""),
                frozen_sha256=dai15.sha(FROZEN), frozen20_sha256=dai15.sha(FROZEN20),
                inputs_sha256={**{f"dai1_source/{q.name}": dai15.sha(q) for q in sorted((OUT1 / "source").glob("*.npz"))},
                               **{f"splits/{q.name}": dai15.sha(q) for q in sorted((ROOT / "splits").glob("*.csv"))},
                               "dai18_splits_new.csv": dai15.sha(OUT18 / "splits_new.csv"),
                               **{f"metadata/{q.name}": dai15.sha(q) for q in sorted((ROOT / "data" / "metadata").glob("*.csv"))},
                               **{f"cache/{q.name}": dai15.sha(q) for q in sorted((ROOT / "data" / "cache").glob("*.npz"))}},
                dai1_eval_units_digest=(units1_digest() if args.stage == "students4" else ""), dai20_provenance_sha256=(dai15.sha(OUT20 / "provenance.json") if (OUT20 / "provenance.json").exists() else ""),
                KS=KS, SGRID=[list(x) for x in SGRID], rng_seed=SEED, wf_tables2_sha256=hashlib.sha256(tables_of(2).tobytes()).hexdigest(),
                consts=dict(AGREE_MIN=AGREE_MIN, GRAD_TOL=P16.GRAD_TOL, EXTRA_T=EXTRA_T, EX1=EX1, FULL_K=FULL_K, N_SEL_CD=N_SEL_CD, NF=NF, WATCH=WATCH, FIXED=FIXED))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "students4":
        fz1, _ = dai3.load_fz1(OUT1)
        loop(out, "students4", todo_students4(args.targets, args.folds, args.seeds), lambda T, f, r: unit_students4(T, f, r, fz, fz1[T]))
    elif args.stage == "cremad":
        teth, sel = cremad_teth(fz), cremad_sel(fz)

        def one(f, r):
            u = P18.Unit6(HOLD, CD, f, r)
            assert all(sorted(u.srcs[k_]) == sorted(C4) for k_ in u.kinds), u.srcs      # ソースは元の 4 コーパス
            return unit_cremad(u, teth, sel)

        loop(out, "cremad", todo_cremad(args.folds, args.seeds), one)
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

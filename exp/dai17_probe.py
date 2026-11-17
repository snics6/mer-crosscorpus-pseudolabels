"""実験 17 の下調べ（DAI-17）：ターゲットの発話に対して良いソースの発話は何か．
A（主．発話ごと）：ターゲットの発話ごとに，「特徴が近く，ソースの分類器の答えも近い」ソースの発話を引き，そのラベルの重みつきの票を「近いソースの発話の答え」にする．
   これを分類器として，微調整した声・顔・言葉の分類器の答えとまとめる（実験 16 のまとめ方）．どのソースの発話が役に立つかを，近い発話のラベルが当たる割合で記述する．
B（コーパスごと）：ターゲットの候補からよく引かれるソースの発話と，信頼度の高いソースの発話に重みを付けてソースの分類器を学び直し，出発点と K 本だけの分類器の UAR を比べる．
C（生徒）：A でまとめた答えを疑似ラベルにして，声・顔・言葉・融合の分類器を提案と同じ手順で学び直し，提案（積の疑似ラベル）と比べる．
設計：docs/設計/実験17_下調べ_設計.md．実験 1 の仮のターゲットの回の保存だけを使う．評価の回の保存は開かない．

段：
  selftest   人工の数での検算
  units      A：120 単位（T × 仮のターゲット c × 分割 × 種）．<out>/units/<名前>.npz と .csv
  reweight   B：60 組（T × c × 分割）× 種 2．<out>/reweight/<名前>.npz と .csv
  students   C：120 単位．units の出力から格子の点を選んで固定し（<out>/frozen_sel.json），疑似ラベルを作り直して生徒を学ぶ．<out>/students/
  summary    格子の点の選択（評価するコーパスを含まない単位），c × K ごとの UAR，差と区間，数え，近い発話の表

使い方：.venv/bin/python exp/dai17_probe.py --stage {selftest,units,reweight,students,summary}
試走：--out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW IEMOCAP --folds 0 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # 試走で台本を別の置き場に置いたときのため
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai17_probe"
ADDENDUM = ROOT / "docs" / "設計" / "実験16・17_下調べ_K10_追補.md"
DESIGN = ROOT / "docs" / "設計" / "実験17_下調べ_設計.md"
KS = (5, 25)
K_MAIN = 5
NF = dai15.NF
MODS = "AVT"
KMAX = 100                              # 引くソースの発話の最大の本数
KNN_K = (20, 100)                       # 票に使う本数
# 近さを測る空間：both（特徴のコサインと，ソースの分類器の点のコサインを，問い合わせごとに標準化して平均したもの．主），feat（特徴だけ），logit（点だけ）
# 引く範囲：mod（声・顔・言葉のそれぞれで引く．答えは 3 つ），joint（3 つのモダリティの特徴と点の 6 つのコサインを，同じく標準化して平均したもの．答えは 1 つ）
# 信頼度：none（使わない），away（別のソースの分類器），home（自分のコーパスの中の交差検証）
MAIN = [("both", sc, k, rl) for sc in ("mod", "joint") for k in KNN_K for rl in ("none", "away")]                  # 選ぶ対象の 8 通り
ABL = ([(sp, "mod", k, rl) for sp in ("feat", "logit") for k in KNN_K for rl in ("none", "away")]                   # 比べるためだけの 9 通り（選ぶ対象にしない）
       + [("both", "mod", 100, "home")])
CFGS = MAIN + ABL
NMAIN = len(MAIN)
CI = {cfg: i for i, cfg in enumerate(CFGS)}
LISTS = [("both", s) for s in "AVTJ"] + [(sp, m) for sp in ("feat", "logit") for m in MODS]                        # 近い発話を引く表（空間, 枠）の 10 通り
LAMS = (0.0625, 0.25, 1.0)              # 近いソースの発話の答え（対照ではソースの分類器）に掛ける倍率
LAM_OF = {f"l{j}": v for j, v in enumerate(LAMS)}
ALPHA = 0.5                             # 票の平滑化
N0_LEVEL = 20.0                         # 水準を等しい重みへ戻す強さ（ラベル何本ぶん）
RANKS = ((0, 5), (5, 20), (20, 100))    # 近さの順位の区切り
ARMS = ("base", "hub", "rel", "hubrel") # B：ソースの発話の重み
TETH = list(dai1.TETHER)                # B：出発点へ引き戻す強さの格子（実験 1 と同じ 14 点）
SGRID = [(r, a) for r in dai1.RHO_GRID for a in dai1.A_GRID]        # C：疑似ラベルの重みと K 本の倍率の格子（実験 1 の提案と同じ 12 点）
TEACH = ("prod", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel", "MS_eq_tied")   # C：疑似ラベルにする版（prod は提案そのもの．MS_eq_tied は MN_eq_sel と同じ倍率の対照）
D_MIN_PT = 2.0                          # 記述の規則 D1〜D3 の下限（点）
HUB_C = 1.0                             # B：引かれた回数に足す数
BASE_TOL = 1e-3                         # B：重みが全部 1 の学び直しと保存のソースの分類器の，確率の差の許し
NONE_AGREE = 0.995                      # B・C：作り直した分類器と実験 1 の保存の，テストの判定の一致の下限
FLOOR = 1.0
N_UNITS = 120
N_COMBOS = 60
SEED = 20261114
SCRIPTS = ("dai17_probe.py", "dai16_probe.py", "dai15.py", "dai12_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
FORMS = ("eq", "F2K", "F6K", "F6orc")
V_N = tuple(f"N_eq_c{i}" for i in range(len(CFGS)))
V_MN = (tuple(f"MN_eq_c{i}_l{j}" for i in range(len(CFGS)) for j in range(len(LAMS)))
        + tuple(f"MN_{f}_c{i}_l{j}" for f in FORMS[1:] for i in range(NMAIN) for j in range(len(LAMS))))
V_CTRL = ("S_eq",) + tuple(f"MS_eq_l{j}" for j in range(len(LAMS)))
BASE = ("prod",) + V_N + V_MN + V_CTRL


def _mn(form: str, idx) -> list:
    return [f"MN_{form}_c{i}_l{j}" for i in idx for j in range(len(LAMS))]


_of = lambda sp, sc: [i for i, c in enumerate(CFGS) if c[0] == sp and c[1] == sc and c[3] != "home"]  # noqa: E731
SELDEF = {"N_eq_sel": [f"N_eq_c{i}" for i in range(NMAIN)], "MS_eq_sel": [f"MS_eq_l{j}" for j in range(len(LAMS))],
          **{f"MN_{f}_sel": _mn(f, range(NMAIN)) for f in ("eq", "F2K", "F6K")},
          "MN_eq_selmod": _mn("eq", _of("both", "mod")), "MN_eq_seljoint": _mn("eq", _of("both", "joint")),      # 引く範囲ごと
          "MN_eq_selfeat": _mn("eq", _of("feat", "mod")), "MN_eq_sellogit": _mn("eq", _of("logit", "mod"))}       # 近さの空間ごと（片方だけ）
TIED = {"MN_F6orc_sel": ("MN_F6K_sel", lambda v: v.replace("MN_F6K_", "MN_F6orc_")),
        "MS_eq_tied": ("MN_eq_sel", lambda v: f"MS_eq_l{v[-1]}")}          # ソースの分類器を足す版を，MN_eq_sel が選んだ倍率と同じ倍率で作る（倍率をそろえた対照）
REFS = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T")
PER_EMO = ("prod", "N_eq_sel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel", "MS_eq_tied", "S_eq", "none_AVT")   # 感情ごとの再現率を出す版（adj の層）
N_RAND = 200                            # 記述：でたらめに 20 本引いたときの別々の話者（塊）の数を出す回数
RUN = {"commit": "", "dirty": True}


# ------------------------------------------------------------------ ソースの側（コーパスごとに 1 回）

CVP: dict = {}
SRCDB: dict = {}


def cv_probs(c: str, m: str) -> np.ndarray:
    """コーパス c の中の交差検証（実験 1 の話者の分割 5 つ．罰則の強さは段 source が選んだ値）で，各発話に付けた確率 (n, 4)．
    どの発話も，自分を学習に使っていない分類器が付けた値である（MELD は同じ対話の発話も学習に使っていない）．"""
    if (c, m) not in CVP:
        cp = dai1.Corpus.get(c)
        X, y = cp.X[m], cp.y
        C = float(np.load(OUT1 / "source" / f"{c}_{m}.npz")["C"])
        P = np.full((len(y), 4), np.nan)
        for f in range(DD.N_FOLDS):
            tr = np.flatnonzero(DD.pool_mask(cp.df, f))
            te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(y[tr], np.arange(len(tr)), len(tr))[None]
            r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array([C]), Cb=math.inf)
            if float(r["grad_max"]) > P16.GRAD_TOL:
                raise SystemExit(f"cv_probs：{c} {m} 分割 {f} の勾配の最大が {float(r['grad_max']):.1e}")
            P[te] = torch.softmax(S.logits(Xs[te], r["W"][0], r["b"][0]), -1).cpu().numpy()
        if not np.isfinite(P).all():
            raise SystemExit(f"cv_probs：{c} {m} に埋まらない発話がある")
        CVP[(c, m)] = P
    return CVP[(c, m)]


def zs(E: np.ndarray, mu=None, sd=None):
    mu = E.mean(0) if mu is None else mu
    sd = np.maximum(E.std(0), 1e-6) if sd is None else sd
    return (E - mu) / sd, mu, sd


def within_label(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """信頼度を，同じラベルの発話の平均が 1 になるように割る（ラベルごとの票の総量を変えない．分類器の答えが特定の感情に寄る癖が票に入らないようにする）．"""
    out = np.ones_like(r, dtype=float)
    for i in np.unique(y):
        mk = y == i
        out[mk] = r[mk] / max(float(r[mk].mean()), 1e-12)
    return out


def tercile_within(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """同じラベルの発話の中の 3 分位（0 = 低い，1 = 中，2 = 高い）．コーパスごとに呼ぶ．"""
    out = np.zeros(len(r), dtype=np.int8)
    for i in np.unique(y):
        mk = y == i
        q1, q2 = np.quantile(r[mk], [1 / 3, 2 / 3])
        out[mk] = (r[mk] > q1).astype(np.int8) + (r[mk] > q2).astype(np.int8)
    return out


def speaker_names(corpus: str, df: pd.DataFrame) -> np.ndarray:
    """話者の名前（区間を作る塊とは別）．IEMOCAP と AFEW は発話一覧の話者の ID（IEMOCAP は 10 人，AFEW は俳優名），
    MELD は別名をまとめた役名，MOSEI はチャンネル（無ければ動画）．"""
    if corpus in ("IEMOCAP", "AFEW"):
        v = df["speaker_id"].astype(str).replace(DD.AFEW_ALIAS)
    elif corpus == "MELD":
        v = df["speaker_group"].astype(str).replace(DD.MELD_ALIAS)
    else:
        v = df["speaker_group"].astype(str)
    return np.char.add(f"{corpus}:", v.to_numpy().astype(str))


def unit_norm(E) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.as_tensor(np.asarray(E), dtype=torch.float32, device=S.DEV), dim=1)


def cat_norm(parts: list) -> torch.Tensor:
    """長さ 1 のベクトルを並べ，全体の長さを 1 に戻す．並べたベクトルどうしのコサインは，部分ごとのコサインの平均になる．"""
    return torch.cat(parts, 1) / math.sqrt(len(parts))


def source_db(srcs: list, slot: str) -> dict:
    """単位のソースの全発話を 1 つの表にする（枠 slot：A・V・T はそのモダリティ，J は 3 つのモダリティを合わせたもの）．
    位置：feat（そのコーパスの平均と標準偏差で標準化した特徴），logit（ソースの分類器の点を並べたもの．自分のコーパスの分類器の分は交差検証の値，
    別のコーパスの分類器の分はそのまま当てた値．コーパスの中で次元ごとに標準化），both（feat と logit を並べたもの）．J は 3 つのモダリティの both を並べたもの．
    r_home：交差検証が自分のラベルに付けた確率．r_away：別のソースの分類器が自分のラベルに付けた確率（平均）．J は 3 つのモダリティの平均．
    どちらも，コーパス × ラベルの中で平均が 1 になるように割る（within_label）．割る前の値は r_home_raw・r_away_raw．
    t_home・t_away：コーパス × ラベルの中の 3 分位（0 = 低い，1 = 中，2 = 高い）．spk：塊（区間を作る単位）の番号．spkid：話者（speaker_names）の番号．
    発話の並びはどの枠でも同じ（コーパスの順，その中は sample_id の順）．"""
    key = (tuple(srcs), slot)
    if key in SRCDB:
        return SRCDB[key]
    if slot == "J":
        dbs = [source_db(srcs, m) for m in MODS]
        d0 = dbs[0]
        assert all((d["sid"] == d0["sid"]).all() for d in dbs)
        off = np.concatenate([[0], np.cumsum(d0["sizes"])])
        raw = {rl: np.mean([d[f"r_{rl}_raw"] for d in dbs], 0) for rl in ("home", "away")}
        nrm = {rl: np.concatenate([within_label(raw[rl][a:b], d0["y"][a:b]) for a, b in zip(off[:-1], off[1:])]) for rl in raw}
        ter = {rl: np.concatenate([tercile_within(raw[rl][a:b], d0["y"][a:b]) for a, b in zip(off[:-1], off[1:])]) for rl in raw}
        SRCDB[key] = dict(d0, feat=None, logit=None, r_home=nrm["home"], r_away=nrm["away"],
                          r_home_raw=raw["home"], r_away_raw=raw["away"], t_home=ter["home"], t_away=ter["away"])
        return SRCDB[key]
    m = slot
    feat, logit, ys, corp, bal, rh, ra, spk, sid, rh_raw, ra_raw, th, ta, spid = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for ci, s in enumerate(srcs):
        cp = dai1.Corpus.get(s)
        X, y = cp.X[m], cp.y
        feat.append(torch.nn.functional.normalize(((X - X.mean(0)) / X.std(0).clamp(min=1e-6)).float(), dim=1))
        E = np.concatenate([P16.centered(cv_probs(s, m) if u == s else P16.src_probs_on(u, s, m)) for u in srcs], 1)
        logit.append(unit_norm(zs(E)[0]))
        cnt = np.bincount(y, minlength=4).astype(float)
        bal.append(len(y) / (4.0 * np.maximum(cnt[y], 1.0)))
        ii = np.arange(len(y))
        rh_raw.append(cv_probs(s, m)[ii, y])
        ra_raw.append(np.mean([P16.src_probs_on(u, s, m)[ii, y] for u in srcs if u != s], 0))
        rh.append(within_label(rh_raw[-1], y)); ra.append(within_label(ra_raw[-1], y))
        th.append(tercile_within(rh_raw[-1], y)); ta.append(tercile_within(ra_raw[-1], y))
        spid.append(speaker_names(s, cp.df))
        ys.append(y); corp.append(np.full(len(y), ci))
        spk.append(np.char.add(f"{s}:", cp.df["cluster"].to_numpy().astype(str))); sid.append(cp.df["sample_id"].to_numpy().astype(str))
    F, L = torch.cat(feat), torch.cat(logit)
    SRCDB[key] = dict(feat=F, logit=L, y=np.concatenate(ys), corp=np.concatenate(corp), bal=np.concatenate(bal),
                      r_none=np.ones(sum(map(len, ys))), r_home=np.concatenate(rh), r_away=np.concatenate(ra), sizes=[len(a) for a in ys],
                      spk=np.unique(np.concatenate(spk), return_inverse=True)[1], spk_name=np.concatenate(spk), sid=np.concatenate(sid),
                      r_home_raw=np.concatenate(rh_raw), r_away_raw=np.concatenate(ra_raw), t_home=np.concatenate(th), t_away=np.concatenate(ta),
                      spkid=np.unique(np.concatenate(spid), return_inverse=True)[1], spkid_name=np.concatenate(spid))
    return SRCDB[key]


def save_srcdb(out: Path, srcs: list) -> None:
    """ソースの発話の表（近い発話の番号が指す並び）を 1 回だけ保存する．計算の後に，近い発話の番号から別の集計を出すためのもの．"""
    d = out / "srcdb"
    d.mkdir(parents=True, exist_ok=True)
    for slot in "AVTJ":
        p = d / f"{'+'.join(srcs)}_{slot}.npz"
        if not p.exists():
            db = source_db(srcs, slot)
            np.savez_compressed(p, y=db["y"].astype(np.int8), corp=db["corp"].astype(np.int8), bal=db["bal"], r_home=db["r_home"], r_away=db["r_away"],
                                r_home_raw=db["r_home_raw"], r_away_raw=db["r_away_raw"], t_home=db["t_home"], t_away=db["t_away"], speaker_name=db["spkid_name"],
                                cluster_name=db["spk_name"], sid=db["sid"], srcs=np.array(srcs), sizes=np.array(db["sizes"]))


def topk_idx(Q: torch.Tensor, DB: torch.Tensor, k: int = KMAX, chunk: int = 4096, with_sim: bool = False):
    """問い合わせ Q の各行に，DB の中でコサインの近い順に k 本の番号を返す（近い順）．with_sim なら，コサインの値も返す．"""
    out = [torch.topk(Q[i:i + chunk] @ DB.T, min(k, DB.shape[0]), dim=1) for i in range(0, Q.shape[0], chunk)]
    idx = np.concatenate([o.indices.cpu().numpy() for o in out])
    return (idx, np.concatenate([o.values.cpu().numpy() for o in out])) if with_sim else idx


def topk_z(parts: list, k: int = KMAX, chunk: int = 2048):
    """parts：[(問い合わせ Q (n, d), 表 D (N, d))]（どちらも長さ 1）．部分ごとに，問い合わせ 1 本ごとに表の全発話とのコサインを平均 0・標準偏差 1 にそろえ，
    部分の数で平均した値（標準化したコサインの平均）の大きい順に k 本を返す．散らばりの違う近さを同じ重さで合わせるため．
    戻り：(番号 (n, k)，合わせた値 (n, k)，選んだ発話の部分ごとのコサイン (n, k, 部分の数))．"""
    n = parts[0][0].shape[0]
    idx, val, cos = [], [], []
    for i in range(0, n, chunk):
        tot = None
        raws = []
        for Q, D in parts:
            C = Q[i:i + chunk] @ D.T
            raws.append(C)
            Z = (C - C.mean(1, keepdim=True)) / C.std(1, keepdim=True).clamp(min=1e-6)
            tot = Z if tot is None else tot + Z
        tv = torch.topk(tot / len(parts), min(k, tot.shape[1]), dim=1)
        idx.append(tv.indices.cpu().numpy()); val.append(tv.values.cpu().numpy())
        cos.append(torch.stack([torch.gather(C, 1, tv.indices) for C in raws], -1).cpu().numpy())
        del tot, raws
    return np.concatenate(idx), np.concatenate(val), np.concatenate(cos)


def both_parts(emb: dict, srcs: list, slot: str, w: int) -> list:
    """両方の近さの部分：枠が A・V・T ならそのモダリティの特徴と点の 2 つ，J なら 3 つのモダリティの 6 つ．w = 0 は候補，1 はテスト．"""
    mods = MODS if slot == "J" else slot
    return [(emb[(sp, m)][w], source_db(srcs, m)[sp]) for m in mods for sp in ("feat", "logit")]


def votes(idx: np.ndarray, db: dict, k: int, rel: str) -> np.ndarray:
    """近い k 本のラベルの重みつきの票 → 確率 (n, 4)．重み：感情の釣り合い（コーパスの中の感情の件数の逆数）× 信頼度．"""
    ii = idx[:, :k]
    Y, W = db["y"][ii], db["bal"][ii] * db[f"r_{rel}"][ii]
    V = np.stack([(W * (Y == i)).sum(1) for i in range(4)], 1)
    return (V + ALPHA) / (V.sum(1, keepdims=True) + 4 * ALPHA)


def target_embed(u, src_p: dict, src_t: dict) -> dict:
    """ターゲットの候補とテストの位置（標準化は候補の統計．ラベルを使わない）．戻り：{(空間, 枠): (候補, テスト)}．"""
    out = {}
    for m in MODS:
        Ep = np.concatenate([P16.centered(src_p[(s, m)]) for s in u.srcs], 1)
        Et = np.concatenate([P16.centered(src_t[(s, m)]) for s in u.srcs], 1)
        Ep, mu, sd = zs(Ep)
        out[("feat", m)] = (torch.nn.functional.normalize(u.Xp[m].float(), dim=1), torch.nn.functional.normalize(u.Xt[m].float(), dim=1))
        out[("logit", m)] = (unit_norm(Ep), unit_norm(zs(Et, mu, sd)[0]))
    return out


def nn_rows(rows: list, sp: str, slot: str, db: dict, it: np.ndarray, yt: np.ndarray, srcs: list) -> None:
    """記述：近い発話のラベルが本当の感情と同じ割合（テスト．近さの順位・ソース・信頼度の 3 分位・感情ごと）．"""
    base = dict(K=-1, version="_nn", space=sp, modality=slot)
    Y = db["y"][it]
    hit = Y == yt[:, None]
    pt = np.bincount(yt, minlength=4) / len(yt)
    for a, b in RANKS + ((0, KMAX),):
        rows.append(dict(base, by="rank", level=f"{a + 1}-{b}", hit=float(hit[:, a:b].mean()), share=1.0))
        pn = np.bincount(Y[:, a:b].ravel(), minlength=4) / Y[:, a:b].size    # その順位の近い発話のラベルの割合
        rows.append(dict(base, by="rank_marginal", level=f"{a + 1}-{b}", hit=float(pt @ pn), share=1.0))   # ラベルの割合だけで決まる当たる割合
        rows.append(dict(base, by="rank_bal", level=f"{a + 1}-{b}",
                         hit=float(np.mean([hit[yt == e][:, a:b].mean() for e in range(4) if (yt == e).any()])), share=1.0))   # 感情ごとの当たる割合の平均（UAR と同じ数え方）
    C = db["corp"][it]
    for ci_, s in enumerate(srcs):
        mk = C == ci_
        rows.append(dict(base, by="source", level=s, hit=(float(hit[mk].mean()) if mk.any() else np.nan), share=float(mk.mean()), db_share=float(db["sizes"][ci_] / sum(db["sizes"]))))
    for rl in ("home", "away"):                                          # 3 分位は，ソースのコーパス × ラベルの中で切ったもの
        Tc = db[f"t_{rl}"][it]
        for ti, nm in enumerate(("low", "mid", "high")):
            mk = Tc == ti
            rows.append(dict(base, by=f"rel_{rl}", level=nm, hit=(float(hit[mk].mean()) if mk.any() else np.nan), share=float(mk.mean())))
        for e in range(4):                                               # ラベルの構成をそろえた版：近い発話のラベル e ごとに分ける
            for ti, nm in enumerate(("low", "mid", "high")):
                mk = (Tc == ti) & (Y == e)
                rows.append(dict(base, by=f"rel_{rl}_label{e}", level=nm, hit=(float(hit[mk].mean()) if mk.any() else np.nan), share=float(mk.mean())))
    pdb = np.bincount(db["y"], minlength=4) / len(db["y"])
    rows.append(dict(base, by="chance", level="all", hit=float(pt @ pdb), share=1.0))                 # 表からでたらめにソースの発話を引いたときの当たる割合
    for ci_, s in enumerate(srcs):
        ps = np.bincount(db["y"][db["corp"] == ci_], minlength=4) / max((db["corp"] == ci_).sum(), 1)
        rows.append(dict(base, by="chance_source", level=s, hit=float(pt @ ps), share=float(db["sizes"][ci_] / sum(db["sizes"]))))
    rg = np.random.default_rng([SEED, zlib.crc32(f"{'+'.join(srcs)}|{slot}".encode())])
    draws = [rg.choice(len(db["y"]), 20, replace=False) for _ in range(N_RAND)]
    for nm, col in (("distinct_speakers_top20", "spkid"), ("distinct_clusters_top20", "spk")):   # hit の列：上位 20 本の中の別々の話者（塊）の数の平均．share の列：表の話者（塊）の数
        rows.append(dict(base, by=nm, level="all", hit=float(np.mean([len(np.unique(r_)) for r_ in db[col][it[:, :20]]])), share=float(len(np.unique(db[col])))))
        rows.append(dict(base, by=nm, level="random", hit=float(np.mean([len(np.unique(db[col][d_])) for d_ in draws])), share=float(len(np.unique(db[col])))))
    for e in range(4):
        mk = yt == e
        rows.append(dict(base, by="true_emotion", level=str(e), hit=(float(hit[mk].mean()) if mk.any() else np.nan), share=float(mk.mean())))
        rows.append(dict(base, by="neighbor_label", level=str(e), hit=(float(hit[Y == e].mean()) if (Y == e).any() else np.nan), share=float((Y == e).mean())))


def mn_weights(form: str, lam: float, DK_M: np.ndarray, DK_N: np.ndarray, DO_M: np.ndarray, DO_N: np.ndarray, K: int) -> np.ndarray:
    """微調整した 3 つの分類器と，近いソースの発話の答え（nE 個．mod なら 3，joint なら 1）の重み (3 + nE, 4)．
    eq：全部 1．F2K：分類器ごとの水準（K 本で測った信頼度の 4 感情の平均．N0_LEVEL 本ぶんの強さで全体の平均へ戻す）．F6K：分類器 × 感情ごとの信頼度（K 本）．
    F6orc：分類器 × 感情ごとの信頼度（候補の K 本以外の全ラベル．参照）．微調整した 3 つの平均が 1 になるように割り（平均が 0 以下なら全部 1），
    近いソースの発話の答えの行に lam × (3 ÷ nE) を掛ける（joint の 1 つの答えは，mod の 3 つの答えと同じ総量にする）．"""
    nE = DK_N.shape[0]
    DKx, DOx = np.concatenate([DK_M, DK_N]), np.concatenate([DO_M, DO_N])
    Wk = {"eq": np.ones((3 + nE, 4)),
          "F2K": np.repeat(P16.mix_rel(DKx, DKx, 4 * K, N0_LEVEL, 0.0, "eq").mean(1, keepdims=True), 4, 1),
          "F6K": DKx, "F6orc": DOx}[form]
    ref = float(Wk[:3].mean())
    W = np.ones((3 + nE, 4)) if ref <= 0 else Wk / ref
    W = W.copy()
    W[3:] *= lam * 3.0 / nE
    return W


def src_avg(L: np.ndarray, nsrc: int) -> np.ndarray:
    """対照のソースの分類器の点：モダリティごとにソースの点を平均して 3 つにし，足す．L の並びは (ソース, モダリティ)．"""
    return L.reshape(nsrc, 3, *L.shape[1:]).mean(0).sum(0)


# ------------------------------------------------------------------ A：1 単位

def run_unit(sv) -> tuple[list, dict]:
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    nsrc = len(u.srcs)
    arr["y_test"], arr["cl_test"], arr["sid_test"] = yt.astype(np.int8), sv.cl, sv.sid
    arr["speaker_test"] = speaker_names(u.c, u.cp.df)[u.test]
    arr["y_pool"], arr["cl_pool"], arr["sid_pool"] = yp.astype(np.int8), sv.cl_pool, sv.sid_pool
    arr["srcs"] = np.array(u.srcs)
    arr["cfgs"] = np.array(["|".join(map(str, c)) for c in CFGS])
    src_p = {(s, m): dai1.probs(u.Xp[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    src_t = {(s, m): dai1.probs(u.Xt[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    # ---- 近いソースの発話を引く（K に依らない）
    emb = target_embed(u, src_p, src_t)
    nn = {}
    for sp, slot in LISTS:
        db = source_db(u.srcs, slot)
        if sp == "both":
            ip, sp_, _ = topk_z(both_parts(emb, u.srcs, slot, 0))
            it, st, ct = topk_z(both_parts(emb, u.srcs, slot, 1))
            arr[f"nncos_test_{sp}_{slot}"] = ct.astype(np.float16)          # 選んだ発話の部分ごとのコサイン（並びは (モダリティ, 特徴・点)）
        else:
            (ip, sp_), (it, st) = topk_idx(emb[(sp, slot)][0], db[sp], with_sim=True), topk_idx(emb[(sp, slot)][1], db[sp], with_sim=True)
        nn[(sp, slot)] = (db, ip, it)
        if sv.r == 0 and sp == "both":                                      # 候補の側は大きいので，主の空間の分を，種 0 の単位にだけ保存する（種に依らない）
            arr[f"nn_pool_{sp}_{slot}"], arr[f"nnsim_pool_{sp}_{slot}"] = ip.astype(np.int32), sp_.astype(np.float16)
        arr[f"nn_test_{sp}_{slot}"] = it.astype(np.int32)                   # テストの発話ごとの，近いソースの発話の番号（表の並び．近い順）
        arr[f"nnsim_test_{sp}_{slot}"] = st.astype(np.float16)              # そのコサインの値
        nn_rows(rows, sp, slot, db, it, yt, u.srcs)
    del emb
    knn_p, knn_t = {}, {}                                                # [格子の番号] = 確率 (答えの数, n, 4)
    for ci, (sp, sc, k, rl) in enumerate(CFGS):
        slots = MODS if sc == "mod" else "J"
        knn_p[ci] = np.stack([votes(nn[(sp, s_)][1], nn[(sp, s_)][0], k, rl) for s_ in slots])
        knn_t[ci] = np.stack([votes(nn[(sp, s_)][2], nn[(sp, s_)][0], k, rl) for s_ in slots])
        arr[f"p_test_knn_c{ci}"], arr[f"p_pool_knn_c{ci}"] = knn_t[ci].astype(np.float32), knn_p[ci].astype(np.float32)
    for (s, m), P in src_t.items():
        arr[f"p_test_src_{s}_{m}"] = P.astype(np.float32)
        arr[f"p_pool_src_{s}_{m}"] = src_p[(s, m)].astype(np.float32)
    for K in KS:
        if sv.fz["main"][str(K)] != "one":
            raise SystemExit(f"{sv.base.name}：実験 1 の主の版が one でない（K={K}）")
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        ynk = yp[nk]
        pool = P16.View({m: TK[K][f"ret_{m}"][0].cpu().numpy().astype(np.float64) for m in MODS}, src_p, u.srcs)
        test = P16.View({m: TK[K][f"ret_{m}"][1].cpu().numpy().astype(np.float64) for m in MODS}, src_t, u.srcs)
        for m in MODS:
            arr[f"p_pool_ret_{m}_K{K}"], arr[f"p_test_ret_{m}_K{K}"] = pool.ret[m].astype(np.float32), test.ret[m].astype(np.float32)
        # ---- 微調整した分類器の K 本の上の確率（1 組を抜いて作り直したもの．実験 16 と同じ）
        fold = dai15.kfold_of(yk)
        sets = [kpos] + [kpos[fold != j] for j in range(NF)]
        Pm, conv = P16.teachers_mod(u, sv.teth, K, sets)
        dev = float(np.abs(P16.softmax(sum(P16.lg(Pm[m][0]) for m in MODS)) - pool.q1).max())
        if dev > P16.CHECK_TEACHER:
            raise SystemExit(f"{sv.base.name}：K 本を全部使って作り直した積が保存の積と {dev:.2e} 違う（K={K}）")
        held_ret = {m: np.stack([Pm[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in MODS}
        LMh = np.stack([P16.centered(held_ret[m]) for m in MODS])
        LSh = np.stack([P16.centered(src_p[(s, m)][kpos]) for s in u.srcs for m in MODS])
        DK_M = np.stack([P16.dprime(held_ret[m], yk)[0] for m in MODS])
        DO_M = np.stack([P16.dprime(pool.ret[m][nk], ynk)[0] for m in MODS])
        arr[f"D_M_K_K{K}"], arr[f"D_M_orc_K{K}"] = DK_M, DO_M
        target = pool.q1.mean(0)
        specs = {"prod": (pool.LM.sum(0), test.LM.sum(0), LMh.sum(0)),
                 "S_eq": (src_avg(pool.LS, nsrc), src_avg(test.LS, nsrc), src_avg(LSh, nsrc))}
        for nm, lam in LAM_OF.items():
            specs[f"MS_eq_{nm}"] = (pool.LM.sum(0) + lam * src_avg(pool.LS, nsrc), test.LM.sum(0) + lam * src_avg(test.LS, nsrc), LMh.sum(0) + lam * src_avg(LSh, nsrc))
        for ci in range(len(CFGS)):
            LNp = np.stack([P16.centered(p_) for p_ in knn_p[ci]])
            LNt = np.stack([P16.centered(p_) for p_ in knn_t[ci]])
            LNh = LNp[:, kpos]
            nE = LNp.shape[0]
            DK_N = np.stack([P16.dprime(p_[kpos], yk)[0] for p_ in knn_p[ci]])           # 近いソースの発話の答えは K 本を学習に使わないので，そのまま測る
            DO_N = np.stack([P16.dprime(p_[nk], ynk)[0] for p_ in knn_p[ci]])
            arr[f"D_N_K_K{K}_c{ci}"], arr[f"D_N_orc_K{K}_c{ci}"] = DK_N, DO_N
            specs[f"N_eq_c{ci}"] = tuple(x.sum(0) * 3.0 / nE for x in (LNp, LNt, LNh))
            Lp, Lt, Lh = np.concatenate([pool.LM, LNp]), np.concatenate([test.LM, LNt]), np.concatenate([LMh, LNh])
            for f in (FORMS if ci < NMAIN else FORMS[:1]):
                for j, lam in enumerate(LAMS):
                    W = mn_weights(f, lam, DK_M, DK_N, DO_M, DO_N, K)
                    specs[f"MN_{f}_c{ci}_l{j}"] = tuple(np.einsum("ky,kny->ny", W, L) for L in (Lp, Lt, Lh))
        assert set(specs) == set(BASE), set(specs) ^ set(BASE)
        resid_max = 0.0
        for v in BASE:
            Sp, St, Sh = (P16.norm_log(x) for x in specs[v])
            if not (np.isfinite(Sp).all() and np.isfinite(St).all()):
                raise SystemExit(f"{sv.base.name}：まとめた点が有限でない（{v}）")
            dl, rs = P16.match_shift(Sp, target)
            resid_max = max(resid_max, rs)
            bk = P16.fit_wb((Sh + dl)[None], yk, np.ones(1), P16.TAU, fit_w=False, fit_b=True)[1]
            bo = P16.fit_wb(Sp[nk][None], ynk, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
            pr, pa, pk, po, pp = St.argmax(1), (St + dl).argmax(1), (St + dl + bk).argmax(1), (St + bo).argmax(1), (Sp + dl)[nk].argmax(1)
            for nm, pred in (("raw", pr), ("adj", pa), ("kb", pk), ("ob", po), ("pool", pp)):
                arr[f"pred_{nm}_K{K}_{v}"] = pred.astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pa), uar_kb=100 * S.uar(yt, pk), uar_ob=100 * S.uar(yt, po),
                             uar_pool=100 * S.uar(ynk, pp), adj_resid=rs))
        Wj, bj = sv.prop("AVT", K)
        refs_t = {"none_AVT": TK[K]["none_AVT"][1].argmax(1).cpu().numpy(), "joint_AVT": S.logits(u.Xt["AVT"], Wj, bj).argmax(1).cpu().numpy()}
        refs_p = {"none_AVT": TK[K]["none_AVT"][0].argmax(1).cpu().numpy()[nk], "joint_AVT": S.logits(u.Xp["AVT"], Wj, bj).argmax(1).cpu().numpy()[nk]}
        for m in MODS:
            refs_t[f"ret_{m}"], refs_p[f"ret_{m}"] = test.ret[m].argmax(1), pool.ret[m][nk].argmax(1)
        for v in REFS:
            arr[f"pred_raw_K{K}_{v}"], arr[f"pred_pool_K{K}_{v}"] = refs_t[v].astype(np.int8), refs_p[v].astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, refs_t[v]), uar_pool=100 * S.uar(ynk, refs_p[v])))
        # 記述：答え 1 つずつの UAR（テスト）と，候補の上の点（感情の平均を引いた対数）の標準偏差（確信の強さの目安）
        sdp = lambda P: float(P16.centered(P).std())  # noqa: E731
        srcpt = {m: np.mean([P16.centered(src_t[(s, m)]) for s in u.srcs], 0) for m in MODS}        # ソースの分類器の点（モダリティごとの平均）
        srcpt["J"] = sum(srcpt[m] for m in MODS)
        for ci, (sp, sc, k, rl) in enumerate(CFGS):
            for e_, slot in enumerate(MODS if sc == "mod" else "J"):
                Lk = P16.centered(knn_t[ci][e_])
                rows.append(dict(K=K, version="_single", modality=slot, name=f"knn_c{ci}", uar_raw=100 * S.uar(yt, knn_t[ci][e_].argmax(1)), sd_pool=sdp(knn_p[ci][e_]),
                                 agree_src=float((Lk.argmax(1) == srcpt[slot].argmax(1)).mean()), corr_src=P16.corr(Lk, srcpt[slot])))   # 近いソースの発話の答えとソースの分類器の重なり
        for m in MODS:
            for s in u.srcs:
                rows.append(dict(K=K, version="_single", modality=m, name=f"src_{s}", uar_raw=100 * S.uar(yt, src_t[(s, m)].argmax(1)), sd_pool=sdp(src_p[(s, m)])))
            rows.append(dict(K=K, version="_single", modality=m, name="src_avg", uar_raw=100 * S.uar(yt, np.mean([P16.centered(src_t[(s, m)]) for s in u.srcs], 0).argmax(1)),
                             sd_pool=float(np.mean([P16.centered(src_p[(s, m)]) for s in u.srcs], 0).std())))
            rows.append(dict(K=K, version="_single", modality=m, name="ret", uar_raw=100 * S.uar(yt, test.ret[m].argmax(1)), sd_pool=sdp(pool.ret[m])))
            arr[f"pcf_{m}_K{K}"] = held_ret[m].astype(np.float32)
        arr[f"kpos_K{K}"] = kpos.astype(np.int32)
        arr[f"kfold_K{K}"] = fold.astype(np.int8)
        rows.append(dict(K=K, version="_check", teacher_dev=dev, grad=conv[0], stalled=conv[1], adj_resid=resid_max,
                         gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
    return rows, arr


# ------------------------------------------------------------------ B：ソースの発話に重みを付けてソースの分類器を学び直す（T × c × 分割）

def weighted_onehot(y: np.ndarray, w: np.ndarray) -> torch.Tensor:
    """感情の釣り合いの重みに発話ごとの重み w を掛けた one-hot (n, 4)．w が全部 1 なら dai_solver.balanced_onehot と同じ．感情ごとの重みの和は n/4 のまま．"""
    n = len(y)
    G = np.zeros((n, 4))
    present = len(np.unique(y))
    for i in np.unique(y):
        mk = y == i
        G[mk, i] = w[mk] / w[mk].sum() * (n / present)
    return S.t(G)


RWFIT: dict = {}


def fit_source_weighted(s: str, m: str, wlist: list) -> tuple[torch.Tensor, torch.Tensor]:
    """ソース s のモダリティ m の分類器を，発話ごとの重み（wlist の各要素）で学び直す．罰則の強さと標準化は実験 1 の段 source と同じ．戻り：W (重みの数, 4, d)，b (重みの数, 4)．"""
    cp = dai1.Corpus.get(s)
    X, y = cp.X[m], cp.y
    Xs = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
    C = float(np.load(OUT1 / "source" / f"{s}_{m}.npz")["C"])
    G = torch.stack([weighted_onehot(y, w) for w in wlist])
    r = S.fit(Xs, G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.full(len(wlist), C), Cb=math.inf)
    if float(r["grad_max"]) > P16.GRAD_TOL:
        raise SystemExit(f"reweight：{s} {m} の勾配の最大が {float(r['grad_max']):.1e}")
    return r["W"], r["b"]


def base_rel_fit(srcs: list, ci: int, m: str):
    """重みが全部 1（base）と，信頼度の重み（rel）の学び直し．ターゲットに依らないので，ソースの組ごとに 1 回だけ解く．
    base が保存のソースの分類器と同じ確率を出すかを確かめる（そのコーパスの全発話の上の確率の差の最大）．"""
    key = (tuple(srcs), ci, m)
    if key not in RWFIT:
        s = srcs[ci]
        db = source_db(srcs, m)
        off = np.concatenate([[0], np.cumsum(db["sizes"])])
        ra = db["r_away"][off[ci]:off[ci + 1]]
        W, b = fit_source_weighted(s, m, [np.ones(len(ra)), ra / ra.mean()])
        cp = dai1.Corpus.get(s)
        Xs = (cp.X[m] - cp.X[m].mean(0)) / cp.X[m].std(0).clamp(min=1e-6)
        dev = float((dai1.probs(Xs, W[0], b[0]) - dai1.probs(Xs, *dai1.load_source(OUT1, s, m))).abs().max())
        if dev > BASE_TOL:
            raise SystemExit(f"reweight：重みが全部 1 の学び直しが，保存のソースの分類器と確率で {dev:.1e} 違う（{s} {m}）")
        RWFIT[key] = (W, b, dev)
    return RWFIT[key]


def run_reweight(T: str, c: str, f: int, fzT: dict, seeds: list) -> tuple[list, dict]:
    svs = {r: dai3.Saved(OUT1, "pseudo", T, c, f, r, fzT) for r in seeds}
    u = svs[seeds[0]].u
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    arr["y_test"], arr["cl_test"], arr["sid_test"] = yt.astype(np.int8), svs[seeds[0]].cl, svs[seeds[0]].sid
    arr["srcs"] = np.array(u.srcs)
    probs_k = {}
    src_p = {(s, m): dai1.probs(u.Xp[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    src_t = {(s, m): dai1.probs(u.Xt[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    emb = target_embed(u, src_p, src_t)                                  # 引かれた回数には候補の位置だけ使う（テストは使わない）
    for m in MODS:
        db = source_db(u.srcs, m)
        ip = topk_z(both_parts(emb, u.srcs, m, 0))[0]                       # 候補（ラベル無し．テストを含まない）から，主の近さ（both）で引く
        hub = np.bincount(ip.ravel(), minlength=len(db["y"])).astype(float)                       # ソースの発話が引かれた回数
        arr[f"hub_{m}"] = hub.astype(np.int32)
        off = np.concatenate([[0], np.cumsum(db["sizes"])])
        Ws, bs = {a: [] for a in ARMS}, {a: [] for a in ARMS}
        for ci, s in enumerate(u.srcs):
            sl = slice(off[ci], off[ci + 1])
            ra = db["r_away"][sl]
            h = (hub[sl] + HUB_C) / (hub[sl].mean() + HUB_C)                 # 引かれた回数に HUB_C を足してから平均で割る（候補が少ないと 1 に近づく）
            Wbr, bbr, dev = base_rel_fit(u.srcs, ci, m)
            Wh, bh = fit_source_weighted(s, m, [h, h * ra / (h * ra).mean()])
            for a, W_, b_ in (("base", Wbr[0], bbr[0]), ("rel", Wbr[1], bbr[1]), ("hub", Wh[0], bh[0]), ("hubrel", Wh[1], bh[1])):
                Ws[a].append(W_); bs[a].append(b_)
            ess = lambda w: float(w.sum() ** 2 / (w ** 2).sum() / len(w))  # noqa: E731  重みの実効の本数 ÷ 本数
            rows.append(dict(K=-1, modality=m, arm="_weights", source=s, hub_mean=float(hub[sl].mean()), hub_zero=float((hub[sl] == 0).mean()),
                             hub_top10_share=float(np.sort(hub[sl])[::-1][:max(1, (sl.stop - sl.start) // 10)].sum() / max(hub[sl].sum(), 1.0)), base_dev=dev,
                             ess_hub=ess(h), ess_rel=ess(ra), ess_hubrel=ess(h * ra), corr_hub_rel=float(np.corrcoef(hub[sl], ra)[0, 1]),
                             **{f"hub_share_label{i}": float(hub[sl][db["y"][sl] == i].sum() / max(hub[sl].sum(), 1.0)) for i in range(4)}))
        start = {a: (torch.stack(Ws[a]).mean(0), torch.stack(bs[a]).mean(0)) for a in ARMS}
        dev0 = float((dai1.probs(u.Xt[m], *start["base"]) - dai1.probs(u.Xt[m], *u.start[m])).abs().max())   # base の出発点と実験 1 の出発点の，テストの確率の差の最大
        if dev0 > BASE_TOL:
            raise SystemExit(f"reweight：base の出発点が実験 1 の出発点と確率で {dev0:.1e} 違う（{m}）")
        for a in ARMS:
            Z0 = S.logits(u.Xt[m], *start[a])
            pr = Z0.argmax(1).cpu().numpy()
            arr[f"pred_K0_{m}_{a}"] = pr.astype(np.int8)
            arr[f"prob_K0_{m}_{a}"] = torch.softmax(Z0, -1).cpu().numpy().astype(np.float32)
            arr[f"start_W_{m}_{a}"], arr[f"start_b_{m}_{a}"] = start[a][0].cpu().numpy().astype(np.float32), start[a][1].cpu().numpy().astype(np.float32)
            rows.append(dict(K="0", modality=m, arm=a, seed=-1, grid=-1, frozen=True, uar=100 * S.uar(yt, pr)))
            # 参照：候補の全ラベルで決めた切片を足す（切片を最良にそろえて，向きの良さだけを比べる．実際には使えない）
            Lp0 = P16.norm_log(S.logits(u.Xp[m], *start[a]).cpu().numpy().astype(np.float64))
            bo = P16.fit_wb(Lp0[None], yp, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
            po = (P16.norm_log(Z0.cpu().numpy().astype(np.float64)) + bo).argmax(1)
            arr[f"pred_K0ob_{m}_{a}"] = po.astype(np.int8)
            rows.append(dict(K="0ob", modality=m, arm=a, seed=-1, grid=-1, frozen=True, uar=100 * S.uar(yt, po)))
        for r_ in seeds:
            ur = svs[r_].u
            for K in KS:
                cw0, cb0 = svs[r_].teth[(m, K)]
                g0 = [gi for gi, (cw, cb) in enumerate(TETH) if abs(cw - cw0) < 1e-12 and abs(cb - cb0) < 1e-12]
                if len(g0) != 1:
                    raise SystemExit(f"reweight：実験 1 の固定の引き戻しの強さ {cw0}, {cb0} が格子に無い")
                nG = len(TETH)                                               # 問の並び：重み × 格子
                G = ur.gK(K)[None].expand(len(ARMS) * nG, -1, -1)
                res = S.fit(ur.Xp[m], G, torch.stack([start[a][0] for a in ARMS for _ in TETH]), torch.stack([start[a][1] for a in ARMS for _ in TETH]),
                            Cw=np.array([cw for _ in ARMS for cw, cb in TETH]), Cb=np.array([cb for _ in ARMS for cw, cb in TETH]))
                if float(res["grad_max"]) > P16.GRAD_TOL:
                    raise SystemExit(f"reweight：K 本の学び直しの勾配の最大が {float(res['grad_max']):.1e}")
                Z = S.logits(ur.Xt[m], res["W"], res["b"])
                Wn, bn = svs[r_].none(m, K)                                  # 実験 1 の保存の「K 本だけ」の分類器．base の固定の強さの解と同じ答えになるはず
                agree = float((Z[ARMS.index("base") * nG + g0[0]].argmax(1) == S.logits(ur.Xt[m], Wn, bn).argmax(1)).float().mean())
                rows.append(dict(K=str(K), modality=m, arm="_check", seed=r_, none_agree=agree, start_dev=dev0, grad=float(res["grad_max"]), stalled=int(res["stalled"]),
                                 gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
                if agree < NONE_AGREE:
                    raise SystemExit(f"reweight：base の K 本だけの分類器が，実験 1 の保存と {100 * (1 - agree):.2f}% の発話で違う（{m} K={K}）")
                for ai, a in enumerate(ARMS):
                    for gi in range(nG):
                        pr = Z[ai * nG + gi].argmax(1).cpu().numpy()
                        arr[f"pred_K{K}_{m}_{a}_g{gi}_r{r_}"] = pr.astype(np.int8)
                        rows.append(dict(K=str(K), modality=m, arm=a, seed=r_, grid=gi, frozen=(gi == g0[0]), uar=100 * S.uar(yt, pr)))
                    arr[f"pred_K{K}_{m}_{a}_r{r_}"] = arr[f"pred_K{K}_{m}_{a}_g{g0[0]}_r{r_}"]          # 実験 1 の固定の強さの解
                    probs_k[(K, m, a, r_)] = torch.softmax(Z[ai * nG + g0[0]], -1).cpu().numpy().astype(np.float64)
                    arr[f"prob_K{K}_{m}_{a}_r{r_}"] = probs_k[(K, m, a, r_)].astype(np.float16)
    for r_ in seeds:                                                     # 3 つのモダリティの K 本だけの分類器の積
        for K in KS:
            for a in ARMS:
                pr = sum(np.log(np.clip(probs_k[(K, m, a, r_)], 1e-12, None)) for m in MODS).argmax(1)
                arr[f"pred_K{K}_POE_{a}_r{r_}"] = pr.astype(np.int8)
                rows.append(dict(K=str(K), modality="POE", arm=a, seed=r_, grid=-1, frozen=True, uar=100 * S.uar(yt, pr)))
    return rows, arr


# ------------------------------------------------------------------ 格子の点の選択（A の単位の出力から．C と集計の段で同じ関数を使う）

def select_points(csvs: pd.DataFrame) -> tuple[dict, list]:
    """コーパス c を評価するときは，c を評価のターゲット T にした単位（c がソースにも仮のターゲットにも入らない）のテストの UAR（adj）の平均が最大の点を選ぶ．
    K ごと．同点は並びで先（引く範囲は mod，本数は 20，信頼度は none，倍率は小さい側）．戻り：({(c, K, 版): 選んだ版}, 記録の行)．"""
    g = csvs[csvs.version.isin([x for L in SELDEF.values() for x in L])].groupby(["T", "K", "version"]).uar_adj.mean()
    choice, chrows = {}, []
    for c in DD.CORPORA:
        for K in KS:
            prod = float(csvs[(csvs["T"] == c) & (csvs.K == K) & (csvs.version == "prod")].uar_adj.mean())
            for name, cands in SELDEF.items():
                vals = [g.get((c, K, v), np.nan) for v in cands]
                if not np.isfinite(vals).any():
                    continue
                i = int(np.nanargmax(vals))
                choice[(c, K, name)] = cands[i]
                ci = int(cands[i].split("_c")[1].split("_")[0]) if "_c" in cands[i] else -1
                lam = LAM_OF.get(cands[i].rsplit("_", 1)[1], np.nan)
                cfg = CFGS[ci] if ci >= 0 else ("", "", -1, "")
                chrows.append(dict(c=c, K=K, version=name, chosen=cands[i], space=cfg[0], scope=cfg[1], k=cfg[2], rel=cfg[3], lam=lam,
                                   at_min_lam=bool(lam == LAMS[0]), at_max_lam=bool(lam == LAMS[-1]),
                                   n_tied=int(np.sum(np.asarray(vals) >= np.nanmax(vals) - 1e-9)), n_cands=len(cands), mean_uar_adj=float(np.nanmax(vals)), mean_uar_adj_prod=prod))
            for name, (src, fn) in TIED.items():
                if (c, K, src) in choice:
                    v = fn(choice[(c, K, src)])
                    choice[(c, K, name)] = v
                    chrows.append(dict(c=c, K=K, version=name, chosen=v, lam=LAM_OF.get(v.rsplit("_", 1)[1], np.nan), tied_to=src,
                                       mean_uar_adj=float(csvs[(csvs["T"] == c) & (csvs.K == K) & (csvs.version == v)].uar_adj.mean()), mean_uar_adj_prod=prod))
    return choice, chrows


def load_unit_csvs(out: Path, allow_partial: bool) -> tuple[list, pd.DataFrame]:
    files = sorted((out / "units").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    if not files:
        return files, pd.DataFrame()
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in files], ignore_index=True)
    if csvs.commit.nunique() != 1 and not allow_partial:
        raise SystemExit(f"単位のコミットが 1 つでない：{csvs.commit.unique()}")
    return files, csvs


# ------------------------------------------------------------------ C：まとめた答えを疑似ラベルにして生徒を学び直す

def teacher_pool(z, K: int, version: str, srcs: list) -> np.ndarray:
    """単位の保存（float32 の確率と信頼度の表）から，版 version の候補の上の疑似ラベルの確率（adj の層）を作り直す．単位の段と同じ式．"""
    LM = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in MODS])
    target = np.exp(P16.norm_log(LM.sum(0))).mean(0)
    if version == "prod":
        return np.exp(P16.norm_log(LM.sum(0)))
    if version.startswith("MS_eq_"):
        LS = np.stack([P16.centered(z[f"p_pool_src_{s}_{m}"].astype(np.float64)) for s in srcs for m in MODS])
        sc = LM.sum(0) + LAM_OF[version.rsplit("_", 1)[1]] * src_avg(LS, len(srcs))
    else:
        _, f, cpart, lpart = version.split("_")
        ci = int(cpart[1:])
        LN = np.stack([P16.centered(p_.astype(np.float64)) for p_ in z[f"p_pool_knn_c{ci}"]])
        W = mn_weights(f, LAM_OF[lpart], z[f"D_M_K_K{K}"], z[f"D_N_K_K{K}_c{ci}"], z[f"D_M_orc_K{K}"], z[f"D_N_orc_K{K}_c{ci}"], K)
        sc = np.einsum("ky,kny->ny", W, np.concatenate([LM, LN]))
    Sp = P16.norm_log(sc)
    dl, _ = P16.match_shift(Sp, target)
    return np.exp(P16.norm_log(Sp + dl))


def run_students(sv, z, choice: dict) -> tuple[list, dict]:
    u = sv.u
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    arr["y_test"], arr["cl_test"], arr["sid_test"] = yt.astype(np.int8), sv.cl, sv.sid
    for K in KS:
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all(), "単位の保存の K 本の位置が作り直した分割と違う"
        nk = np.setdiff1d(np.arange(u.n), kpos)
        teach = {}
        for t_ in TEACH:
            v = "prod" if t_ == "prod" else choice.get((u.c, K, t_))
            if v is None:
                continue                                                  # 試走で c を評価のターゲットにした単位が無いとき
            Q = teacher_pool(z, K, v, u.srcs)
            teach[t_] = Q
            pred_saved = z[f"pred_pool_K{K}_{v}"].astype(int)
            rows.append(dict(K=K, student="_teacher", teacher=t_, chosen=v, uar_pool=100 * S.uar(yp[nk], Q[nk].argmax(1)), maxp=float(Q.max(1).mean()),
                             agree_saved=float((Q[nk].argmax(1) == pred_saved).mean())))
            if rows[-1]["agree_saved"] < NONE_AGREE:
                raise SystemExit(f"{sv.base.name}：作り直した疑似ラベルが単位の保存の判定と {100 * (1 - rows[-1]['agree_saved']):.2f}% の発話で違う（{v} K={K}）")
        names = list(teach)
        gk = u.gK(K)
        nG = len(SGRID)
        for s_ in DD.STUDENTS:
            W0, b0 = u.start[s_]
            cw, cb = sv.teth[(s_, K)]
            pt = sv.fz["select"][f"{s_}|{K}|joint|one"]                    # 実験 1 の提案の主の版の設定（積の疑似ラベルに合わせて，同じ単位のテストで選ばれた値）
            g0 = [gi for gi, (r_, a_) in enumerate(SGRID) if abs(r_ - float(pt["rho"])) < 1e-12 and int(a_) == int(pt["a"])]
            if len(g0) != 1:
                raise SystemExit(f"{sv.base.name}：実験 1 の提案の設定 {pt} が格子に無い")
            # 疑似ラベル × 格子の全部の点を解く（問の並び：疑似ラベル × 格子）．引き戻しの強さは実験 1 の固定の値（K 本だけの分類器に合わせて選ばれた値）を共通に使う
            G = torch.stack([float(a_) * gk + S.soft_rows(S.t(teach[t_]), dai1.mval(r_, u.n)) for t_ in names for r_, a_ in SGRID])
            res = S.fit(u.Xp[s_], G, W0, b0, Cw=cw, Cb=cb)
            if float(res["grad_max"]) > P16.GRAD_TOL or not res["finite_p"].all():
                raise SystemExit(f"{sv.base.name}：生徒の学び直しが解けていない（{s_} K={K}．勾配の最大 {float(res['grad_max']):.1e}）")
            Z = S.logits(u.Xt[s_], res["W"], res["b"])
            Zp = S.logits(u.Xp[s_], res["W"], res["b"])
            Wp, bp = sv.prop(s_, K)                                        # 実験 1 の保存の提案．積の疑似ラベルで，実験 1 の設定の点で学んだ生徒と同じ答えになるはず
            agree = float((Z[names.index("prod") * nG + g0[0]].argmax(1) == S.logits(u.Xt[s_], Wp, bp).argmax(1)).float().mean())
            Wn, bn = sv.none(s_, K)
            none_t, none_p = S.logits(u.Xt[s_], Wn, bn).argmax(1).cpu().numpy(), S.logits(u.Xp[s_], Wn, bn).argmax(1).cpu().numpy()
            arr[f"pred_K{K}_{s_}_none"] = none_t.astype(np.int8)
            rows.append(dict(K=K, student=s_, teacher="_check", prop_agree=agree, grad=float(res["grad_max"]), stalled=int(res["stalled"]), rho=pt["rho"], a=pt["a"], grid_fixed=g0[0],
                             uar_none=100 * S.uar(yt, none_t), uar_pool_none=100 * S.uar(yp[nk], none_p[nk]),
                             gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
            if agree < NONE_AGREE:
                raise SystemExit(f"{sv.base.name}：積の疑似ラベルで学んだ生徒が，実験 1 の保存の提案と {100 * (1 - agree):.2f}% の発話で違う（{s_} K={K}）")
            for j, t_ in enumerate(names):
                for gi, (r_, a_) in enumerate(SGRID):
                    q = j * nG + gi
                    pr = Z[q].argmax(1).cpu().numpy()
                    pp = Zp[q].argmax(1).cpu().numpy()
                    arr[f"pred_K{K}_{s_}_{t_}_g{gi}"] = pr.astype(np.int8)
                    arr[f"prob_K{K}_{s_}_{t_}_g{gi}"] = torch.softmax(Z[q], -1).cpu().numpy().astype(np.float16)   # 格子の全点の確率（主の比べは集計の段で選び直す点）
                    rows.append(dict(K=K, student=s_, teacher=t_, grid=gi, rho=r_, a=a_, fixed=(gi == g0[0]), uar=100 * S.uar(yt, pr), uar_pool=100 * S.uar(yp[nk], pp[nk]),
                                     mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean())))   # mimic：生徒が疑似ラベルの判定を候補の上で再現した割合
                arr[f"pred_K{K}_{s_}_{t_}"] = arr[f"pred_K{K}_{s_}_{t_}_g{g0[0]}"]                      # 実験 1 の設定の点
    return rows, arr


# ------------------------------------------------------------------ 段

def stage_units(out: Path, targets: list, folds=None, seeds=None) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    udir = out / "units"
    udir.mkdir(parents=True, exist_ok=True)
    for T in targets:
        todo = dai3.pseudo_units(OUT1, T)
        if folds is not None:
            todo = [x for x in todo if x[1] in folds]
        if seeds is not None:
            todo = [x for x in todo if x[2] in seeds]
        for c, f, r in todo:
            name = f"{T}_{c}_f{f}_r{r}"
            if (udir / f"{name}.csv").exists():
                if (pd.read_csv(udir / f"{name}.csv").commit == RUN["commit"]).all():
                    print(f"{name}：済み（飛ばす）", flush=True)
                    continue
                raise SystemExit(f"{name} が別のコミットで既にある")
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fz1[T])
            save_srcdb(out, sv.u.srcs)
            rows, arr = run_unit(sv)
            df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0, dai1_commits=",".join(sorted(sv.commits)))
            np.savez_compressed(udir / f"{name}.npz", **arr)
            df.to_csv(udir / f"{name}.csv", index=False)
            ck = df[df.version == "_check"]
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(df)}．作り直しの差 {ck.teacher_dev.max():.1e}．勾配の最大 {ck.grad.max():.1e}．止まり {int(ck.stalled.sum())}．"
                  f"GPU の最大 {ck.gpu_max_mb.max():.0f} MB", flush=True)
            del sv, rows, arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def stage_reweight(out: Path, targets: list, folds=None, seeds=None) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    rdir = out / "reweight"
    rdir.mkdir(parents=True, exist_ok=True)
    for T in targets:
        todo = dai3.pseudo_units(OUT1, T)
        combos = sorted({(c, f) for c, f, r in todo if folds is None or f in folds})
        for c, f in combos:
            ss = sorted(r for c2, f2, r in todo if c2 == c and f2 == f and (seeds is None or r in seeds))
            name = f"{T}_{c}_f{f}"
            if (rdir / f"{name}.csv").exists():
                if (pd.read_csv(rdir / f"{name}.csv").commit == RUN["commit"]).all():
                    print(f"{name}：済み（飛ばす）", flush=True)
                    continue
                raise SystemExit(f"{name} が別のコミットで既にある")
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            rows, arr = run_reweight(T, c, f, fz1[T], ss)
            pd.DataFrame(rows).assign(T=T, c=c, fold=f, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0).to_csv(rdir / f"{name}.csv", index=False)
            np.savez_compressed(rdir / f"{name}.npz", **arr)
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(rows)}", flush=True)
            del rows, arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def stage_students(out: Path, allow_partial: bool) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    files, csvs = load_unit_csvs(out, allow_partial)
    choice, chrows = select_points(csvs)
    fp = out / "frozen_sel.json"                                          # 選んだ点を固定する（生徒の段の途中で選び直さない）
    rec = {f"{c}|{K}|{name}": v for (c, K, name), v in sorted(choice.items())}
    if fp.exists():
        if json.load(open(fp))["choice"] != rec:
            raise SystemExit("frozen_sel.json の選んだ点が，単位の出力から選び直した点と違う")
    else:
        json.dump(dict(choice=rec, units_commit=sorted(map(str, csvs.commit.unique())), n_units=len(files), time=time.strftime("%Y-%m-%d %H:%M:%S")),
                  open(fp, "w"), ensure_ascii=False, indent=1)
    sdir = out / "students"
    sdir.mkdir(exist_ok=True)
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        if (sdir / f"{p.stem}.csv").exists():
            if (pd.read_csv(sdir / f"{p.stem}.csv").commit == RUN["commit"]).all():
                print(f"{p.stem}：済み（飛ばす）", flush=True)
                continue
            raise SystemExit(f"{p.stem} が別のコミットで既にある")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fz1[T])
        rows, arr = run_students(sv, np.load(p, allow_pickle=False), choice)
        df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0)
        np.savez_compressed(sdir / f"{p.stem}.npz", **arr)
        df.to_csv(sdir / f"{p.stem}.csv", index=False)
        ck = df[df.teacher == "_check"]
        print(f"{p.stem}：{time.time() - t0:.0f} 秒．行 {len(df)}．提案との一致の最小 {ck.prop_agree.min():.4f}．勾配の最大 {ck.grad.max():.1e}", flush=True)
        del sv, rows, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def row_rng(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


def contrasts() -> list:
    full, two = ("adj", "kb", "raw", "pool", "ob"), ("adj", "kb")
    sel = ("N_eq_sel", "MS_eq_sel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MN_F6orc_sel")
    con = [(v, "prod", full) for v in sel] + [(v, "prod", two) for v in V_N + V_MN + V_CTRL + ("MN_eq_selmod", "MN_eq_seljoint", "MN_eq_selfeat", "MN_eq_sellogit", "MS_eq_tied")]
    con += [("MN_eq_sel", "MS_eq_sel", full), ("MN_eq_sel", "MS_eq_tied", full), ("N_eq_sel", "S_eq", full), ("MN_F2K_sel", "MN_eq_sel", two), ("MN_F6K_sel", "MN_F2K_sel", two),
            ("MN_F6orc_sel", "MN_F6K_sel", two), ("MN_eq_selmod", "MN_eq_selfeat", full), ("MN_eq_selmod", "MN_eq_sellogit", full), ("MN_eq_seljoint", "MN_eq_selmod", full)]
    for k in KNN_K:                                                      # 近いソースの発話の答えだけの積での比べ：信頼度，空間，引く範囲，本数
        for sc in ("mod", "joint"):
            con += [(f"N_eq_c{CI[('both', sc, k, 'away')]}", f"N_eq_c{CI[('both', sc, k, 'none')]}", two)]
        for sp in ("feat", "logit"):
            con += [(f"N_eq_c{CI[(sp, 'mod', k, 'away')]}", f"N_eq_c{CI[(sp, 'mod', k, 'none')]}", two)]
            for rl in ("none", "away"):
                con += [(f"N_eq_c{CI[('both', 'mod', k, rl)]}", f"N_eq_c{CI[(sp, 'mod', k, rl)]}", two)]
        for rl in ("none", "away"):
            con += [(f"N_eq_c{CI[('both', 'joint', k, rl)]}", f"N_eq_c{CI[('both', 'mod', k, rl)]}", two)]
    for sc in ("mod", "joint"):
        for rl in ("none", "away"):
            con += [(f"N_eq_c{CI[('both', sc, 100, rl)]}", f"N_eq_c{CI[('both', sc, 20, rl)]}", two)]
    con += [(f"N_eq_c{CI[('both', 'mod', 100, 'home')]}", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}", two)]
    con += [(v, ref, ("adj",)) for v in ("prod", "MN_eq_sel", "MN_F2K_sel") for ref in ("none_AVT", "joint_AVT")]
    merged: dict = {}
    for a, b, ws in con:
        merged[(a, b)] = tuple(dict.fromkeys(merged.get((a, b), ()) + tuple(ws)))
    return [(a, b, ws) for (a, b), ws in merged.items()]


CRITERIA = [("R1", "近いソースの発話を足す・全ラベルの信頼度", "MN_F6orc_sel", "prod"),
            ("R2", "近いソースの発話を足す・等しい重み", "MN_eq_sel", "prod"), ("R2", "近いソースの発話を足す・水準は K 本", "MN_F2K_sel", "prod"),
            ("R2", "近いソースの発話を足す・感情ごとの信頼度は K 本", "MN_F6K_sel", "prod"),
            ("対照", "ソースの分類器を足す・等しい重み", "MS_eq_sel", "prod"), ("対照", "ソースの発話を引く − ソースの分類器を足す", "MN_eq_sel", "MS_eq_sel"),
            ("対照", "ソースの発話を引く − ソースの分類器を足す（同じ倍率）", "MN_eq_sel", "MS_eq_tied"),
            ("対照", "近いソースの発話だけ − ソースの分類器だけ", "N_eq_sel", "S_eq"),
            ("R3", "信頼度（別のソース）で重みづける − 重みづけない（両方の近さ，モダリティごと，100 本）", f"N_eq_c{CI[('both', 'mod', 100, 'away')]}", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}"),
            ("R3", "信頼度（別のソース）で重みづける − 重みづけない（両方の近さ，3 つを合わせる，100 本）", f"N_eq_c{CI[('both', 'joint', 100, 'away')]}", f"N_eq_c{CI[('both', 'joint', 100, 'none')]}"),
            ("R3", "信頼度（別のソース）で重みづける − 重みづけない（特徴だけ，100 本）", f"N_eq_c{CI[('feat', 'mod', 100, 'away')]}", f"N_eq_c{CI[('feat', 'mod', 100, 'none')]}"),
            ("R3", "信頼度（自分のコーパス）で重みづける − 重みづけない（両方の近さ，モダリティごと，100 本）", f"N_eq_c{CI[('both', 'mod', 100, 'home')]}", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}"),
            ("近さ", "両方の近さ − 特徴だけ（足した版）", "MN_eq_selmod", "MN_eq_selfeat"), ("近さ", "両方の近さ − 分類器の答えだけ（足した版）", "MN_eq_selmod", "MN_eq_sellogit"),
            ("近さ", "両方の近さ − 特徴だけ（近い発話だけ，100 本，信頼度なし）", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}", f"N_eq_c{CI[('feat', 'mod', 100, 'none')]}"),
            ("近さ", "両方の近さ − 分類器の答えだけ（近い発話だけ，100 本，信頼度なし）", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}", f"N_eq_c{CI[('logit', 'mod', 100, 'none')]}"),
            ("範囲", "3 つのモダリティを合わせて引く − モダリティごと（足した版）", "MN_eq_seljoint", "MN_eq_selmod"),
            ("範囲", "3 つのモダリティを合わせて引く − モダリティごと（近い発話だけ，100 本，信頼度なし）", f"N_eq_c{CI[('both', 'joint', 100, 'none')]}", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}"),
            ("比べ", "100 本 − 20 本（両方の近さ，モダリティごと，信頼度なし）", f"N_eq_c{CI[('both', 'mod', 100, 'none')]}", f"N_eq_c{CI[('both', 'mod', 20, 'none')]}"),
            ("比べ", "水準は K 本 − 等しい重み", "MN_F2K_sel", "MN_eq_sel"), ("比べ", "感情ごと − 水準だけ", "MN_F6K_sel", "MN_F2K_sel")]


def stage_summary(out: Path, allow_partial: bool) -> None:
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    files, csvs = load_unit_csvs(out, allow_partial)

    def level(d: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(d[k]) for k in d if k[0] == f]) for f in sorted({k[0] for k in d})]))

    def eff_row(A: dict, B: dict, c: str, key: tuple, pool: bool = False) -> dict:
        es = (P12.effect_stats_pool if pool else ST.effect_stats)(A, B, geff[c], row_rng(*key))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"], n_fr=es["n_units"], n_folds=es["n_folds"], p=es.get("p", np.nan),
                    df=es.get("df", np.nan), verdict=P16.verdict(es["lo"], es["hi"], es["effect"]))

    def count_row(d: pd.DataFrame) -> dict:
        up, dn = d[d.verdict == "上回る"].c.tolist(), d[d.verdict == "下回る"].c.tolist()
        return dict(n_corpora=len(d), n_up=len(up), n_down=len(dn), hit=len(up) >= 2, down2=len(dn) >= 2, n_up_noAFEW=len([x for x in up if x != "AFEW"]),
                    n_down_noAFEW=len([x for x in dn if x != "AFEW"]),
                    up=",".join(up), down=",".join(dn))

    choice: dict = {}
    if files:
        # ---- 格子の点の選択
        choice, chrows = select_points(csvs)
        pd.DataFrame(chrows).to_csv(sdir / "sel.csv", index=False)
        fp = out / "frozen_sel.json"
        if fp.exists() and json.load(open(fp))["choice"] != {f"{c}|{K}|{name}": v for (c, K, name), v in sorted(choice.items())} and not allow_partial:
            raise SystemExit("frozen_sel.json の選んだ点が，単位の出力から選び直した点と違う")
        layers = {"adj": "pred_adj", "kb": "pred_kb", "raw": "pred_raw", "pool": "pred_pool", "ob": "pred_ob"}
        cells = {w: {} for w in layers}
        emo: dict = {}
        for p in files:
            T, c, f, r = p.stem.rsplit("_", 3)
            f, r = int(f[1:]), int(r[1:])
            z = np.load(p, allow_pickle=False)
            yt = z["y_test"].astype(int)
            for K in KS:
                nkm = np.ones(len(z["y_pool"]), bool)
                nkm[z[f"kpos_K{K}"]] = False
                ypk, clp = z["y_pool"].astype(int)[nkm], z["cl_pool"][nkm]
                names = {v: v for v in BASE + REFS}
                names.update({name: choice[(c, K, name)] for name in list(SELDEF) + list(TIED) if (c, K, name) in choice})
                for v, src in names.items():
                    for w, key in layers.items():
                        kk = f"{key}_K{K}_{src}"
                        if v in REFS and w in ("adj", "kb", "ob"):
                            kk = f"pred_raw_K{K}_{src}"
                        if kk not in z.files:
                            continue
                        pred = z[kk].astype(int)
                        cnt = ST.counts(ypk, pred, clp) if w == "pool" else ST.counts(yt, pred, z["cl_test"])
                        P12._add(cells[w].setdefault((c, K, v), {}), (f, r), cnt, f"{p.stem} {w} {v}")
                        if w == "adj" and v in PER_EMO:
                            e_ = emo.setdefault((c, K, v), [np.zeros(4), np.zeros(4), np.zeros(4), 0])
                            e_[0] += cnt[1].sum(0); e_[1] += cnt[2].sum(0); e_[2] += np.bincount(pred, minlength=4); e_[3] += 1
        for w in cells:
            P12._finish(cells[w])
        pd.DataFrame([dict(c=c, K=K, version=v, emotion=i, recall=100 * e_[0][i] / max(e_[1][i], 1), pred_share=100 * e_[2][i] / max(e_[2].sum(), 1),
                           true_share=100 * e_[1][i] / max(e_[1].sum(), 1), n_units=e_[3]) for (c, K, v), e_ in sorted(emo.items()) for i in range(4)]).to_csv(sdir / "per_emotion.csv", index=False)
        pd.DataFrame([dict(c=c, K=K, version=v, **{f"uar_{w}": (level(cells[w][(c, K, v)]) if (c, K, v) in cells[w] else np.nan) for w in layers}, n_fr=len(d))
                      for (c, K, v), d in sorted(cells["adj"].items())]).to_csv(sdir / "levels.csv", index=False)
        eff = []
        for c in DD.CORPORA:
            for K in KS:
                for a, b, ws in contrasts():
                    for w in ws:
                        A, B = cells[w].get((c, K, a)), cells[w].get((c, K, b))
                        if A is None or B is None:
                            continue
                        if set(A) != set(B):
                            raise SystemExit(f"{c} K={K} {a} と {b} の単位がそろわない（{w}）")
                        eff.append(dict(c=c, K=K, a=a, b=b, where=w, **eff_row(A, B, c, ("A", c, K, a, b, w), pool=(w == "pool"))))
        eff = pd.DataFrame(eff)
        eff.to_csv(sdir / "effects.csv", index=False)
        crit = []
        for rule, axis, a, b in CRITERIA:
            for K in KS:
                for w in ("adj", "kb", "raw", "pool", "ob"):
                    d = eff[(eff.a == a) & (eff.b == b) & (eff["where"] == w) & (eff.K == K)]
                    if len(d):
                        crit.append(dict(rule=rule, axis=axis, a=a, b=b, K=K, where=w, main=(K == K_MAIN and w == "adj"),
                                         proceed=(rule == "R2" and K == K_MAIN and w == "adj"), **count_row(d)))
        pd.DataFrame(crit).to_csv(sdir / "criteria.csv", index=False)
        # ---- 近い発話の記述
        nn = csvs[csvs.version == "_nn"]
        nn = nn[nn.seed == nn.groupby(["T", "c", "fold"]).seed.transform("min")]          # 近い発話の記述は種に依らない（同じ値が 2 回入るのを避ける）
        nb = nn.groupby(["c", "space", "modality", "by", "level"])[["hit", "share"]].agg(["mean", "std", "count"])
        nb.columns = [f"{a}_{b}" for a, b in nb.columns]
        nb.reset_index().to_csv(sdir / "neighbors.csv", index=False)
        nn[nn.by == "source"].groupby(["T", "c", "space", "modality", "level"])[["hit", "share", "db_share"]].mean().reset_index().to_csv(sdir / "neighbors_by_source.csv", index=False)
        # 記述の規則（D1〜D3）：単位ごとに量を出し，c ごとに平均する．2 点以上のコーパスを数える
        key = ["T", "c", "fold", "seed", "space", "modality"]
        pv = nn[nn.by == "rank"].pivot_table(index=key, columns="level", values="hit")
        pm = nn[nn.by == "rank_marginal"].pivot_table(index=key, columns="level", values="hit")
        pb = nn[nn.by == "rank_bal"].pivot_table(index=key, columns="level", values="hit")    # 感情ごとの当たる割合の平均．ラベルが本当の感情と無関係なら 0.25
        ch = nn[nn.by == "chance"].set_index(key).hit
        lift = pv - pm                                                            # 当たる割合 − ラベルの割合だけで決まる当たる割合
        q = pd.DataFrame({"D1_rank": 100 * (pb["1-5"] - pb["21-100"]), "D3_bal": 100 * (pb[f"1-{KMAX}"] - 0.25), "D3_assoc": 100 * lift[f"1-{KMAX}"],
                          "D1_assoc": 100 * (lift["1-5"] - lift["21-100"]),
                          "D1_rank_rawhit": 100 * (pv["1-5"] - pv["21-100"]), "D3_vs_table": 100 * (pv[f"1-{KMAX}"] - ch)})   # 後の 3 つは並べるだけ（規則に使わない）
        for rl in ("home", "away"):
            lab = nn[nn.by.isin([f"rel_{rl}_label{e}" for e in range(4)])].pivot_table(index=key + ["by"], columns="level", values="hit")
            q[f"D2_rel_{rl}"] = 100 * (lab["high"] - lab["low"]).groupby(level=key).mean()          # 近い発話のラベル 4 つの平均（欠けは除く）
        q = q.reset_index().melt(id_vars=key, var_name="quantity", value_name="value")
        perT = q.groupby(["c", "space", "modality", "quantity", "T"]).value.mean().reset_index()
        dr = q.groupby(["c", "space", "modality", "quantity"]).value.agg(["mean", "std", "count"]).reset_index()
        dr = dr.merge(perT.groupby(["c", "space", "modality", "quantity"]).value.agg(min_T="min", max_T="max").reset_index())
        dr["ge2"] = dr["mean"] >= D_MIN_PT
        d3 = dr[dr.quantity.isin(["D3_bal", "D3_assoc"])].groupby(["c", "space", "modality"]).agg(mean=("mean", "min"), std=("std", "max"), count=("count", "min"),
                                                                                              min_T=("min_T", "min"), max_T=("max_T", "min"), ge2=("ge2", "all")).reset_index()
        dr = pd.concat([dr, d3.assign(quantity="D3")], ignore_index=True)        # D3 の規則：2 つの量（D3_bal と D3_assoc）の両方が下限以上．mean の列は小さいほう
        dr.to_csv(sdir / "neighbors_rules.csv", index=False)
        dc = dr.groupby(["space", "modality", "quantity"]).apply(lambda z: pd.Series(dict(
            n_corpora=len(z), n_ge2=int(z.ge2.sum()), hit3=bool(z.ge2.sum() >= 3), n_ge2_noAFEW=int(z[z.c != "AFEW"].ge2.sum()), n_le_m2=int((z["mean"] <= -D_MIN_PT).sum()),
            corpora=",".join(z[z.ge2].c))), include_groups=False).reset_index()
        dc.to_csv(sdir / "neighbors_count.csv", index=False)
        csvs[csvs.version == "_single"].groupby(["c", "K", "modality", "name"])[["uar_raw", "sd_pool"]].agg(["mean", "std", "count"]).pipe(
            lambda x: x.set_axis([f"{a}_{b}" for a, b in x.columns], axis=1)).reset_index().to_csv(sdir / "single.csv", index=False)
        chk = csvs[csvs.version == "_check"]
        json.dump(dict(n_units=len(files), teacher_dev_max=float(chk.teacher_dev.max()), grad_max=float(chk.grad.max()), stalled=int(chk.stalled.sum()),
                       gpu_max_mb=float(chk.gpu_max_mb.max()),
                       adj_resid_max=float(chk.adj_resid.max()), sec_total=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())),
                  open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
        print(f"summary（A）：単位 {len(files)}，差の行 {len(eff)}，数えの行 {len(crit)}", flush=True)
    # ---- B
    rfiles = sorted((out / "reweight").glob("*.npz"))
    if len(rfiles) != N_COMBOS and not allow_partial:
        raise SystemExit(f"B の出力が {len(rfiles)} 個（{N_COMBOS} のはず）．試走なら --allow-partial")
    if rfiles:
        rc = pd.concat([pd.read_csv(p.with_suffix(".csv"), dtype={"K": str}) for p in rfiles], ignore_index=True)
        if rc.commit.nunique() != 1 and not allow_partial:
            raise SystemExit(f"B のコミットが 1 つでない：{rc.commit.unique()}")
        # 引き戻しの強さの選択：コーパス c を評価するときは，c を評価のターゲット T にした組（c がソースにも仮のターゲットにも入らない）のテストの UAR の平均が最大の点．重みごと
        gsel = rc[(rc.grid >= 0)].groupby(["T", "K", "modality", "arm", "grid"]).uar.mean()
        tch, trows = {}, []
        for c in DD.CORPORA:
            for K in KS:
                for m in MODS:
                    for a in ARMS:
                        vals = [gsel.get((c, str(K), m, a, gi), np.nan) for gi in range(len(TETH))]
                        if not np.isfinite(vals).any():
                            continue
                        gi = int(np.nanargmax(vals))                          # 同点は並びで先（強く引き戻す側）
                        tch[(c, K, m, a)] = gi
                        trows.append(dict(c=c, K=K, modality=m, arm=a, grid=gi, Cw=TETH[gi][0], Cb=TETH[gi][1], mean_uar=float(np.nanmax(vals)),
                                          at_edge=bool(TETH[gi][0] in (TETH[0][0], TETH[-1][0]))))
        pd.DataFrame(trows).to_csv(sdir / "reweight_sel.csv", index=False)
        cells, wrows, crows = {}, [], []
        for p in rfiles:
            T, c, f = p.stem.rsplit("_", 2)
            f = int(f[1:])
            z = np.load(p, allow_pickle=False)
            yt = z["y_test"].astype(int)
            d = rc[(rc["T"] == T) & (rc.c == c) & (rc.fold == f)]
            wrows.append(d[d.arm == "_weights"])
            crows.append(d[d.arm == "_check"])
            seeds = sorted({int(q_.group(1)) for q_ in (re.search(r"_r(\d+)$", k) for k in z.files if k.startswith("pred_K")) if q_})
            for a in ARMS:
                for m in list(MODS) + ["POE"]:
                    for r in seeds:
                        if m != "POE":
                            for Kk in ("0", "0ob"):
                                P12._add(cells.setdefault((c, Kk, m, a), {}), (f, r), ST.counts(yt, z[f"pred_K{Kk}_{m}_{a}"].astype(int), z["cl_test"]), f"{p.stem} K{Kk}")
                        for K in KS:
                            P12._add(cells.setdefault((c, str(K), m, a), {}), (f, r), ST.counts(yt, z[f"pred_K{K}_{m}_{a}_r{r}"].astype(int), z["cl_test"]), f"{p.stem} K{K}")
                            if (c, K, m, a) in tch:
                                P12._add(cells.setdefault((c, str(K), m, f"{a}_sel"), {}), (f, r),
                                         ST.counts(yt, z[f"pred_K{K}_{m}_{a}_g{tch[(c, K, m, a)]}_r{r}"].astype(int), z["cl_test"]), f"{p.stem} K{K} sel")
        P12._finish(cells)
        pd.DataFrame([dict(c=c, K=K, modality=m, arm=a, uar=level(d), n_fr=len(d)) for (c, K, m, a), d in sorted(cells.items())]).to_csv(sdir / "reweight_levels.csv", index=False)
        eff = []
        pairs = [(a, "base") for a in ARMS[1:]] + [(f"{a}_sel", "base_sel") for a in ARMS[1:]] + [("base_sel", "base")]
        for c in DD.CORPORA:
            for K in ("0", "0ob") + tuple(map(str, KS)):
                for m in list(MODS) + ["POE"]:
                    for a, b in pairs:
                        A, B = cells.get((c, K, m, a)), cells.get((c, K, m, b))
                        if A is None or B is None:
                            continue
                        if set(A) != set(B):
                            raise SystemExit(f"B：{c} K={K} {m} {a} と {b} の単位がそろわない")
                        eff.append(dict(c=c, K=K, modality=m, a=a, b=b, **eff_row(A, B, c, ("B", c, K, m, a, b)), uar_b=level(B)))
        eff = pd.DataFrame(eff)
        eff.to_csv(sdir / "reweight_effects.csv", index=False)
        crit = []
        for (K, m, a, b), q_ in eff.groupby(["K", "modality", "a", "b"]):
            main = bool(K == str(K_MAIN) and m in MODS and a.endswith("_sel") and b == "base_sel")
            crit.append(dict(K=K, modality=m, a=a, b=b, main=main, proceed=main, **count_row(q_)))
        pd.DataFrame(crit).to_csv(sdir / "reweight_criteria.csv", index=False)
        wcols = ["hub_mean", "hub_zero", "hub_top10_share", "base_dev", "ess_hub", "ess_rel", "ess_hubrel", "corr_hub_rel"] + [f"hub_share_label{i}" for i in range(4)]
        pd.concat(wrows).groupby(["c", "modality", "source"])[wcols].mean().reset_index().to_csv(sdir / "reweight_weights.csv", index=False)
        ck = pd.concat(crows)
        json.dump(dict(n_combos=len(rfiles), none_agree_min=float(ck.none_agree.min()), start_dev_max=float(ck.start_dev.max()), grad_max=float(ck.grad.max()),
                       stalled=int(ck.stalled.sum()), base_dev_max=float(pd.concat(wrows).base_dev.max()), gpu_max_mb=float(ck.gpu_max_mb.max())),
                  open(sdir / "reweight_checks.json", "w"), ensure_ascii=False, indent=1)
        print(f"summary（B）：組 {len(rfiles)}，差の行 {len(eff)}，数えの行 {len(crit)}", flush=True)
    # ---- C
    sfiles = sorted((out / "students").glob("*.npz"))
    if len(sfiles) != N_UNITS and not allow_partial:
        raise SystemExit(f"C の出力が {len(sfiles)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    if sfiles:
        sc = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in sfiles], ignore_index=True)
        if sc.commit.nunique() != 1 and not allow_partial:
            raise SystemExit(f"C のコミットが 1 つでない：{sc.commit.unique()}")
        gs = sc[sc.grid.notna() & (sc.grid >= 0)].groupby(["T", "K", "student", "teacher", "grid"]).uar.mean()
        sch, srows = {}, []
        for c in DD.CORPORA:                                              # 格子の点の選択：c を評価のターゲット T にした単位（c を含まない）のテストの UAR の平均が最大の点
            for K in KS:
                for s_ in DD.STUDENTS:
                    for t_ in TEACH:
                        vals = [gs.get((c, K, s_, t_, gi), np.nan) for gi in range(len(SGRID))]
                        if not np.isfinite(vals).any():
                            continue
                        gi = int(np.nanargmax(vals))                     # 同点は並びで先（疑似ラベルの重みが小さい側）
                        sch[(c, K, s_, t_)] = gi
                        srows.append(dict(c=c, K=K, student=s_, teacher=t_, grid=gi, rho=SGRID[gi][0], a=SGRID[gi][1], mean_uar=float(np.nanmax(vals)),
                                          n_tied=int(np.sum(np.asarray(vals) >= np.nanmax(vals) - 1e-9))))
        pd.DataFrame(srows).to_csv(sdir / "students_sel.csv", index=False)
        cells, emo = {}, {}
        for p in sfiles:
            T, c, f, r = p.stem.rsplit("_", 3)
            f, r = int(f[1:]), int(r[1:])
            z = np.load(p, allow_pickle=False)
            yt = z["y_test"].astype(int)
            for K in KS:
                for s in DD.STUDENTS:
                    for t_, kk in [(t_, f"pred_K{K}_{s}_{t_}") for t_ in TEACH + ("none",)] + \
                                   [(f"{t_}_sel", f"pred_K{K}_{s}_{t_}_g{sch[(c, K, s, t_)]}") for t_ in TEACH if (c, K, s, t_) in sch]:
                        if kk in z.files:
                            cnt = ST.counts(yt, z[kk].astype(int), z["cl_test"])
                            P12._add(cells.setdefault((c, K, s, t_), {}), (f, r), cnt, f"{p.stem} {kk}")
                            e_ = emo.setdefault((c, K, s, t_), [np.zeros(4), np.zeros(4), 0])
                            e_[0] += cnt[1].sum(0); e_[1] += cnt[2].sum(0); e_[2] += 1
        P12._finish(cells)
        pd.DataFrame([dict(c=c, K=K, student=s, teacher=t_, uar=level(d), n_fr=len(d)) for (c, K, s, t_), d in sorted(cells.items())]).to_csv(sdir / "students_levels.csv", index=False)
        pd.DataFrame([dict(c=c, K=K, student=s, teacher=t_, emotion=i, recall=100 * e_[0][i] / max(e_[1][i], 1), n_units=e_[2])
                      for (c, K, s, t_), e_ in sorted(emo.items()) for i in range(4)]).to_csv(sdir / "students_per_emotion.csv", index=False)
        eff = []
        pairs = ([(f"{t_}_sel", "prod_sel") for t_ in TEACH[1:]] + [("MN_eq_sel_sel", "MS_eq_sel_sel"), ("MN_eq_sel_sel", "MS_eq_tied_sel"), ("prod_sel", "prod"), ("prod_sel", "none")]
                 + [(t_, "prod") for t_ in TEACH[1:]] + [("MN_eq_sel", "MS_eq_sel"), ("MN_eq_sel", "MS_eq_tied"), ("prod", "none")])   # 主は格子の点を選び直した版（_sel）．実験 1 の設定のままの版は副
        for c in DD.CORPORA:
            for K in KS:
                for s in DD.STUDENTS:
                    for a, b in pairs:
                        A, B = cells.get((c, K, s, a)), cells.get((c, K, s, b))
                        if A is None or B is None:
                            continue
                        if set(A) != set(B):
                            raise SystemExit(f"C：{c} K={K} {s} {a} と {b} の単位がそろわない")
                        eff.append(dict(c=c, K=K, student=s, a=a, b=b, **eff_row(A, B, c, ("C", c, K, s, a, b))))
        eff = pd.DataFrame(eff)
        eff.to_csv(sdir / "students_effects.csv", index=False)
        crit = []
        for (K, s, a, b), q_ in eff.groupby(["K", "student", "a", "b"]):
            main = bool(K == K_MAIN and a in ("MN_eq_sel_sel", "MN_F2K_sel_sel", "MN_F6K_sel_sel") and b == "prod_sel")
            crit.append(dict(K=K, student=s, a=a, b=b, main=main, proceed=main, **count_row(q_)))
        pd.DataFrame(crit).to_csv(sdir / "students_criteria.csv", index=False)
        # 3 つの条件の量（記述）：疑似ラベルが今の分類器より当たる幅（候補の K 本以外）と，生徒が疑似ラベルを再現した割合
        tt = sc[sc.student == "_teacher"][["T", "c", "fold", "seed", "K", "teacher", "uar_pool", "maxp"]].rename(columns={"uar_pool": "uar_pool_teacher"})
        st = sc[(sc.teacher != "_check") & (sc.student != "_teacher") & (sc.fixed == True)][["T", "c", "fold", "seed", "K", "student", "teacher", "uar", "uar_pool", "mimic"]]  # noqa: E712  実験 1 の設定の点
        nn_ = sc[sc.teacher == "_check"][["T", "c", "fold", "seed", "K", "student", "uar_none", "uar_pool_none"]]
        cond = st.merge(tt, on=["T", "c", "fold", "seed", "K", "teacher"]).merge(nn_, on=["T", "c", "fold", "seed", "K", "student"])
        cond["edge"] = cond.uar_pool_teacher - cond.uar_pool_none                 # 条件 2 の量：疑似ラベルの UAR − K 本だけの分類器の UAR（候補）
        cond["gain_none"] = cond.uar - cond.uar_none
        cond.groupby(["c", "K", "student", "teacher"])[["edge", "mimic", "gain_none", "uar_pool_teacher", "maxp"]].mean().reset_index().to_csv(sdir / "students_conditions.csv", index=False)
        ck = sc[sc.teacher == "_check"]
        json.dump(dict(n_units=len(sfiles), prop_agree_min=float(ck.prop_agree.min()), grad_max=float(ck.grad.max()), stalled=int(ck.stalled.sum()),
                       teacher_agree_min=float(sc[sc.student == "_teacher"].agree_saved.min()), gpu_max_mb=float(ck.gpu_max_mb.max())),
                  open(sdir / "students_checks.json", "w"), ensure_ascii=False, indent=1)
        print(f"summary（C）：単位 {len(sfiles)}，差の行 {len(eff)}，数えの行 {len(crit)}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    rng = np.random.default_rng(0)
    # 1. 近い発話を引く：同じ点は自分が 1 番．票は，近い発話のラベルに集まる
    DB = unit_norm(rng.normal(size=(200, 16)))
    idx, sim = topk_idx(DB[:50], DB, k=10, with_sim=True)
    assert (idx[:, 0] == np.arange(50)).all() and np.allclose(sim[:, 0], 1, atol=1e-5) and (np.diff(sim, axis=1) <= 1e-6).all()
    yb = rng.integers(0, 4, 200)
    db = dict(y=yb, bal=np.ones(200), r_none=np.ones(200), r_home=rng.uniform(0.1, 1, 200), r_away=rng.uniform(0.1, 1, 200))
    v = votes(idx, db, 1, "none")
    assert (v.argmax(1) == yb[:50]).all() and np.allclose(v.sum(1), 1)
    v2 = votes(idx, db, 10, "home")
    assert np.allclose(v2.sum(1), 1) and (v2 > 0).all()
    assert np.allclose(votes(idx, dict(db, r_home=np.ones(200)), 10, "home"), votes(idx, db, 10, "none"))      # 信頼度が全部 1 なら，使わない版と同じ票
    # 2. 信頼度の重み：信頼度 0 の発話は票に入らない
    db0 = dict(db, r_home=np.where(yb == 0, 0.0, 1.0))
    assert np.allclose(votes(idx, db0, 10, "home")[:, 0], ALPHA / ((db0["r_home"][idx] > 0).sum(1) + 4 * ALPHA))
    # 3. 重みつきの one-hot：重みが全部 1 なら balanced_onehot と同じ．感情ごとの和は n/4．信頼度はラベルの中の比だけが入る
    y = rng.integers(0, 4, 60)
    assert np.allclose(weighted_onehot(y, np.ones(60)).cpu().numpy(), S.balanced_onehot(y, np.arange(60), 60).cpu().numpy())
    assert np.allclose(weighted_onehot(y, rng.uniform(0.2, 3, 60)).cpu().numpy().sum(0), 15.0)
    r = rng.uniform(0.1, 1, 60)
    rn = within_label(r, y)
    assert all(abs(rn[y == i].mean() - 1) < 1e-12 for i in range(4))
    assert np.allclose(weighted_onehot(y, rn).cpu().numpy(), weighted_onehot(y, r).cpu().numpy())
    tc = tercile_within(r, y)
    assert set(np.unique(tc)) == {0, 1, 2} and all(abs((tc[y == i] == t_).mean() - 1 / 3) < 0.15 for i in range(4) for t_ in range(3))
    # 4. 対照のソースの分類器の平均：並びは (ソース, モダリティ)
    L = np.zeros((6, 5, 4))
    for si in range(2):
        for mi in range(3):
            L[si * 3 + mi] = 10 * si + mi
    assert np.allclose(src_avg(L, 2)[0, 0], 5.0 + 6.0 + 7.0)
    # 5. 感情ごとの当たる割合の平均：近い発話のラベルが本当の感情と無関係なら，ラベルの割合が偏っていても 0.25
    yt_ = rng.choice(4, 4000, p=[0.57, 0.2, 0.13, 0.1])
    Y_ = rng.choice(4, (4000, 50), p=[0.7, 0.1, 0.1, 0.1])
    hb = np.mean([(Y_ == yt_[:, None])[yt_ == e].mean() for e in range(4)])
    assert abs(hb - 0.25) < 0.01, hb
    # 6. 並べた位置のコサインは，部分ごとのコサインの平均
    a1, a2, b1, b2 = (unit_norm(rng.normal(size=(7, d))) for d in (16, 16, 8, 8))
    assert torch.allclose((cat_norm([a1, b1]) * cat_norm([a2, b2])).sum(1), ((a1 * a2).sum(1) + (b1 * b2).sum(1)) / 2, atol=1e-6)
    assert torch.allclose(cat_norm([a1, b1]).norm(dim=1), torch.ones(7, device=a1.device), atol=1e-6)
    # 7. 重み：eq は，微調整した 3 つが 1，近い発話の答えが lam × (3 ÷ 答えの数)．信頼度が全部同じなら F2K・F6K も eq と同じ
    D3, D1 = np.full((3, 4), 0.8), np.full((1, 4), 0.8)
    W = mn_weights("eq", 0.25, D3, D1, D3, D1, 5)
    assert W.shape == (4, 4) and np.allclose(W[:3], 1) and np.allclose(W[3], 0.75)
    assert np.allclose(mn_weights("eq", 0.25, D3, D3, D3, D3, 5)[3:], 0.25)
    for f in ("F2K", "F6K", "F6orc"):
        assert np.allclose(mn_weights(f, 0.25, D3, D1, D3, D1, 5), W)
    Dz = np.zeros((3, 4))
    assert np.allclose(mn_weights("F6K", 1.0, Dz, D3, Dz, D3, 5)[:3], 1)                                  # 微調整した 3 つの平均が 0 なら全部 1
    # 8. 標準化
    E = rng.normal(size=(100, 8)) * 3 + 1
    Z, mu, sd = zs(E)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1)
    # 9. 版の名前
    names = BASE + tuple(SELDEF) + tuple(TIED) + REFS
    assert len(set(names)) == len(names)
    known = set(names)
    for a, b, _ in contrasts():
        assert a in known and b in known, (a, b)
    for _, _, a, b in CRITERIA:
        assert a in known and b in known, (a, b)
        assert any(x == (a, b) for x in [(c[0], c[1]) for c in contrasts()]), (a, b)
    assert all(set(v) <= set(BASE) for v in SELDEF.values())
    assert all(t_ == "prod" or t_ in SELDEF or t_ in TIED for t_ in TEACH)
    # 10. 標準化したコサインの平均：部分が 1 つなら，ふつうのコサインの順位と同じ．2 つの部分の片方を何倍にしても順位は変わらない
    Q = unit_norm(rng.normal(size=(30, 16))); D1_ = unit_norm(rng.normal(size=(300, 16)))
    Q2 = unit_norm(rng.normal(size=(30, 8))); D2_ = unit_norm(rng.normal(size=(300, 8)))
    i1, _, c1 = topk_z([(Q, D1_)], k=10)
    assert (i1 == topk_idx(Q, D1_, k=10)).all() and np.allclose(c1[..., 0], (Q @ D1_.T).cpu().numpy()[np.arange(30)[:, None], i1], atol=1e-5)
    ia = topk_z([(Q, D1_), (Q2, D2_)], k=10)[0]
    ib = topk_z([(Q, D1_), (Q2 * 0.1, D2_)], k=10)[0]
    assert (ia == ib).all()
    assert len(CFGS) == len(set(CFGS)) == 17 and NMAIN == 8 and all(len(v) > 0 for v in SELDEF.values())
    print(f"selftest：通った（版 {len(BASE)}，差の定義 {len(contrasts())}，格子 {len(CFGS)}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "reweight", "students", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=None, help="流す K（既定は KS）．渡すと出力は <OUT_PROD>_k<K...> の置き場だけ（追補の設計）")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    global KS, K_MAIN
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    if args.ks:
        KS = tuple(sorted(set(args.ks)))
        K_MAIN = K_MAIN if K_MAIN in KS else KS[0]                          # 追補の回では，その K を規則の主の K として数える
        out_k = Path(f"{OUT_PROD}_k{'_'.join(map(str, KS))}").resolve()
        if prod:
            sys.exit("--ks を渡すときは，既定の置き場ではなく別の置き場に書く")
        prod = out == out_k
        if prod and not dai1.committed_clean(ADDENDUM):
            sys.exit("追補の設計の文書がコミットされていない")
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or set(args.targets) != set(DD.CORPORA) or args.allow_partial):
        sys.exit("本番の置き場では対象を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    out.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                python=platform.python_version(), pandas=pd.__version__, this_script_sha256=dai15.sha(Path(__file__).resolve()),
                scripts={n: dai15.sha(src_dir / n) for n in SCRIPTS if (src_dir / n).exists()},
                design_sha256=(dai15.sha(DESIGN) if DESIGN.exists() else ""), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1],
                inputs_sha256={**{f"source/{q.name}": dai15.sha(q) for q in sorted((OUT1 / "source").glob("*.npz"))},
                               **{f"splits/{q.name}": dai15.sha(q) for q in sorted((ROOT / "splits").glob("*.csv"))}},
                KS=KS, KMAX=KMAX, CFGS=[list(map(str, c)) for c in CFGS], NMAIN=NMAIN, LAMS=LAMS, ALPHA=ALPHA, N0_LEVEL=N0_LEVEL, RANKS=RANKS, ARMS=ARMS, TEACH=TEACH,
                TETH=TETH, SGRID=SGRID, HUB_C=HUB_C, FLOOR=FLOOR, BASE_TOL=BASE_TOL, NONE_AGREE=NONE_AGREE, D_MIN_PT=D_MIN_PT, TAU=P16.TAU, GRAD_TOL=P16.GRAD_TOL,
                n_versions=len(BASE), rng_seed=SEED)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        stage_units(out, args.targets, args.folds, args.seeds)
        if prod:
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    elif args.stage == "reweight":
        stage_reweight(out, args.targets, args.folds, args.seeds)
        if prod:
            (out / "reweight.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    elif args.stage == "students":
        stage_students(out, args.allow_partial)
        if prod:
            (out / "students.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

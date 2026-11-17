"""【本番】E77（GPU）：DAI 論文の比較実験の第 1 波（`DAI_paper/reviews/comparison-design/proposal_v8.md`．設計は `docs/exploratory-analyses.md` E77）。

第 1 波は，E39 の学習器（`baselines_distill.Prob`）の束ねた解き方で解ける手法と，学習しない手法を回す．確率的な勾配法の手法
（DANN・MME・xMUDA 型・Co-teaching）と CORAL は第 2 波（別の台本）で回す．選択（仮のターゲットで K ごとに選ぶ）と判定は
ステップ 4 の解析の台本で行う．この台本は全候補の値を書き出すだけである．

  --kind  audio_wavlm / video / text_roberta（筋の 4 の主．生徒は種類の E39．E59・E61 の --kind と同じ作り方）
  --world fuse-audio-video-text / fuse-audio-video（筋の 5 の主と筋の 4 の副．生徒はモダリティごとと融合の E39．E60・E61 と同じ作り方）

手法（method, param, meff）：
  none     ：E39（K 本だけで学び直す．K=0 は出発点）
  saerens  ：Saerens ほか 2002 の EM で出力を補正（付録）
  massbal  ：候補の上の予測の感情ごとの総量が等しくなるよう出力を補正（群 A）
  self     ：自己学習（τ ∈ {0.5, 0.7, 0.9}．3 巡）
  entmin   ：エントロピー最小化（none の解から始める）
  im       ：情報最大化（SHOT の IM・TIM の形．α ∈ {0.1, 0.3, 1}．none の解から始める）
  t3a      ：T3A（主の版．切片を捨てる．M ∈ {1, 5, 20, 50, 100, all}）．t3a_b は切片を残す補助の版
  lp       ：ラベル伝播（K ≥ 1．k ∈ {10, 20, 50}，α_LP ∈ {0.9, 0.99}，形 ∈ {hard, soft}，γ = 3）の結果を弱いラベルにして E39 から学び直す
  kd       ：温度つき蒸留（同じ参照の候補．T ∈ {2, 4}）
  dist     ：参照の確率を弱いラベルにする（param は「参照:重みの付け方」．重みの付け方 ∈ {bal ①, raw ②, hard0.5・hard0.7・hard0.9 ③}）．
             oursx・除去実験の行・ours はこの行の部分集合として解析で選ぶ
  own      ：生徒自身の出発点の確率を釣り合いを取って弱いラベルにする（除去実験「自分の出発点」）
  cotrain・cotrain_prod・cotrain_fused・coreg・coreg_sum（--world のみ．E61 と同じ）
m_eff ∈ {100, 500, 2000, n}（全部の世界と種類．n は候補 1 本あたり重み 1）

保存（評価のターゲットだけ）：`<T>.csv`（target, rep, row_id, student, method, param, meff, K, uar），
`<T>_r<rep>_logp.npy`（行 × テストの発話 × 4 の float32 の対数の確率．行の並びは `<T>_r<rep>_rowids.npy`），`<T>_r<rep>_pred.npy`（予測した感情．uint8），
`<T>_r<rep>_te.npy`（テストの発話の行番号），`<T>_r<rep>_W_<生徒>.npy`・`_b_<生徒>.npy`・`_wrows_<生徒>.npy`（線形の分類器の生の特徴の上の W・b の float32 と行の番号），
`<T>_r<rep>_teachers.npz`（候補の上の教師の確率），`<T>_samples.csv`（行番号 → sample_id・話者），`<T>_settings.json`，`<T>_conv.csv`（所要と収束），`<T>_diag.csv`（疑似ラベルの記録），
`<T>_ps_<仮のターゲット>_r<rep>_none.npz`（仮のターゲットの none の対数の確率．δ の材料）．
仮のターゲット：`<T>_pseudo.csv`（pseudo, rep, student, method, param, meff, K, uar）．`provenance.json`・`provenance_log.jsonl`

    ./.venv/bin/python exp/compare_dai.py --kind audio_wavlm [--target …] [--reps …] [--pseudo-reps …] [--pseudo-targets …] [--out-dir …]
    ./.venv/bin/python exp/compare_dai.py --world fuse-audio-video-text
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob, hard_rows, mass, temp_rows, tens  # noqa: E402
from data import groups_for, load_any  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import WORLDS, beliefs1, blocks, make_students, sub_group  # noqa: E402
from select_labelfree import present_weights  # noqa: E402
from teach_student39 import Student  # noqa: E402

KS = BD.KS                                  # ("0", "1", "5", "10", "25")
KS_LAB = ("1", "5", "10", "25")
KS_LAB_ALL = KS_LAB                          # 話者の分割と K 本の抜き方（CB.Ctx）は常に全 K で作る．--ks で解く K を絞っても K=5・25 の K 本は変わらない
COPY_PSEUDO = False                          # True（--copy-pseudo）で COPY_SRC の写しを使う
NC_SEEDS = (1, 2)                            # 凸でない entmin・IM を評価のターゲットで揺らした出発点から解き直す種（種 0 は none の解から．--nc-seeds で変える）
NC_SCALE = 0.1
COREG_PSEUDO = True                          # False（--coreg-eval-only）：共正則化を仮のターゲットで解かない（DAI 論文の短縮 b）
COPY_SRC = {"IEMOCAP-script": "IEMOCAP-impro"}  # --copy-pseudo：台本の仮のターゲットは即興の結果を写す（DAI 論文の短縮 a．ソース・設定・分割が同じ）
C_FIT = BD.C_FIT                            # 0.01（合併の下書きの C．E60・E61 と同じ）
MEFFS = ("100", "500", "2000", "n")
TAUS = ("0.5", "0.7", "0.9")
WGTS = ("bal", "raw", "hard0.5", "hard0.7", "hard0.9")
KD_TEMPS = (2, 4)
IM_ALPHAS = (0.1, 0.3, 1.0)
T3A_MS = ("1", "5", "20", "50", "100", "all")
LP_KS = (10, 20, 50)
LP_ALPHAS = (0.9, 0.99)
LP_FORMS = ("hard", "soft")
LP_GAMMA = 3
LP_CG_ITERS = 50
REF_MIN = 500                                # 2 段目の参照を作る蒸留の m_in（E61 と同じ）
REF_KS = ("5", "25")                         # 合併とターゲットのみの参照の行を作る K（主表の K．計算量のため）
DT, DEV = logreg.DTYPE, logreg.DEVICE
GRID = {"KS": KS, "meffs": MEFFS, "taus": TAUS, "wgts": WGTS, "kd_temps": KD_TEMPS, "im_alphas": IM_ALPHAS, "t3a_ms": T3A_MS,
        "lp_ks": LP_KS, "lp_alphas": LP_ALPHAS, "lp_forms": LP_FORMS, "lp_gamma": LP_GAMMA, "lp_cg_iters": LP_CG_ITERS,
        "ref_ks": REF_KS, "rounds_self": BD.ROUNDS_SELF, "rounds_co": BD.ROUNDS_CO, "co_c": BD.CO_C, "ref_min": REF_MIN, "c_fit": C_FIT}


# ---------------------------------------------------------------- 記録

class Rec:
    """行（UAR）を集め，評価のターゲットならテストの発話ごとの対数の確率（float32）・予測した感情（uint8）と線形の分類器の W・b を持つ．
    仮のターゲット（keep=False）では none の行の対数の確率だけを持つ（判定の基準 δ の材料）。"""

    def __init__(self, tag, yte, keep):
        self.tag, self.yte, self.keep = tag, np.asarray(yte), keep
        self.rows, self.P, self.A, self.W, self.conv, self.none_lp, self.diag, self.xform = [], [], [], {}, [], {}, [], {}
        self.convp = []

    def add(self, student, method, param, meff, K, probs=None, Wb=None, Zte=None, logits=None):
        if Wb is not None:
            logits = Zte @ Wb[0].T + Wb[1]
        if logits is not None:
            lp = torch.log_softmax(torch.as_tensor(logits, dtype=DT, device=DEV), 1)
        else:
            if not torch.is_tensor(probs):
                probs = torch.as_tensor(np.asarray(probs), dtype=DT, device=DEV)
            lp = torch.log(probs.to(DT).clamp(min=1e-300))
            lp = lp - torch.logsumexp(lp, 1, keepdim=True)
        if not torch.isfinite(lp).all():
            raise FloatingPointError(f"NaN/Inf：{self.tag} {student} {method} {param} {meff} K={K}")
        pred = lp.argmax(1)
        u = CB.uar(self.yte, pred.cpu().numpy())
        rid = len(self.rows)
        self.rows.append({**self.tag, "row_id": rid, "student": student, "method": method, "param": str(param), "meff": str(meff),
                          "K": K, "uar": float(u)})
        if self.keep:
            self.P.append(lp.to(torch.float32).cpu().numpy())
            self.A.append(pred.to(torch.uint8).cpu().numpy())
            if Wb is not None and student in self.xform:                         # 生徒の座標 Z = ((X − c)/sd) @ Am の係数を生の特徴の上へ写す
                Am, sd, c = self.xform[student]
                Wr = (Wb[0] @ Am) / sd
                br = Wb[1] - Wr @ c
                self.W.setdefault(student, []).append((rid, Wr.to(torch.float32).cpu().numpy(), br.to(torch.float32).cpu().numpy()))
        elif method == "none":
            self.none_lp[(student, K)] = lp.to(torch.float32).cpu().numpy()

    def save(self, out, T, r, te):
        if not self.keep:
            return
        np.save(out / f"{T}_r{r}_logp.npy", np.stack(self.P))
        np.save(out / f"{T}_r{r}_pred.npy", np.stack(self.A))
        np.save(out / f"{T}_r{r}_rowids.npy", np.arange(len(self.rows)))
        np.save(out / f"{T}_r{r}_te.npy", np.asarray(te))
        if getattr(self, "teachers", None):
            np.savez(out / f"{T}_r{r}_teachers.npz", **self.teachers)
        for st, lst in self.W.items():                                          # 生の特徴の上の係数（logits = X_raw @ Wᵀ + b）
            np.save(out / f"{T}_r{r}_W_{st}.npy", np.stack([w for _, w, _ in lst]))
            np.save(out / f"{T}_r{r}_b_{st}.npy", np.stack([b for _, _, b in lst]))
            np.save(out / f"{T}_r{r}_wrows_{st}.npy", np.array([i for i, _, _ in lst]))

    def save_pseudo(self, out, T, c, r, te):
        """仮のターゲットの none の対数の確率（生徒 × K）とテストの行番号・ラベル。"""
        keys = sorted(self.none_lp)
        np.savez(out / f"{T}_ps_{c}_r{r}_none.npz", te=np.asarray(te), y=self.yte, keys=np.array([f"{a}|{b}" for a, b in keys]),
                 **{f"lp{i}": self.none_lp[k] for i, k in enumerate(keys)})


def tk(P, K):
    return P[K] if isinstance(P, dict) else P


class Prob2(Prob):
    """`baselines_distill.Prob` と同じ問を同じ手順で解き，束ねた解きごとに L-BFGS の反復の回数と止まったときの勾配の最大を記録する。"""

    def __init__(self, s, yq, label_sets):
        super().__init__(s, yq, label_sets)
        self.stats = []
        self.per = []                                                            # 問ごとの反復の回数と止まったときの勾配（関門の C6）

    def note(self, what, keys, fits):
        cp = self.s["cp"]
        for key, f in zip(keys, fits):
            self.per.append({"what": what, "key": str(key), "n_iter": int(f.n_iter[cp]), "grad": float(f.grad[cp])})

    def solve(self, items, u=None, inits=None, what="-"):
        s, cp = self.s, self.s["cp"]
        keys, weights, us, ini, out = [], [], [], [], {}
        for j, (key, w4, K) in enumerate(items):
            w = self.wt[K] if w4 is None else self.wt[K] + w4
            uj = 0.0 if u is None else float(u[j])
            if float(w.sum()) == 0.0 and uj == 0.0:
                out[key] = (self.W0, self.b0)
                continue
            keys.append(key)
            weights.append(w)
            us.append(uj)
            wb = None if inits is None else inits[j]
            ini.append(None if wb is None else logreg.Fitted(self.zero, self.one, {cp: wb[0]}, {cp: wb[1]}, {cp: 0}, {cp: 0.0}))
        if not keys:
            return out
        extra = None
        if any(us):
            uv = torch.tensor(us, dtype=DT, device=DEV)
            n1 = self.n

            def fn(L, ix):
                lp = torch.log_softmax(L[:n1], 2)
                return -(lp.exp() * lp).sum(2).sum(0) * (uv if ix is None else uv[ix])
            extra = (fn, [x * self.n for x in us])
        t0 = time.perf_counter()
        fits = logreg.fit_batch(self.X, self.y, [np.ones(len(self.y), bool)] * len(keys), (cp,), anchor=s["st"].anchor(s["rho_m"]),
                                anchor_c=s["st"].C, stats=(self.zero, self.one), Cb=[s["cb"]] * len(keys), weights=weights, extra=extra,
                                inits=ini if any(i is not None for i in ini) else None)
        self.stats.append({"what": what, "n_prob": len(keys), "sec": time.perf_counter() - t0,
                           "n_iter_max": max(int(f.n_iter[cp]) for f in fits), "grad_max": max(float(f.grad[cp]) for f in fits)})
        self.note(what, keys, fits)
        for key, f in zip(keys, fits):
            out[key] = (f.W[cp], f.b[cp])
        return out


# ---------------------------------------------------------------- 学習しない補正（確率を返す）

def saerens_p(Pp, Pt, iters=200):
    pi = np.full(4, 0.25)
    for _ in range(iters):
        A = Pp * pi
        A /= A.sum(1, keepdims=True)
        pi_new = A.mean(0)
        done = np.abs(pi_new - pi).max() < 1e-9
        pi = pi_new
        if done:
            break
    Q = Pt * pi
    return Q / Q.sum(1, keepdims=True)


def massbal_p(Pp, Pt, iters=500):
    a = np.ones(4)
    for _ in range(iters):
        A = Pp * a
        A /= A.sum(1, keepdims=True)
        a_new = a * 0.25 / np.clip(A.mean(0), 1e-12, None)
        a_new /= a_new.sum()
        done = np.abs(a_new - a).max() < 1e-12
        a = a_new
        if done:
            break
    Q = Pt * a
    return Q / Q.sum(1, keepdims=True)


# ---------------------------------------------------------------- 情報最大化（IM・TIM）

def im_mass(meff, K, n):
    """TIM の既定値（α = λ = 0.1）の固定版は，K 本の交差エントロピーの和（重み 4K）に対して λ = 0.1 なので m = 4K/λ = 40K。"""
    return 40.0 * int(K) if meff == "tim" else mass(meff, n)


def shaken(none, seed, rep_, student):
    """none の解の W に，W の要素の標準偏差の NC_SCALE 倍の正規の乱れを足した出発点（種・反復・生徒で決まる）。切片は動かさない。"""
    g = torch.Generator(device="cpu").manual_seed(int(1_000_003 * seed + 1_009 * rep_ + sum(map(ord, student))))
    out = {}
    for K in KS:
        W, b = none[K]
        e = torch.randn(W.shape, generator=g, dtype=torch.float64).to(device=W.device, dtype=W.dtype)
        out[K] = (W + NC_SCALE * W.std() * e, b)
    return out


def im_solve(pr, none, combos, what="im"):
    """combos = [(α, m_eff, K)]．目的に m·(α·条件のエントロピーの平均 − 予測の平均のエントロピー) を足し，none[K] の解から始める
    （出発点を揺らすときは none に揺らした解を渡す）。"""
    s, cp, n = pr.s, pr.s["cp"], pr.n
    if not combos:
        return {}
    alphas = torch.tensor([a for a, _, _ in combos], dtype=DT, device=DEV)
    ms = torch.tensor([im_mass(m, K, n) for _, m, K in combos], dtype=DT, device=DEV)

    def fn(L, ix):
        lp = torch.log_softmax(L[:n], 2)
        p = lp.exp()
        cond = -(p * lp).sum(2).mean(0)
        pbar = p.mean(0)
        marg = -(pbar * pbar.clamp(min=1e-12).log()).sum(1)
        a_, m_ = (alphas, ms) if ix is None else (alphas[ix], ms[ix])
        return m_ * (a_ * cond - marg)
    ini = [logreg.Fitted(pr.zero, pr.one, {cp: none[K][0]}, {cp: none[K][1]}, {cp: 0}, {cp: 0.0}) for _, _, K in combos]
    fits = logreg.fit_batch(pr.X, pr.y, [np.ones(len(pr.y), bool)] * len(combos), (cp,), anchor=s["st"].anchor(s["rho_m"]),
                            anchor_c=s["st"].C, stats=(pr.zero, pr.one), Cb=[s["cb"]] * len(combos),
                            weights=[pr.wt[K] for _, _, K in combos], extra=(fn, [float(x) for x in ms]), inits=ini)
    if hasattr(pr, "note"):
        pr.note(what, combos, fits)
    return {key: (f.W[cp], f.b[cp]) for key, f in zip(combos, fits)}


# ---------------------------------------------------------------- T3A

def t3a_probs(pr, Wb, lab_pos, lab_y, Ms):
    """Iwasawa & Matsuo 2021 の 3.1 節の形を候補の上で：支えの初期値は W の行を正規化したもの，候補は Wb の予測の感情に入れて
    エントロピーの小さい M 本を残す（K 本は真のラベルで足し，落とさない）．戻りは {M: (主の点数, 切片を残した補助の点数)}。"""
    Z, Zte = pr.s["Z"], pr.s["Zte"]
    zc = F.normalize(Z, dim=1)
    W, b = Wb
    L = Z @ W.T + b
    p = torch.softmax(L, 1)
    ent = -(p * p.clamp(min=1e-12).log()).sum(1)
    lab = p.argmax(1)
    unl = torch.ones(len(Z), dtype=torch.bool, device=Z.device)
    if len(lab_pos):
        unl[torch.as_tensor(lab_pos, device=Z.device)] = False
    init = F.normalize(W, dim=1)
    out = {}
    for Mv in Ms:                                                                # 公式の実装（matsuolab/T3A）：正規化した支えの和を正規化し，正規化しない特徴との内積
        protos = []
        for k in range(4):
            cand = torch.nonzero(unl & (lab == k)).squeeze(1)
            if Mv != "all" and len(cand) > int(Mv):
                cand = cand[torch.argsort(ent[cand])[:int(Mv)]]
            parts = [init[k:k + 1], zc[cand]]
            if len(lab_pos):
                lk = [int(i) for i, y_ in zip(lab_pos, lab_y) if int(y_) == k]
                if lk:
                    parts.append(zc[torch.as_tensor(lk, device=Z.device)])
            protos.append(F.normalize(torch.cat(parts).sum(0), dim=0))
        Pm = torch.stack(protos)
        Lt = Zte @ Pm.T
        out[Mv] = (Lt, Lt + b)
    return out


# ---------------------------------------------------------------- ラベル伝播（Zhou ほか 2004．Iscen ほか 2019 の形）

def lp_scores(Z, lab_pos, lab_y, k, alpha, gamma=LP_GAMMA, iters=LP_CG_ITERS, info=None):
    """候補の k 近傍のグラフ（余弦の類似度の γ 乗．負は 0）で K 本のラベルを伝播させた F（n×4．行は 0 以上）。共役勾配法で解く。"""
    n = len(Z)
    zn = F.normalize(Z.to(torch.float32), dim=1)
    kk = min(k, n - 1)
    ri, ci, vs = [], [], []
    for a in range(0, n, 2048):
        S = zn[a:a + 2048] @ zn.T
        ar = torch.arange(S.shape[0], device=Z.device)
        S[ar, a + ar] = -2.0
        v, i = S.topk(kk, dim=1)
        ri.append((a + ar).repeat_interleave(kk))
        ci.append(i.reshape(-1))
        vs.append(v.clamp(min=0).reshape(-1) ** gamma)
    ri, ci, vs = torch.cat(ri), torch.cat(ci), torch.cat(vs).to(DT)
    Wm = torch.sparse_coo_tensor(torch.stack([torch.cat([ri, ci]), torch.cat([ci, ri])]), torch.cat([vs, vs]), (n, n)).coalesce()
    d = torch.sparse.sum(Wm, 1).to_dense()
    dm = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    Y = torch.zeros(n, 4, dtype=DT, device=Z.device)
    Y[torch.as_tensor(lab_pos, device=Z.device), torch.as_tensor(lab_y, device=Z.device)] = 1.0

    def A(X):
        return X - alpha * dm[:, None] * torch.sparse.mm(Wm, dm[:, None] * X)
    Xs = torch.zeros_like(Y)
    R = Y - A(Xs)
    P = R.clone()
    rs = (R * R).sum(0)
    it = 0
    for it in range(1, iters + 1):
        AP = A(P)
        a_ = rs / (P * AP).sum(0).clamp(min=1e-30)
        Xs = Xs + a_ * P
        R = R - a_ * AP
        rs_new = (R * R).sum(0)
        if float(rs_new.max()) < 1e-20:
            break
        P = R + (rs_new / rs.clamp(min=1e-30)) * P
        rs = rs_new
    if info is not None:                                                         # 共役勾配法の回数と最後の残差（関門の C6）
        info.update({"cg_iters": it, "cg_resid": float(((Y - A(Xs)) ** 2).sum(0).max().sqrt())})
    return Xs.clamp(min=0)


def lp_rows(Fs, lab_pos, form, meff, n):
    """伝播の結果から弱いラベルの行の重み（K 本の行と，F が全部 0 の行は除く．釣り合いは取らない．総重み m_eff）。"""
    Fs = Fs.clone()
    if len(lab_pos):
        Fs[torch.as_tensor(lab_pos, device=Fs.device)] = 0.0
    tot = Fs.sum(1)
    ok = tot > 0
    if form == "hard":
        R = torch.zeros_like(Fs)
        R[torch.nonzero(ok).squeeze(1), Fs[ok].argmax(1)] = 1.0
    else:
        R = torch.where(ok[:, None], Fs / tot.clamp(min=1e-30)[:, None], torch.zeros_like(Fs))
    r = R.T.reshape(-1)
    if float(r.sum()) == 0.0:
        return r
    return r / r.sum() * mass(meff, n)


# ---------------------------------------------------------------- 群 A・B（1 つの生徒）

def run_ab(pr, rec, student, lab):
    """none・saerens・massbal・self・entmin・im・t3a・lp．戻りは none の解と候補の上の確率（K ごと）。lab[K] = (候補の中の位置, 真のラベル)。"""
    s, n = pr.s, pr.n
    Zte = s["Zte"]
    none = pr.solve([(K, None, K) for K in KS], what="none")
    P0 = {K: pr.probs(none[K], s["Z"]) for K in KS}
    for K in KS:
        rec.add(student, "none", "-", "0", K, Wb=none[K], Zte=Zte)
        Pp, Pt = P0[K].cpu().numpy(), pr.probs(none[K], Zte).cpu().numpy()
        rec.add(student, "saerens", "-", "0", K, probs=saerens_p(Pp, Pt))
        rec.add(student, "massbal", "-", "0", K, probs=massbal_p(Pp, Pt))
    keys = [(tau, meff, K) for tau in TAUS for meff in MEFFS for K in KS]
    cur = {key: P0[key[2]] for key in keys}
    for rnd in range(BD.ROUNDS_SELF):
        hr = {key: hard_rows(cur[key], key[0], key[1]) for key in keys}
        sol = pr.solve([(key, hr[key], key[2]) for key in keys], what="self")
        for key in keys:                                                         # 巡ごとの疑似ラベルの本数と正しさ（候補のラベルは測定だけに使う），テストの UAR
            on = hr[key].view(4, n).sum(0) > 0
            lab_ = hr[key].view(4, n).argmax(0)
            rec.diag.append({**rec.tag, "student": student, "method": "self", "param": key[0], "meff": key[1], "K": key[2], "round": rnd + 1,
                             "n_pseudo": int(on.sum()), "acc_pseudo": float((lab_[on] == pr.yt[on]).float().mean()) if on.any() else float("nan"),
                             "uar_test": float(CB.uar(rec.yte, (Zte @ sol[key][0].T + sol[key][1]).argmax(1).cpu().numpy()))})
        cur = {key: pr.probs(sol[key], s["Z"]) for key in keys}
    for (tau, meff, K), Wb in sol.items():
        rec.add(student, "self", tau, meff, K, Wb=Wb, Zte=Zte)
    keys = [(meff, K) for meff in MEFFS for K in KS]
    sol = pr.solve([(key, None, key[1]) for key in keys], u=[mass(meff, n) / n for meff, K in keys], inits=[none[K] for meff, K in keys],
                   what="entmin")
    for (meff, K), Wb in sol.items():
        rec.add(student, "entmin", "-", meff, K, Wb=Wb, Zte=Zte)
    im_combos = [(a, meff, K) for a in IM_ALPHAS for meff in MEFFS for K in KS] + [(0.1, "tim", K) for K in KS_LAB]
    for (a, meff, K), Wb in im_solve(pr, none, im_combos).items():
        rec.add(student, "im", f"a{a}", meff, K, Wb=Wb, Zte=Zte)
    if rec.keep and NC_SEEDS:                                                    # 凸でない 2 手法は出発点を揺らして解き直す（評価のターゲットだけ．関門の C1・C5）
        for seed in NC_SEEDS:
            st0 = shaken(none, seed, int(rec.tag["rep"]), student)
            sol = pr.solve([(key, None, key[1]) for key in keys], u=[mass(meff, n) / n for meff, K in keys],
                           inits=[st0[K] for meff, K in keys], what=f"entmin_s{seed}")
            for (meff, K), Wb in sol.items():
                rec.add(student, "entmin", f"-s{seed}", meff, K, Wb=Wb, Zte=Zte)
            for (a, meff, K), Wb in im_solve(pr, st0, im_combos, what=f"im_s{seed}").items():
                rec.add(student, "im", f"a{a}s{seed}", meff, K, Wb=Wb, Zte=Zte)
    for K in KS:
        for Mv, (lm, lb) in t3a_probs(pr, none[K], *lab[K], T3A_MS).items():
            rec.add(student, "t3a", f"M{Mv}", "0", K, logits=lm)
            rec.add(student, "t3a_b", f"M{Mv}", "0", K, logits=lb)
    items = []
    for K in KS_LAB:
        for k in LP_KS:
            for al in LP_ALPHAS:
                cg = {}
                Fs = lp_scores(s["Z"], *lab[K], k, al, info=cg)
                on = Fs.sum(1) > 0
                if len(lab[K][0]):
                    on[torch.as_tensor(lab[K][0], device=on.device)] = False
                rec.diag.append({**rec.tag, "student": student, "method": "lp", "param": f"k{k}a{al}", "meff": "-", "K": K, "round": 0,
                                 "n_pseudo": int(on.sum()),
                                 "acc_pseudo": float((Fs[on].argmax(1) == pr.yt[on]).float().mean()) if on.any() else float("nan"),
                                 "uar_test": float("nan"), **cg})
                for form in LP_FORMS:
                    for meff in MEFFS:
                        items.append(((f"k{k}a{al}{form}", meff, K), lp_rows(Fs, lab[K][0], form, meff, n), K))
    for (param, meff, K), Wb in pr.solve(items, what="lp").items():
        rec.add(student, "lp", param, meff, K, Wb=Wb, Zte=Zte)
    return none, P0


# ---------------------------------------------------------------- 群 C（参照を使う手法．1 つの生徒）

def kdh_solve(pr, items):
    """Hinton ほか 2015 の形の蒸留：生徒の出力にも温度 T をかけ，T² を掛けた教師との交差エントロピーを足す．
    items = [(key, 教師の確率 (n×4), T, m_eff, K)]．項は m_eff/n × T² × Σ_i Σ_k t_ik(T)·(−log s_ik(T))．出発点から解く。"""
    s, cp, n = pr.s, pr.s["cp"], pr.n
    if not items:
        return {}
    Tv = torch.tensor([float(T) for _, _, T, _, _ in items], dtype=DT, device=DEV)
    ms = torch.tensor([mass(m, n) for _, _, _, m, _ in items], dtype=DT, device=DEV)
    teach = torch.stack([torch.softmax(torch.log(tens(P).clamp(min=1e-12)) / float(T), 1) for _, P, T, _, _ in items], 1)   # n × 問 × 4

    def fn(L, ix):
        T_, m_, t_ = (Tv, ms, teach) if ix is None else (Tv[ix], ms[ix], teach[:, ix])
        ls = torch.log_softmax(L[:n] / T_[None, :, None], 2)
        return m_ / n * T_ ** 2 * (-(t_ * ls).sum(2).sum(0))
    fits = logreg.fit_batch(pr.X, pr.y, [np.ones(len(pr.y), bool)] * len(items), (cp,), anchor=s["st"].anchor(s["rho_m"]),
                            anchor_c=s["st"].C, stats=(pr.zero, pr.one), Cb=[s["cb"]] * len(items),
                            weights=[pr.wt[K] for _, _, _, _, K in items], extra=(fn, [float(x) for x in ms]))
    if hasattr(pr, "note"):
        pr.note("kdh", [it[0] for it in items], fits)
    return {key: (f.W[cp], f.b[cp]) for (key, _, _, _, _), f in zip(items, fits)}


def run_c(pr, rec, student, kd_teachers, dist_teachers, own):
    s, n, Zte = pr.s, pr.n, pr.s["Zte"]
    items = [((name, T, meff, K), temp_rows(tk(P, K), T, meff), K) for name, P in kd_teachers.items() for T in KD_TEMPS for meff in MEFFS for K in KS]
    for (name, T, meff, K), Wb in pr.solve(items, what="kd").items():
        rec.add(student, "kd", f"{name}T{T}", meff, K, Wb=Wb, Zte=Zte)
    items = [((name, T, meff, K), tk(P, K), T, meff, K) for name, P in kd_teachers.items() for T in KD_TEMPS for meff in MEFFS for K in KS]
    for (name, T, meff, K), Wb in kdh_solve(pr, items).items():
        rec.add(student, "kdh", f"{name}T{T}", meff, K, Wb=Wb, Zte=Zte)

    def rows_of(P, wgt, meff):
        if wgt == "bal":
            return weak_rows(P, mass(meff, n))
        if wgt == "raw":
            return temp_rows(P, 1, meff)
        return hard_rows(P, wgt[4:], meff)
    items = [((name, wgt, meff, K), rows_of(tk(P, K), wgt, meff), K)
             for name, P in dist_teachers.items() for wgt in WGTS for meff in MEFFS for K in KS]
    for (name, wgt, meff, K), Wb in pr.solve(items, what="dist").items():
        rec.add(student, "dist", f"{name}:{wgt}", meff, K, Wb=Wb, Zte=Zte)
    items = [((meff, K), weak_rows(own, mass(meff, n)), K) for meff in MEFFS for K in KS]
    for (meff, K), Wb in pr.solve(items, what="own").items():
        rec.add(student, "own", "bal", meff, K, Wb=Wb, Zte=Zte)


def cotrain_rec(prs, fused, rec, world):
    """E61 の cotrain と同じ手順（`baselines_distill.cotrain`）で，確率と W・b を記録する。"""
    views = list(prs)
    n = fused.n
    keys = [(c, K) for c in BD.CO_C for K in KS]
    pl = {key: torch.zeros(4, n, dtype=DT, device=DEV) for key in keys}
    taken = {key: set((torch.nonzero(prs[views[0]].wt[key[1]]).squeeze(1) % n).tolist()) for key in keys}
    for t in range(BD.ROUNDS_CO + 1):
        sols = {v: prs[v].solve([(key, pl[key].reshape(-1), key[1]) for key in keys]) for v in views}
        if t == BD.ROUNDS_CO:
            break
        picks = {key: [] for key in keys}
        for v in views:
            for key in keys:
                conf, lab = prs[v].probs(sols[v][key], prs[v].s["Z"]).max(1)
                used = taken[key] | {i for i, _ in picks[key]}
                conf = conf.clone()
                if used:
                    conf[torch.as_tensor(sorted(used), device=DEV)] = -1.0
                for k in range(4):
                    ck = torch.where(lab == k, conf, torch.full_like(conf, -1.0))
                    top = torch.topk(ck, min(key[0], n))
                    picks[key] += [(int(i), k) for i, c_ in zip(top.indices.tolist(), top.values.tolist()) if c_ >= 0]
        for key in keys:
            new = [(i, k) for i, k in picks[key] if i not in taken[key]]
            for i, k in new:
                pl[key][k, i] = 1.0
                taken[key].add(i)
            on = pl[key].sum(0) > 0
            yt0 = prs[views[0]].yt
            rec.diag.append({**rec.tag, "student": "*", "method": "cotrain", "param": f"c{key[0]}", "meff": "n", "K": key[1], "round": t + 1,
                             "n_pseudo": int(on.sum()), "acc_pseudo": float((pl[key].argmax(0)[on] == yt0[on]).float().mean()) if on.any() else float("nan"),
                             "uar_test": float("nan")})
    prod = {key: torch.zeros(len(rec.yte), 4, dtype=DT, device=DEV) for key in keys}
    for v in views:
        for key, Wb in sols[v].items():
            rec.add(v, "cotrain", f"c{key[0]}", "n", key[1], Wb=Wb, Zte=prs[v].s["Zte"])
            prod[key] = prod[key] + torch.log_softmax(prs[v].s["Zte"] @ Wb[0].T + Wb[1], 1)
    for key in keys:
        rec.add(world, "cotrain_prod", f"c{key[0]}", "n", key[1], probs=torch.softmax(prod[key], 1))
    for key, Wb in fused.solve([(key, pl[key].reshape(-1), key[1]) for key in keys]).items():
        rec.add(world, "cotrain_fused", f"c{key[0]}", "n", key[1], Wb=Wb, Zte=fused.s["Zte"])


def coreg_rec(prs, rec, world):
    """E61 の coreg と同じ問（`baselines_distill.coreg`）で，確率と W・b を記録する。"""
    views = list(prs)
    n = prs[views[0]].n
    B = len(MEFFS)
    mv = torch.tensor([mass(m, n) for m in MEFFS], dtype=DT, device=DEV)
    yt = prs[views[0]].yt
    for K in KS:
        wt = prs[views[0]].wt[K].view(4, n).sum(0)
        S = wt.sum() + mv
        params = {v: (prs[v].W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                      prs[v].b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True)) for v in views}
        opt = torch.optim.LBFGS([p for v in views for p in params[v]], lr=1.0, max_iter=3000, max_eval=6000, tolerance_grad=1e-6,
                                tolerance_change=1e-14, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = torch.zeros(B, dtype=DT, device=DEV)
            Lc = []
            for v in views:
                pr = prs[v]
                W, b = params[v]
                L = torch.einsum("nd,bkd->nbk", pr.s["Z"], W) + b[None]
                ce = -torch.log_softmax(L, 2).gather(2, yt.view(n, 1, 1).expand(n, B, 1)).squeeze(2)
                total = total + (wt[:, None] * ce).sum(0) + ((W - pr.W0) ** 2).sum((1, 2)) / (2 * pr.s["cp"]) + ((b - pr.b0) ** 2).sum(1) / (2 * pr.s["cb"])
                Lc.append(L - L.mean(2, keepdim=True))
            dis = sum(((Lc[a] - Lc[c]) ** 2).sum((0, 2)) for a in range(len(views)) for c in range(a + 1, len(views)))
            f = ((total + dis * (mv / n) / 2) / S).sum()
            f.backward()
            return f
        opt.step(closure)
        st = opt.state[opt._params[0]]
        rec.conv.append({**rec.tag, "method": "coreg", "K": K, "n_iter": int(st.get("n_iter", -1)),
                         "grad_max": float(max(p.grad.abs().max() for v in views for p in params[v]))})
        with torch.no_grad():
            Lsum = 0
            for v in views:
                W, b = params[v]
                Lte = torch.einsum("nd,bkd->nbk", prs[v].s["Zte"], W) + b[None]
                for j, meff in enumerate(MEFFS):
                    rec.add(v, "coreg", "-", meff, K, Wb=(W[j].detach(), b[j].detach()), Zte=prs[v].s["Zte"])
                Lsum = Lsum + Lte
            for j, meff in enumerate(MEFFS):
                rec.add(world, "coreg_sum", "-", meff, K, probs=torch.softmax(Lsum[:, j], 1))


# ---------------------------------------------------------------- 参照の行（ソースのみ・合併・ターゲットのみ・候補の全ラベルの上限）

def ref_rows(rec, student, pr, raw, label_sets):
    """raw = (候補の中心化した生の特徴, テストの中心化した生の特徴, [(ソースの生の特徴, ラベル)], 合併の C)．
    ref_srconly：E39 の下書きの尺度に使う合併のソースの分類器（ソースを合わせて中心化，C は合併の C）．
    ref_merged：ソース（各ソースを自分の平均で中心化）と K 本を合わせて学んだ分類器（同じ C）．ref_tgtonly：K 本だけで学んだ分類器（C = 1）．
    ref_wstar：候補の全ラベルで E39 の学習器を学び直した分類器（上限の参照．候補のラベルは参照にだけ使う）。"""
    if not rec.keep:                                                             # 参照の行は評価のターゲットでだけ作る（計算量．仮のターゲットの選択には使わない）
        return
    Xu, Xte, srcs, C = raw
    st = pr.s["st"]
    t0 = time.perf_counter()
    for K in KS:
        rec.add(student, "ref_srconly", "-", "0", K, logits=st.src.logits(Xte, st.C))
    Xs = torch.cat([X - X.mean(0) for X, _ in srcs])
    ys = np.concatenate([np.asarray(y) for _, y in srcs])
    yq = pr.yt
    for K in REF_KS:
        idx = torch.as_tensor(np.asarray(label_sets[K], int), device=DEV)
        f = logreg.fit(torch.cat([Xs, Xu[idx]]), M.labels(np.concatenate([ys, yq[idx].cpu().numpy()])), (C,))
        rec.add(student, "ref_merged", "-", "0", K, logits=f.logits(Xte, C))
        g = logreg.fit(Xu[idx], yq[idx], (1.0,))
        rec.add(student, "ref_tgtonly", "-", "0", K, logits=g.logits(Xte, 1.0))
    w = torch.zeros(4 * pr.n, dtype=DT, device=DEV)
    w[yq * pr.n + torch.arange(pr.n, device=DEV)] = present_weights(yq)
    Wb = pr.solve([("w", w, "0")], what="wstar")["w"]
    for K in KS:
        rec.add(student, "ref_wstar", "-", "0", K, Wb=Wb, Zte=pr.s["Zte"])
    pr.stats.append({"what": "ref", "n_prob": 0, "sec": time.perf_counter() - t0, "n_iter_max": -1, "grad_max": float("nan")})


def flush_stats(rec, prs):
    for k, pr in prs.items():
        for st_ in pr.stats:
            rec.conv.append({**rec.tag, "student": k, **st_})
        for st_ in getattr(pr, "per", []):
            rec.convp.append({**rec.tag, "student": k, **st_})


# ---------------------------------------------------------------- 世界と種類の 1 反復

def lab_of(label_sets, yq):
    yq = np.asarray(yq)
    return {K: (np.asarray(label_sets[K], int), yq[np.asarray(label_sets[K], int)] if len(label_sets[K]) else np.array([], int)) for K in KS}


def world_unit(world, stu, yq, yte, label_sets, rec, raw):
    mods = WORLDS[world]
    for k in mods + [world]:
        st = stu[k]["st"]
        rec.xform[k] = (st.mats[stu[k]["rho_m"]][0], st.src.sd, raw[k][4] + st.src.mu)
    prs = {k: Prob2(stu[k], yq, label_sets) for k in mods + [world]}
    for k in mods + [world]:
        ref_rows(rec, k, prs[k], raw[k][:4], label_sets)
    lab = lab_of(label_sets, yq)
    none_f, P0_f = run_ab(prs[world], rec, world, lab)
    pF = tens(stu[world]["P"])
    poe0 = tens(beliefs1(stu, world)["poe"])
    lg, lg0 = {K: 0 for K in KS}, {K: 0 for K in KS}
    lt, lt0, lpoe_t = {K: 0 for K in KS}, {K: 0 for K in KS}, 0
    for m in mods:                                                               # 2 段目の参照：ref は pFK，ref0 は pF に教わった積
        sol = prs[m].solve([(("ref", K), weak_rows(P0_f[K], REF_MIN), K) for K in KS] + [(("ref0", K), weak_rows(pF, REF_MIN), K) for K in KS])
        Zt_m = prs[m].s["Zte"]
        lpoe_t = lpoe_t + torch.log_softmax(Zt_m @ prs[m].W0.T + prs[m].b0, 1)
        for K in KS:
            lg[K] = lg[K] + prs[m].probs(sol[("ref", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
            lg0[K] = lg0[K] + prs[m].probs(sol[("ref0", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
            lt[K] = lt[K] + torch.log_softmax(Zt_m @ sol[("ref", K)][0].T + sol[("ref", K)][1], 1)
            lt0[K] = lt0[K] + torch.log_softmax(Zt_m @ sol[("ref0", K)][0].T + sol[("ref0", K)][1], 1)
    ref = {K: torch.softmax(lg[K], 1) for K in KS}
    ref0 = {K: torch.softmax(lg0[K], 1) for K in KS}
    Zt_f = prs[world].s["Zte"]                                                   # 教師モデルそのもののテストの行（ref・ref0 はテストでも作れる）
    for K in KS:
        rec.add(world, "teacher", "pF", "0", K, logits=Zt_f @ prs[world].W0.T + prs[world].b0)
        rec.add(world, "teacher", "pFK", "0", K, Wb=none_f[K], Zte=Zt_f)
        rec.add(world, "teacher", "poe0", "0", K, logits=lpoe_t)
        rec.add(world, "teacher", "ref", "0", K, logits=lt[K])
        rec.add(world, "teacher", "ref0", "0", K, logits=lt0[K])
    if rec.keep:                                                                 # 候補の上の教師の確率（第 2 波と同じ教師かを確かめる）
        rec.teachers = {"pF": pF.cpu().numpy(), "poe0": poe0.cpu().numpy(),
                        **{f"pFK_{K}": P0_f[K].cpu().numpy() for K in KS}, **{f"ref_{K}": ref[K].cpu().numpy() for K in KS},
                        **{f"ref0_{K}": ref0[K].cpu().numpy() for K in KS}}
    for m in mods:
        _, P0_m = run_ab(prs[m], rec, m, lab)
        run_c(prs[m], rec, m, {"pF": pF, "pFK": P0_f, "ref": ref}, {"pFK": P0_f, "ref": ref, "pF": pF, "ownK": P0_m}, tens(stu[m]["P"]))
    run_c(prs[world], rec, world, {"poe0": poe0, "ref": ref}, {"ref": ref, "ref0": ref0, "poe0": poe0, "ownK": P0_f}, pF)
    t0 = time.perf_counter()
    cotrain_rec({m: prs[m] for m in mods}, prs[world], rec, world)
    t1 = time.perf_counter()
    if rec.keep or COREG_PSEUDO:                                                 # --coreg-eval-only：仮のターゲットでは解かない（短縮 b）
        coreg_rec({m: prs[m] for m in mods}, rec, world)
    rec.conv.append({**rec.tag, "student": "*", "what": "cotrain_total", "sec": t1 - t0})
    if rec.keep or COREG_PSEUDO:
        rec.conv.append({**rec.tag, "student": "*", "what": "coreg_total", "sec": time.perf_counter() - t1})
    flush_stats(rec, prs)


def kind_unit(kind, s, yq, yte, label_sets, rec, raw):
    st = s["st"]
    rec.xform[kind] = (st.mats[s["rho_m"]][0], st.src.sd, raw[4] + st.src.mu)
    pr = Prob2(s, yq, label_sets)
    ref_rows(rec, kind, pr, raw[:4], label_sets)
    run_ab(pr, rec, kind, lab_of(label_sets, yq))
    flush_stats(rec, {kind: pr})


# ---------------------------------------------------------------- ターゲットごと

def check_resume(out, T, commit):
    """終わったターゲットは，同じコミットで作ったものだけ飛ばす（違えば止める）。"""
    f = out / f"{T}.csv"
    if not f.exists():
        return False
    c = set(pd.read_csv(f, usecols=["commit"]).commit.astype(str))
    if c != {commit}:
        raise SystemExit(f"{f} は別のコミット {c} で作られている（今は {commit}）．出力の場所を変えるか，消してから流し直す")
    return True


def set_ks(ks):
    """解く K を絞る（--ks．DAI 論文の短縮 c．判定に使う K だけ）．K 本の抜き方は KS_LAB_ALL のまま変えない。"""
    global KS, KS_LAB
    bad = [k for k in ks if k not in BD.KS]
    if bad or not ks:
        raise SystemExit(f"--ks は {BD.KS} の中から選ぶ（{ks}）")
    KS = tuple(k for k in BD.KS if k in ks)
    KS_LAB = tuple(k for k in KS if k != "0")
    GRID["KS"] = KS


def versions():
    """計算機とライブラリの版（再現の手順に書く．関門の C6）。"""
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        drv = ""
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": drv,
            "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0], "host": __import__("socket").gethostname()}


def pseudo_pairs(S, pseudo_targets, pseudo_reps):
    """仮のターゲットで計算する（群, 反復）の組（run_world・run_kind の繰り返しと同じ規則）。"""
    out = []
    for c in S:
        pool_c = [s_ for s_ in S if CM.corp(s_) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        out += [(c, int(r)) for r in pseudo_reps]
    return out


def copy_pseudo(out, T, script, ks, commit, pairs, args_now, drop_methods=()):
    """台本 T の仮のターゲットの結果を，同じ出力の場所にある写し元（即興）の結果から作る（--copy-pseudo．短縮 a）．

    止める条件（関門の I12-3）：写し元の `<src>.csv` が無い／今と違うコミット，`_settings.json` の不一致，
    写し元を回したときの引数（args_now の鍵）が今と違う，（仮のターゲット, 反復）の組が pairs とそろわない，K が ks とそろわない，
    E77 で npz の本数が組の数と違う．写した行には写し元のコミット `src_commit` と `copied_from` を残す。"""
    src = COPY_SRC[T]
    stop = lambda m: (_ for _ in ()).throw(SystemExit(f"{T} の写し：{m}"))  # noqa: E731
    f0 = out / f"{src}.csv"
    if not f0.exists():
        stop(f"写し元 {f0} が無い（写し元の評価の反復が終わっていない）")
    c0 = set(pd.read_csv(f0, usecols=["commit"]).commit.astype(str))
    if c0 != {commit}:
        stop(f"写し元のコミット {c0} が今 {commit} と違う")
    a_, b_ = (json.loads((out / f"{g}_settings.json").read_text()) for g in (src, T))
    if a_ != b_:
        stop(f"設定が違う：{a_} != {b_}")
    ent = [json.loads(x) for x in (out / "provenance_log.jsonl").read_text().splitlines() if x.strip()]
    ent = [e for e in ent if e.get("commit") == commit and src in e.get("targets", [])]
    if not ent:
        stop("写し元を回したときの記録（provenance_log.jsonl）が無い")
    for k, v in args_now.items():
        if json.dumps(ent[-1].get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            stop(f"写し元の引数 {k}={ent[-1].get(k)} が今 {v} と違う")
    d = pd.read_csv(out / f"{src}_pseudo.csv", dtype={"K": str})
    d = d[d.K.isin(ks) & ~d.method.isin(drop_methods)].rename(columns={"commit": "src_commit"}).assign(copied_from=src)
    got = set(zip(d.pseudo.astype(str), d.rep.astype(int)))
    if got != set(pairs):
        stop(f"（仮のターゲット, 反復）の組がそろわない：写し元 {sorted(got)}，期待 {sorted(pairs)}")
    if set(d.K) != set(ks):
        stop(f"K がそろわない：写し元 {sorted(set(d.K))}，今 {list(ks)}")
    extra = []
    for name in ("conv", "diag", "convp"):
        g = out / f"{src}_{name}.csv"
        e = pd.read_csv(g, dtype={"K": str}) if g.exists() and g.stat().st_size > 1 else pd.DataFrame()
        if not e.empty and "pseudo" in e.columns:
            e = e[e.pseudo.notna()].rename(columns={"commit": "src_commit"}).assign(copied_from=src)
            if "K" in e.columns:
                e = e[e.K.isna() | e.K.astype(str).str.replace(r"\.0$", "", regex=True).isin(ks)]
        elif name in ("conv", "convp") and script == "E77":
            stop(f"写し元の {g.name} が無い")
        extra.append(e.to_dict("records") if not e.empty else [])
    n_npz = 0
    if script == "E77":
        fs = [out / f"{src}_ps_{c}_r{r}_none.npz" for c, r in pairs]
        miss = [x.name for x in fs if not x.exists()]
        if miss:
            stop(f"写し元の npz が無い：{miss}")
        for g in fs:
            shutil.copyfile(g, out / g.name.replace(f"{src}_ps_", f"{T}_ps_", 1))
            n_npz += 1
    print(f"{T}：仮のターゲットを {src} から写した（{len(d)} 行，組 {len(pairs)}，npz {n_npz} 本）", flush=True)
    return d.to_dict("records"), extra[0], extra[1], extra[2]


def samples_table(name, T):
    df, _ = load_any(T, name)
    return df[["sample_id", "speaker_id", "speaker_group"]].reset_index(drop=True)


def save_settings(out, T, sets, extra=None):
    names = ("gamma", "rho_s", "rho_m", "cb", "cp", "step_c")
    (out / f"{T}_settings.json").write_text(json.dumps({"E39": {k: dict(zip(names, [None if v is None else float(v) for v in vals]))
                                                                 for k, vals in sets.items()}, **(extra or {})}, ensure_ascii=False, indent=1))


def run_world(world, T, data_f, data_m, reps, pseudo_reps, out, cache, commit, pseudo_targets=None):
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
    save_settings(out, T, sets, {"c_fit": C_FIT, "sources": S})
    prows, conv, diag, convp = [], [], [], []
    if COPY_PSEUDO and T in COPY_SRC:
        prows, conv, diag, convp = copy_pseudo(out, T, "E77", KS, commit, pseudo_pairs(S, pseudo_targets, pseudo_reps),
                                               {"KS": list(KS), "coreg_pseudo": COREG_PSEUDO, "nc_seeds": list(NC_SEEDS), "nc_scale": NC_SCALE},
                                               () if COREG_PSEUDO else ("coreg", "coreg_sum"))
        S_loop = []
    else:
        S_loop = S
    for c in S_loop:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        info_c = {k: M5.pool_info(data_m[k] if k != world else data_f, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            stu = make_students(world, c, data_f, data_m, pool_c, pool, te, sets, info_c)
            rec = Rec({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag, rec.convp = conv, diag, convp
            world_unit(world, stu, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), rec, raw_world(world, data_f, data_m, c, pool_c, pool, te))
            rec.save_pseudo(out, T, c, r, te)
            prows += rec.rows
            del stu, rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, C_FIT, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, reps, list(KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        rec = Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag, rec.convp = conv, diag, convp
        world_unit(world, stu, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec, raw_world(world, data_f, data_m, T, S, pool, te))
        rec.save(out, T, r, te)
        rows += rec.rows
        del stu, rec
        torch.cuda.empty_cache()
    pd.DataFrame(conv).assign(commit=commit).to_csv(out / f"{T}_conv.csv", index=False)
    pd.DataFrame(diag).assign(commit=commit).to_csv(out / f"{T}_diag.csv", index=False)
    pd.DataFrame(convp).assign(commit=commit).to_csv(out / f"{T}_convp.csv", index=False)
    pd.DataFrame(rows).assign(commit=commit).to_csv(out / f"{T}.csv", index=False)


def raw_world(world, data_f, data_m, T, S, pool, te):
    """世界の各生徒の (候補の中心化した生の特徴, テストの中心化した生の特徴, ソース, 合併の C, 中心)．中心は `make_students` と同じ候補の平均。"""
    out = {}
    for k in WORLDS[world] + [world]:
        dat_of = (lambda g: data_m[k][g]) if k != world else (lambda g: data_f[g])
        dat = dat_of(T)
        mu = M.rows(dat.X, pool).mean(0)
        out[k] = (M.rows(dat.X, pool) - mu, M.rows(dat.X, te) - mu, [(dat_of(s_).X, dat_of(s_).y) for s_ in S], C_FIT, mu)
    return out


def run_kind(kind, T, data, reps, pseudo_reps, out, cache, commit, pseudo_targets=None):
    """`baselines_distill.run_kind` と同じ生徒（仮のターゲットは合併の C = 0.01，評価は E25 の合併の C）。"""
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    save_settings(out, T, {kind: (gamma, rho_s, rho_m, cb, cp, None)}, {"c_fit_pseudo": C_FIT, "c_eval": C_T, "sources": S})

    def student_of(ev, Xc, src, lin, n_of, pool_s, C, te, ct):
        st = Student(ev, Xc, src, lin, n_of, pool_s, C, gamma, rho_s, [rho_m])
        Am = st.mats[rho_m][0]
        return {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((M.rows(ev.X, te) - ct - st.src.mu) / st.src.sd) @ Am}
    prows, conv, diag, convp = [], [], [], []
    if COPY_PSEUDO and T in COPY_SRC:
        prows, conv, diag, convp = copy_pseudo(out, T, "E77", KS, commit, pseudo_pairs(S, pseudo_targets, pseudo_reps),
                                               {"KS": list(KS), "coreg_pseudo": COREG_PSEUDO, "nc_seeds": list(NC_SEEDS), "nc_scale": NC_SCALE})
        S_loop = []
    else:
        S_loop = S
    for c in S_loop:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        src_c, lin_c, n_c = M5.pool_info(data, pool_c, C_FIT, cache)
        evc = data[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            s = student_of(evc, M.rows(evc.X, pool) - ctxc.ct[r], src_c, lin_c, n_c, pool_c, C_FIT, te, ctxc.ct[r])
            rec = Rec({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag, rec.convp = conv, diag, convp
            kind_unit(kind, s, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), rec,
                      (M.rows(evc.X, pool) - ctxc.ct[r], M.rows(evc.X, te) - ctxc.ct[r], [(data[s_].X, data[s_].y) for s_ in pool_c], C_FIT, ctxc.ct[r]))
            rec.save_pseudo(out, T, c, r, te)
            prows += rec.rows
            del s, rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, list(KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        s = student_of(ev, M.rows(ev.X, pool) - ctx.ct[r], src_pc, lin, n_of, S, C_T, te, ctx.ct[r])
        rec = Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag, rec.convp = conv, diag, convp
        kind_unit(kind, s, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), rec,
                  (M.rows(ev.X, pool) - ctx.ct[r], M.rows(ev.X, te) - ctx.ct[r], [(data[s_].X, data[s_].y) for s_ in S], C_T, ctx.ct[r]))
        rec.save(out, T, r, te)
        rows += rec.rows
        del s, rec
        torch.cuda.empty_cache()
    pd.DataFrame(conv).assign(commit=commit).to_csv(out / f"{T}_conv.csv", index=False)
    pd.DataFrame(diag).assign(commit=commit).to_csv(out / f"{T}_diag.csv", index=False)
    pd.DataFrame(convp).assign(commit=commit).to_csv(out / f"{T}_convp.csv", index=False)
    pd.DataFrame(rows).assign(commit=commit).to_csv(out / f"{T}.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--world")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pseudo-targets", nargs="+", help="試走用．仮のターゲットをこの群だけにする（空の指定 'none' で仮のターゲットを回さない）")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--gpu-fraction", type=float, default=0.6)
    ap.add_argument("--ks", nargs="+", help="解く K（既定は全部．DAI 論文の短縮 c は 0 5 25）．K 本の抜き方は変わらない")
    ap.add_argument("--copy-pseudo", action="store_true", help="IEMOCAP-script の仮のターゲットを同じ出力の場所の IEMOCAP-impro の結果から写す（短縮 a）")
    ap.add_argument("--coreg-eval-only", action="store_true", help="共正則化を仮のターゲットで解かない（短縮 b）")
    ap.add_argument("--nc-seeds", nargs="*", type=int, default=[1, 2], help="entmin・IM を揺らした出発点から解き直す種（評価のターゲットだけ．空で行わない）")
    args = ap.parse_args()
    if bool(args.kind) == bool(args.world):
        ap.error("--kind か --world のどちらか 1 つを指定する")
    global COPY_PSEUDO, COREG_PSEUDO, NC_SEEDS
    if args.ks:
        set_ks(args.ks)
    COPY_PSEUDO, COREG_PSEUDO, NC_SEEDS = args.copy_pseudo, not args.coreg_eval_only, tuple(args.nc_seeds)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    name = args.kind or args.world
    if args.world and args.world not in WORLDS:
        ap.error(f"世界は {list(WORLDS)} のどれか")
    out = args.out_dir or ROOT / "results" / f"e77_{name}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    groups = groups_for(name)
    todo = [t for t in (args.target or groups) if not check_resume(out, t, commit)]
    if not todo:
        print(f"{name}：全ターゲット済み", flush=True)
        return 0
    ptg = ["__none__"] if args.pseudo_targets == ["none"] else args.pseudo_targets   # 'none'：仮のターゲットを回さない（試走用）
    prov = {"script": "exp/compare_dai.py", "experiment": "E77", "argv": sys.argv[1:], "commit": commit, "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo, "reps": args.reps, "pseudo_reps": args.pseudo_reps,
            "pseudo_targets": args.pseudo_targets, "gpu_fraction": args.gpu_fraction, "copy_pseudo": COPY_PSEUDO, "coreg_pseudo": COREG_PSEUDO,
            "nc_seeds": list(NC_SEEDS), "nc_scale": NC_SCALE, **versions(), **GRID,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.world:
        data_f = {g: M.load(g, name) for g in groups}
        data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(name)}
        cache = {k: {} for k in WORLDS[name] + [name]}
    else:
        data = {g: M.load(g, name) for g in groups}
        cache = {}
    for t in todo:
        t0 = time.perf_counter()
        samples_table(name, t).to_csv(out / f"{t}_samples.csv", index_label="row")
        if args.world:
            run_world(name, t, data_f, data_m, args.reps, args.pseudo_reps, out, cache, commit, ptg)
        else:
            run_kind(name, t, data, args.reps, args.pseudo_reps, out, cache, commit, ptg)
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{name} {t}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

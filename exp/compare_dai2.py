"""【本番】E78（GPU）：DAI 論文の比較実験の第 2 波（`DAI_paper/reviews/comparison-design/proposal_v8.md` 3.2 節．設計は `docs/exploratory-analyses.md` E78）。

第 1 波（E77．`exp/compare_dai.py`）で回さなかった手法を，同じ生徒・同じ分割・同じ K 本で回し，同じ形で保存する．
  dann    ：DANN（Ganin ほか 2016）の目的を，特徴の線形の写像 A（恒等で初期化，‖A − I‖² の罰則）と E39 の頭に当てた版．仮説転移の形
            （K 本の損失 ＋ 出発点への罰則 ＋ DA の項．ソースは DA の項にだけ使う）．領域の判別器は隠れ層 1 つ（256）．逆の勾配の係数は原論文の予定で 0 から上げる
  mme     ：MME（Saito ほか 2019）の目的を同じ写像と頭に当てた版．頭は候補の予測のエントロピーを大きく，写像は小さくする（逆の勾配）．ソースのラベルの損失は小さい重みで足す
  mapctl  ：MME・DANN の同じ形の対照（DA の項の重み 0．同じ写像・罰則・初期値・つまみ）
  xmuda   ：xMUDA 型の相互の模倣（Jaritz ほか 2020．--world のモダリティの生徒だけ）．各モダリティに写像 A_m と主の頭・模倣の頭を置き，模倣の頭は
            候補の上で残りのモダリティの主の頭の予測（積を正規化，勾配を止める）との KL を学ぶ
  cot     ：Co-teaching（Han ほか 2018 の Algorithm 1．--world の全部の生徒）．参照の硬いラベル，同じミニバッチから 2 つの頭が損失の小さい割合 R(t) を選び合う．
            τ = 0 は対照（候補を全部使う）．τ_fix は固定版（仮のターゲットの候補で測った参照の誤りの割合の中央値．評価のターゲットでだけ回す）
  coral   ：CORAL（Sun ほか 2016）＋ none．ソースごとに，ソースと候補の上位 r 次元の PCA の部分空間の中で，ソースの共分散（Ledoit–Wolf）を候補の共分散に
            そろえてから E39 の下書きを作り直し，K 本で学び直す．(r, ρ_M, C′, C_b) は仮のターゲットで選ぶ（解析で）．標準化の基準と尺度の基準は元のソースの合併の下書きのまま
種は 3 つ（dann・mme・mapctl・xmuda・cot）．行の param に種を含める．選択と判定はステップ 4 の解析（E79）で，第 1 波の出力と合わせて行う。

保存は第 1 波と同じ形（`results/e78_<種類か世界>/`）．線形の分類器の W・b は，写像を掛けた後の z の上の係数（W A）で保存する．coral の W・b は
標準化した特徴（計量を掛ける前）の上の係数で保存する（`<T>_r<反復>_W_<生徒>.npy` の行に対応する `<T>.csv` の wspace の列）．
診断：`<T>_diag.csv`（cot の重なりの割合 v と正規化した ṽ，選んだ候補・落ちた候補の生徒と参照の一致の割合，仮のターゲットでは硬いラベルの正しさの割合），
`<T>_noise.csv`（仮のターゲットの候補で測った参照の誤りの割合）

    ./.venv/bin/python exp/compare_dai2.py --kind audio_wavlm [--target …] [--reps …] [--pseudo-reps …] [--pseudo-targets …] [--methods …]
    ./.venv/bin/python exp/compare_dai2.py --world fuse-audio-video-text
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import baselines_distill as BD  # noqa: E402
import combine_drafts as CD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import compare_dai as E77  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import Prob, mass, tens  # noqa: E402
from compare_dai import KS, Rec, check_resume, lab_of, samples_table  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import WORLDS, beliefs1, blocks, make_students, sub_group  # noqa: E402
from select_labelfree import present_weights  # noqa: E402
from teach_student39 import Student  # noqa: E402

DT, DEV = logreg.DTYPE, logreg.DEVICE
F32 = torch.float32
C_FIT = E77.C_FIT
KS_LAB = E77.KS_LAB
KS_LAB_ALL = E77.KS_LAB_ALL                    # K 本の抜き方は常に全 K（--ks で解く K を絞っても変わらない）
SEEDS = (0, 1, 2)
MAP_LRS, MAP_ITERS, MAP_LAMS, MAP_MUS, MME_SRCW = (1e-3, 1e-2), (200, 1000), (0.01, 0.1, 1.0), (1.0, 10.0), (0.0, 0.1)
XM_KAPPAS = (0.1, 1.0, 10.0)
CT_TAUS, CT_LRS, CT_MEFFS, CT_EPOCHS, CT_TK, CT_BATCH, CT_PERTURB = (0.0, 0.2, 0.4, 0.6), (1e-3, 1e-2), ("500", "n"), 50, 10, 128, 0.1
CT_STEPS = 2000                              # 更新の回数をターゲットに依らずそろえる（エポック数 = max(⌈2000/M⌉, T_k)．候補の少ないターゲットで収束しない件と計算量）
CORAL_R, CORAL_RHO, CORAL_CP, CORAL_CB, CORAL_MIN_RATIO = (32, 64, 128), (1.0, 0.97, 0.9), (0.01, 0.03, 0.1), (2.0, 8.0, None), 5
BATCH, DISC_H, SRC_BATCH = 128, 256, 128
GRID = {"seeds": SEEDS, "map_lrs": MAP_LRS, "map_iters": MAP_ITERS, "map_lams": MAP_LAMS, "map_mus": MAP_MUS, "mme_srcw": MME_SRCW,
        "xm_kappas": XM_KAPPAS, "ct_taus": CT_TAUS, "ct_lrs": CT_LRS, "ct_meffs": CT_MEFFS, "ct_epochs": CT_EPOCHS, "ct_tk": CT_TK,
        "ct_batch": CT_BATCH, "ct_perturb": CT_PERTURB, "ct_steps": CT_STEPS, "tf32": True, "coral_r": CORAL_R, "coral_rho": CORAL_RHO, "coral_cp": CORAL_CP,
        "coral_cb": [c if c is not None else "free" for c in CORAL_CB], "coral_min_ratio": CORAL_MIN_RATIO, "batch": BATCH,
        "disc_h": DISC_H, "src_batch": SRC_BATCH, "c_fit": C_FIT}
ALL_METHODS = ("dann", "mme", "mapctl", "xmuda", "cot", "coral")
COPY_PSEUDO = False                                                            # --copy-pseudo（短縮 a）


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef[:, None, None] * g, None


def grl(x, coef):
    """x（問 × 本数 × 次元）の勾配を，問ごとに −coef 倍にして返す。"""
    return _GRL.apply(x, coef)


def lab_tensors(lab, n_all_K="25"):
    """入れ子の K 本の和集合（K=25）の位置と真のラベル，K ごとの重み（present_weights．使わない行は 0）。"""
    pos, y = lab[n_all_K]
    pos_t = torch.as_tensor(pos, device=DEV, dtype=torch.long)
    y_t = torch.as_tensor(np.asarray(y), device=DEV, dtype=torch.long)
    wK = {}
    for K in KS:
        w = torch.zeros(len(pos), dtype=F32, device=DEV)
        if K != "0":
            pk = set(int(i) for i in lab[K][0])
            m = torch.tensor([int(i) in pk for i in pos], device=DEV)
            if m.any():
                w[m] = present_weights(y_t[m]).to(F32)
        wK[K] = w
    return pos_t, y_t, wK


def ce_rows(L, y):
    """L（問 × 本数 × 4）と y（本数）から本数ごとの交差エントロピー（問 × 本数）。"""
    return -torch.log_softmax(L, 2).gather(2, y.view(1, -1, 1).expand(L.shape[0], -1, 1)).squeeze(2)


# ---------------------------------------------------------------- DANN・MME・同じ形の対照

def map_problems(methods, seeds):
    """(方法, 学習率, 反復の回数, λ, μ, ソースの重み, 種, K) の全候補。"""
    out = []
    for lr in MAP_LRS:
        for iters in MAP_ITERS:
            for seed in seeds:
                if "dann" in methods:
                    out += [("dann", lr, iters, lam, mu, 0.0, seed, K) for lam in MAP_LAMS for mu in MAP_MUS for K in KS]
                if "mme" in methods:
                    out += [("mme", lr, iters, lam, mu, sw, seed, K) for lam in MAP_LAMS for mu in MAP_MUS for sw in MME_SRCW for K in KS]
                if "mapctl" in methods:
                    out += [("mapctl", lr, iters, 0.0, mu, 0.0, seed, K) for mu in MAP_MUS for K in KS]
    return out


def map_param(c):
    meth, lr, iters, lam, mu, sw, seed, K = c
    if meth == "mapctl":
        return f"lr{lr}it{iters}mu{mu}s{seed}"
    return f"lr{lr}it{iters}lam{lam}mu{mu}" + (f"sw{sw}" if meth == "mme" else "") + f"s{seed}"


def run_map(pr, Zs, ys, rec, student, lab, problems):
    """1 つの生徒で dann・mme・mapctl の問 problems を解く．問は (学習率, 反復の回数, 種) ごとに束ねる。"""
    if not problems:
        return
    Z = pr.s["Z"].to(F32)
    Zte = pr.s["Zte"].to(F32)
    Zs = Zs.to(F32)
    n, d = Z.shape
    W0, b0 = pr.W0.to(F32), pr.b0.to(F32)
    cp, cb = float(pr.s["cp"]), pr.s["cb"]
    pos, ylab, wK = lab_tensors(lab)
    Zl = Z[pos]
    ys_t = torch.as_tensor(np.asarray(ys), device=DEV, dtype=torch.long)
    I = torch.eye(d, dtype=F32, device=DEV)
    step = 30 if d <= 1024 else 12
    groups = {}
    for c in problems:
        groups.setdefault((c[1], c[2], c[6]), []).append(c)
    for (lr, iters, seed), probs in groups.items():
        for a in range(0, len(probs), step):
            chunk = probs[a:a + step]
            out = _train_map(Z, Zl, ylab, wK, Zs, ys_t, W0, b0, cp, cb, [(c[0], c[3], c[4], c[5], c[7]) for c in chunk], lr, iters, seed, I)
            for c, (Weff, bb, dg) in zip(chunk, out):
                rec.add(student, c[0], map_param(c), "-", c[7], Wb=(Weff.to(DT), bb.to(DT)), Zte=Zte.to(DT))
                rec.diag.append({**rec.tag, "student": student, "method": c[0], "param": map_param(c), "K": c[7], **dg})
            del out
            torch.cuda.empty_cache()


def _train_map(Z, Zl, ylab, wK, Zs, ys, W0, b0, cp, cb, chunk, lr, iters, seed, I):
    B = len(chunk)
    n, d = Z.shape
    g = torch.Generator(device=DEV).manual_seed(1000 + seed)                    # 判別器の初期値
    gb = torch.Generator(device=DEV).manual_seed(5000 + seed)                   # ミニバッチ（同じ種の DANN・MME・対照で同じ列）
    A = I.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
    W = W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
    b = b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True)
    D1 = (torch.randn(1, d, DISC_H, generator=g, device=DEV, dtype=F32) / math.sqrt(d)).repeat(B, 1, 1).requires_grad_(True)   # 束の組み方に依らない初期値
    c1 = torch.zeros(B, 1, DISC_H, device=DEV, dtype=F32, requires_grad=True)
    D2 = (torch.randn(1, DISC_H, 1, generator=g, device=DEV, dtype=F32) / math.sqrt(DISC_H)).repeat(B, 1, 1).requires_grad_(True)
    c2 = torch.zeros(B, 1, 1, device=DEV, dtype=F32, requires_grad=True)
    meth = [c[0] for c in chunk]
    lam = torch.tensor([c[1] for c in chunk], dtype=F32, device=DEV)
    mu = torch.tensor([c[2] for c in chunk], dtype=F32, device=DEV)
    sw = torch.tensor([c[3] for c in chunk], dtype=F32, device=DEV)
    wl = torch.stack([wK[c[4]] for c in chunk])                                   # 問 × K 本の和集合
    is_dann = torch.tensor([m_ == "dann" for m_ in meth], device=DEV)
    is_mme = torch.tensor([m_ == "mme" for m_ in meth], device=DEV)
    opt = torch.optim.Adam([A, W, b, D1, c1, D2, c2], lr=lr)
    first = {}
    for step in range(iters):
        p = step / max(iters - 1, 1)
        sched = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        it = torch.randint(0, n, (BATCH,), generator=gb, device=DEV)
        is_ = torch.randint(0, len(Zs), (SRC_BATCH,), generator=gb, device=DEV)
        opt.zero_grad()
        loss = torch.zeros(B, dtype=F32, device=DEV)
        if len(Zl):
            Hl = torch.einsum("nd,bed->bne", Zl, A)
            Ll = torch.einsum("bne,bke->bnk", Hl, W) + b[:, None]
            loss = loss + (wl * ce_rows(Ll, ylab)).sum(1)
        loss = loss + ((W - W0) ** 2).sum((1, 2)) / (2 * cp) + ((A - I) ** 2).sum((1, 2)) * mu
        if cb is not None:
            loss = loss + ((b - b0) ** 2).sum(1) / (2 * cb)
        Ht = torch.einsum("nd,bed->bne", Z[it], A)
        if is_dann.any():
            Hs = torch.einsum("nd,bed->bne", Zs[is_], A)
            Hcat = grl(torch.cat([Hs, Ht], 1), lam * sched)
            o = torch.einsum("bnh,bho->bno", torch.relu(torch.einsum("bnd,bdh->bnh", Hcat, D1) + c1), D2) + c2
            tgt = torch.cat([torch.ones(SRC_BATCH), torch.zeros(BATCH)]).to(DEV).view(1, -1, 1).expand(B, -1, 1)
            ld = F.binary_cross_entropy_with_logits(o, tgt, reduction="none").mean((1, 2))
            loss = loss + torch.where(is_dann, ld * n, torch.zeros_like(ld))
        if is_mme.any():
            Lt = torch.einsum("bne,bke->bnk", grl(Ht, torch.ones(B, device=DEV, dtype=F32)), W) + b[:, None]
            lp = torch.log_softmax(Lt, 2)
            H = -(lp.exp() * lp).sum(2).mean(1)
            Hs = torch.einsum("nd,bed->bne", Zs[is_], A)
            Ls = torch.einsum("bne,bke->bnk", Hs, W) + b[:, None]
            cs = ce_rows(Ls, ys[is_]).mean(1)
            loss = loss + torch.where(is_mme, -lam * n * H + sw * n * cs, torch.zeros_like(H))
        loss.sum().backward()
        opt.step()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("dann/mme の損失が NaN/Inf")
        if step in (0, iters - 1):                                               # DA の項が働いたかを測る量（最初と最後）
            with torch.no_grad():
                rec_ = {"ld": ld.detach() if is_dann.any() else None, "H": H.detach() if is_mme.any() else None}
                first.setdefault(step, rec_)
                first[step] = rec_
    with torch.no_grad():
        Weff = torch.einsum("bke,bed->bkd", W, A)
        an = ((A - I) ** 2).sum((1, 2)).sqrt()
        st0, st1 = first.get(0, {}), first.get(iters - 1, {})
        diag = [{"anorm": float(an[j]),
                 "dom_first": float(st0["ld"][j]) if st0.get("ld") is not None else float("nan"),
                 "dom_last": float(st1["ld"][j]) if st1.get("ld") is not None else float("nan"),
                 "ent_first": float(st0["H"][j]) if st0.get("H") is not None else float("nan"),
                 "ent_last": float(st1["H"][j]) if st1.get("H") is not None else float("nan")} for j in range(B)]
        return [(Weff[j].clone(), b[j].clone(), diag[j]) for j in range(B)]


# ---------------------------------------------------------------- xMUDA 型（世界のモダリティの生徒）

def xm_problems(seeds):
    return [(lr, iters, kap, mu, seed, K) for lr in MAP_LRS for iters in MAP_ITERS for seed in seeds for kap in XM_KAPPAS for mu in MAP_MUS for K in KS]


def run_xmuda(prs, mods, rec, lab, problems):
    Zs_ = {m: prs[m].s["Z"].to(F32) for m in mods}
    Zte_ = {m: prs[m].s["Zte"].to(F32) for m in mods}
    n = len(Zs_[mods[0]])
    pos, ylab, wK = lab_tensors(lab)
    groups = {}
    for c in problems:
        groups.setdefault((c[0], c[1], c[4]), []).append((c[2], c[3], c[5]))
    for (lr, iters, seed), chunk in groups.items():
        if True:
            if True:
                B = len(chunk)
                g = torch.Generator(device=DEV).manual_seed(2000 + seed)
                P = {}
                for m in mods:
                    d = Zs_[m].shape[1]
                    W0, b0 = prs[m].W0.to(F32), prs[m].b0.to(F32)
                    P[m] = {"A": torch.eye(d, device=DEV, dtype=F32).unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                            "W": W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                            "b": b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True),
                            "V": W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                            "c": b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True), "W0": W0, "b0": b0,
                            "I": torch.eye(d, device=DEV, dtype=F32), "cp": float(prs[m].s["cp"]), "cb": prs[m].s["cb"]}
                kap = torch.tensor([c[0] for c in chunk], dtype=F32, device=DEV)
                mu = torch.tensor([c[1] for c in chunk], dtype=F32, device=DEV)
                wl = torch.stack([wK[c[2]] for c in chunk])
                opt = torch.optim.Adam([P[m][k] for m in mods for k in ("A", "W", "b", "V", "c")], lr=lr)
                klrec = {}
                for step in range(iters):
                    it = torch.randint(0, n, (BATCH,), generator=g, device=DEV)
                    opt.zero_grad()
                    loss = torch.zeros(B, dtype=F32, device=DEV)
                    main_lp, mim_lp = {}, {}
                    for m in mods:
                        q = P[m]
                        H = torch.einsum("nd,bed->bne", Zs_[m][it], q["A"])
                        main_lp[m] = torch.log_softmax(torch.einsum("bne,bke->bnk", H, q["W"]) + q["b"][:, None], 2)
                        mim_lp[m] = torch.log_softmax(torch.einsum("bne,bke->bnk", H, q["V"]) + q["c"][:, None], 2)
                        if len(pos):
                            Hl = torch.einsum("nd,bed->bne", Zs_[m][pos], q["A"])
                            Ll = torch.einsum("bne,bke->bnk", Hl, q["W"]) + q["b"][:, None]
                            loss = loss + (wl * ce_rows(Ll, ylab)).sum(1)
                        loss = loss + ((q["W"] - q["W0"]) ** 2).sum((1, 2)) / (2 * q["cp"]) + ((q["A"] - q["I"]) ** 2).sum((1, 2)) * mu
                        if q["cb"] is not None:
                            loss = loss + ((q["b"] - q["b0"]) ** 2).sum(1) / (2 * q["cb"])
                    for m in mods:
                        tgt = torch.log_softmax(sum(main_lp[o].detach() for o in mods if o != m), 2)
                        kl = (tgt.exp() * (tgt - mim_lp[m])).sum(2).mean(1)
                        loss = loss + kap * n * kl
                        if step in (0, iters - 1):                               # 模倣の項が働いたかを測る量（最初と最後の KL）
                            klrec.setdefault(m, {})[step] = kl.detach()
                    loss.sum().backward()
                    opt.step()
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("xmuda の損失が NaN/Inf")
                with torch.no_grad():
                    for m in mods:
                        Weff = torch.einsum("bke,bed->bkd", P[m]["W"], P[m]["A"])
                        an = ((P[m]["A"] - P[m]["I"]) ** 2).sum((1, 2)).sqrt()
                        for j, (k_, mu_, K) in enumerate(chunk):
                            par = f"lr{lr}it{iters}kap{k_}mu{mu_}s{seed}"
                            rec.add(m, "xmuda", par, "-", K, Wb=(Weff[j].to(DT), P[m]["b"][j].to(DT)), Zte=Zte_[m].to(DT))
                            rec.diag.append({**rec.tag, "student": m, "method": "xmuda", "param": par, "K": K, "anorm": float(an[j]),
                                             "kl_first": float(klrec[m][0][j]), "kl_last": float(klrec[m][iters - 1][j])})
                del P, opt
                torch.cuda.empty_cache()


# ---------------------------------------------------------------- Co-teaching

def run_cot(pr, rec, student, lab, teachers, tau_fix=None, ytrue=None, diag=None):
    """teachers：{名前: K → 候補の上の確率}．tau_fix：{名前: τ}（評価のターゲットだけ．固定版）．ytrue：候補の真のラベル（仮のターゲットで診断だけに使う）．
    学習率と種も 1 回に束ねる（問ごとの学習率の Adam，種ごとのミニバッチの順）。"""
    Z = pr.s["Z"].to(F32)
    Zte = pr.s["Zte"].to(F32)
    n, d = Z.shape
    W0, b0 = pr.W0.to(F32), pr.b0.to(F32)
    cp, cb = float(pr.s["cp"]), pr.s["cb"]
    pos, ylab, wK = lab_tensors(lab)
    Mb = math.ceil(n / CT_BATCH)
    chunk = [(name, tau, meff, K, lr, seed, "ctl" if tau == 0 else "sel") for name in teachers for tau in CT_TAUS for meff in CT_MEFFS
             for K in KS for lr in CT_LRS for seed in SEEDS]
    if tau_fix is not None:
        chunk += [(name, float(tau_fix[name][K]), "n", K, 1e-3, seed, "fix") for name in teachers for K in KS for seed in SEEDS]
    name0 = next(iter(teachers))                                                 # 候補の項の重み 0（K 本と罰則だけ）．none との差で確率的な勾配法の届かなさを測る
    chunk += [(name0, 0.0, "0", K, lr, seed, "zero") for K in KS for lr in CT_LRS for seed in SEEDS]
    B = len(chunk)
    hl = torch.stack([E77.tk(teachers[c[0]], c[3]).to(F32).argmax(1) for c in chunk])            # 問 × n の硬いラベル
    lmask = torch.zeros(B, n, dtype=torch.bool, device=DEV)                                     # 問の K 本（選ぶ対象から外す）
    for j, c in enumerate(chunk):
        if c[3] != "0":
            lmask[j, torch.as_tensor(np.asarray(lab[c[3]][0], int), device=DEV)] = True
    seed_of = torch.tensor([SEEDS.index(c[5]) for c in chunk], device=DEV)
    gens = [torch.Generator(device=DEV).manual_seed(3000 + sd) for sd in SEEDS]
    sdW = W0.std() * CT_PERTURB
    noise = torch.stack([torch.randn(4, d + 1, generator=gens[int(i)], device=DEV, dtype=F32) for i in seed_of.tolist()]) * sdW
    Wf = W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
    bf = b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True)
    Wg = (W0.unsqueeze(0).repeat(B, 1, 1) + noise[:, :, :d]).detach().requires_grad_(True)
    bg = (b0.unsqueeze(0).repeat(B, 1) + noise[:, :, d]).detach().requires_grad_(True)
    params = [Wf, bf, Wg, bg]
    lr_v = torch.tensor([c[4] for c in chunk], dtype=F32, device=DEV)
    m_ = [torch.zeros_like(p) for p in params]
    v_ = [torch.zeros_like(p) for p in params]
    tau = torch.tensor([c[1] for c in chunk], dtype=F32, device=DEV)
    mw = torch.tensor([0.0 if c[2] == "0" else mass(c[2], n) / n for c in chunk], dtype=F32, device=DEV)
    wl = torch.stack([wK[c[3]] for c in chunk])
    ov_sum, ov_cnt = torch.zeros(B, device=DEV), torch.zeros(B, device=DEV)
    agree = torch.zeros(B, 4, device=DEV)                                                       # 選んだ一致・選んだ数・落ちた一致・落ちた数（最後のエポック）
    prec = torch.zeros(B, 4, device=DEV)                                                        # 選んだ正しさ・選んだ数・全体の正しさ・全体の数
    yt_all = torch.as_tensor(np.asarray(ytrue), device=DEV) if ytrue is not None else None
    t = 0
    n_ep = max(math.ceil(CT_STEPS / Mb), CT_TK)
    for ep in range(n_ep):
        R = 1.0 - torch.minimum(tau * (ep + 1) / CT_TK, tau)
        perms = torch.stack([torch.randperm(n, generator=gg, device=DEV) for gg in gens])          # 種 × n
        last = ep == n_ep - 1
        for mb in range(Mb):
            idx = perms[:, mb * CT_BATCH:(mb + 1) * CT_BATCH][seed_of]                              # 問 × 本数（同じ種の問は同じミニバッチ）
            valid = ~lmask.gather(1, idx)
            yb = hl.gather(1, idx)
            x = Z[idx]
            Lf = torch.einsum("bnd,bkd->bnk", x, Wf) + bf[:, None]
            Lg = torch.einsum("bnd,bkd->bnk", x, Wg) + bg[:, None]
            lf = -torch.log_softmax(Lf, 2).gather(2, yb.unsqueeze(2)).squeeze(2)
            lg_ = -torch.log_softmax(Lg, 2).gather(2, yb.unsqueeze(2)).squeeze(2)
            inf = torch.full_like(lf, float("inf"))
            kk = torch.clamp((R * valid.sum(1)).round().long(), min=1)
            rank_f = torch.where(valid, lf.detach(), inf).argsort(1).argsort(1)
            rank_g = torch.where(valid, lg_.detach(), inf).argsort(1).argsort(1)
            sel_f = (rank_f < kk[:, None]) & valid
            sel_g = (rank_g < kk[:, None]) & valid
            loss = (mw / R) * ((lf * sel_g).sum(1) + (lg_ * sel_f).sum(1))                     # 選んだ 1 本の重みは m_eff/(R(t)·n)．総重みを τ に依らず m_eff にそろえる
            if len(pos):
                for Wx, bx in ((Wf, bf), (Wg, bg)):
                    Ll = torch.einsum("nd,bkd->bnk", Z[pos], Wx) + bx[:, None]
                    loss = loss + (wl * ce_rows(Ll, ylab)).sum(1) / Mb
            for Wx, bx in ((Wf, bf), (Wg, bg)):
                loss = loss + ((Wx - W0) ** 2).sum((1, 2)) / (2 * cp) / Mb
                if cb is not None:
                    loss = loss + ((bx - b0) ** 2).sum(1) / (2 * cb) / Mb
            for p in params:
                p.grad = None
            loss.sum().backward()
            t += 1
            with torch.no_grad():
                for p, m1, v1 in zip(params, m_, v_):
                    m1.mul_(0.9).add_(p.grad, alpha=0.1)
                    v1.mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                    step = (m1 / (1 - 0.9 ** t)) / ((v1 / (1 - 0.999 ** t)).sqrt() + 1e-8)
                    p.sub_(lr_v.view(-1, *([1] * (p.dim() - 1))) * step)
                if last:
                    nv = valid.sum(1) > 0
                    ov_sum += torch.where(nv, (sel_f & sel_g).sum(1).float() / sel_f.sum(1).clamp(min=1).float(), torch.zeros(B, device=DEV))
                    ov_cnt += nv.float()
                    ag = (Lf.argmax(2) == yb).float()
                    drop = valid & ~sel_f
                    agree += torch.stack([(ag * sel_f).sum(1), sel_f.sum(1).float(), (ag * drop).sum(1), drop.sum(1).float()], 1)
                    if yt_all is not None:
                        ok = (yb == yt_all[idx]).float()
                        prec += torch.stack([(ok * sel_f).sum(1), sel_f.sum(1).float(), (ok * valid).sum(1), valid.sum(1).float()], 1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("cot の損失が NaN/Inf")
    with torch.no_grad():
        v = ov_sum / ov_cnt.clamp(min=1)
        for j, (name, tau_j, meff, K, lr, seed, kind) in enumerate(chunk):
            meth = {"sel": "cot", "ctl": "cotctl", "fix": "cotfix", "zero": "cotzero"}[kind]
            param = f"{name}:tau{tau_j:.4f}lr{lr}s{seed}" if kind == "fix" else f"{name}:tau{tau_j}lr{lr}s{seed}"
            rec.add(student, meth, param, meff, K, Wb=(Wf[j].to(DT), bf[j].to(DT)), Zte=Zte.to(DT))
            rec.add(student, meth + "_g", param, meff, K, Wb=(Wg[j].to(DT), bg[j].to(DT)), Zte=Zte.to(DT))
            if diag is not None:
                vt = float(v[j])
                diag.append({**rec.tag, "student": student, "method": meth, "param": param, "meff": meff, "K": K, "v": vt,
                             "v_norm": (vt - (1 - tau_j)) / tau_j if tau_j > 0 else float("nan"),
                             "agree_sel": float(agree[j, 0] / agree[j, 1].clamp(min=1)), "agree_drop": float(agree[j, 2] / agree[j, 3]) if float(agree[j, 3]) > 0 else float("nan"),
                             "n_sel": float(agree[j, 1]), "n_drop": float(agree[j, 3]),
                             "prec_sel": float(prec[j, 0] / prec[j, 1].clamp(min=1)) if ytrue is not None else float("nan"),
                             "prec_all": float(prec[j, 2] / prec[j, 3].clamp(min=1)) if ytrue is not None else float("nan")})
    del Wf, bf, Wg, bg, params, m_, v_
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- CORAL ＋ none

def lw_cov(X):
    S, _ = ledoit_wolf(X.cpu().numpy().astype(np.float64), assume_centered=True)
    return torch.as_tensor(S, dtype=DT, device=DEV)


def msqrt(S, inv=False):
    lam, V = torch.linalg.eigh(S)
    lam = lam.clamp(min=1e-10)
    return (V * (lam.rsqrt() if inv else lam.sqrt())) @ V.T


def coral_lin(srcs, Xu, rho_s, r, svd=None):
    """ソースごとに候補 Xu（中心化済み）の共分散へ上位 r 次元の部分空間の中でそろえ，`meta5.pool_info` と同じ形の閉じた形の向き lin を作り直す。"""
    Xs_c = {s: X - X.mean(0) for s, (X, y) in srcs.items()}
    Xp = torch.cat([X for X, _ in srcs.values()])                               # `meta5.pool_info` と同じく，中心化しない合併の平均と標準偏差で標準化する
    mu_p, sd_p = Xp.mean(0), Xp.std(0, unbiased=False)
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    del Xp
    lin = {}
    for s, (X, y) in srcs.items():
        Xc = Xs_c[s]
        if r is None:                                                            # そろえない（`pool_info` の lin と一致することの検算に使う）
            X2 = X
        else:
            U = svd[s][:r].T                                                     # ソースと候補を合わせた特徴の主成分（r に依らないので 1 回だけ作る）
            Ps, Pt = Xc @ U, Xu @ U
            Cs, Ct = lw_cov(Ps - Ps.mean(0)), lw_cov(Pt - Pt.mean(0))
            Pn = (Ps - Ps.mean(0)) @ msqrt(Cs, inv=True) @ msqrt(Ct) + Ps.mean(0)
            X2 = X + (Pn - Ps) @ U.T
        Z = (X2 - mu_p) / sd_p
        yt = torch.as_tensor(np.array(y), device=Z.device)
        m = torch.stack([Z[yt == k].mean(0) for k in range(4)])
        Rr = Z - m[yt]
        lam, V = torch.linalg.eigh(Rr.T @ Rr / (len(yt) - 4))
        lam = lam.clamp(min=0.0)
        mt = m - Z.mean(0)
        Pm = V.T @ mt.T
        W = (V @ (Pm / ((1 - rho_s) * lam + rho_s * lam.mean())[:, None])).T
        lin[(s, rho_s)] = (W / sd_p, -0.5 * (mt * W).sum(1))
    return lin


def run_coral(ev, Xu, te_X, src_pc, srcs, n_of, pool_s, C, gamma, rho_s, label_sets, yq, rec, student, lin0, c_raw, cp39, cb39):
    """CORAL ＋ none．Xu：候補（中心化済み），te_X：テスト（同じ中心で中心化済み），srcs：{ソース: (特徴, ラベル)}。"""
    n_min = min(len(Xu), min(len(X) for X, _ in srcs.values()))
    svd = {}
    for s_, (X, _) in srcs.items():                                              # 主成分は r に依らないので，ソースごとに上位 max(r) 本を 1 回だけ作る
        both = torch.cat([X - X.mean(0), Xu])
        _, _, Vt = torch.linalg.svd(both - both.mean(0), full_matrices=False)
        svd[s_] = Vt[:max(CORAL_R)]
        del both, Vt
    cps = tuple(sorted(set(CORAL_CP) | {float(cp39)}))                           # E39 の C′・C_b を格子に入れる（固定版は r = 64 と E39 の値）
    cbs_grid = tuple(sorted(set(c for c in CORAL_CB if c is not None) | ({float(cb39)} if cb39 is not None else set()))) + (None,)
    lab_w = {}
    yt = M.labels(yq)
    for K in KS:
        idx = np.asarray(label_sets[K], int)
        w = torch.zeros(len(Xu), dtype=DT, device=DEV)
        if len(idx):
            it = torch.as_tensor(idx, device=DEV)
            w[it] = present_weights(yt[it])
        lab_w[K] = (idx, w)
    for r in (None,) + CORAL_R:                                                  # r = None は対照（そろえずに同じ (ρ_M, C′, C_b) を選び直す）
        if r is not None and n_min < CORAL_MIN_RATIO * r:
            continue
        t0 = time.perf_counter()
        lin = coral_lin(srcs, Xu, rho_s, r, svd)
        if r is None:                                                            # そろえない版は元の下書きの lin と一致するはず（I7）
            dmax = max(float((lin[k_][0] - lin0[k_][0]).abs().max() / lin0[k_][0].abs().max().clamp(min=1e-30)) for k_ in lin)
            rec.conv.append({**rec.tag, "student": student, "what": "coral_lin_check", "rel_diff": dmax})
            if dmax > 1e-6:
                raise AssertionError(f"CORAL の下書きの尺度が元と違う（相対差 {dmax:.2e}）")
        st = Student(ev, Xu, src_pc, lin, n_of, pool_s, C, gamma, rho_s, list(CORAL_RHO))
        Xstd = st.Xstd
        Xte_std = (te_X - st.src.mu) / st.src.sd
        st.Xstd = None
        for rho_m in CORAL_RHO:
            Am = st.mats[rho_m][0]
            Zc = Xstd @ Am
            Zt = Xte_std @ Am
            groups = [(K, cbv) for K in KS for cbv in cbs_grid]
            masks, ws, cbs = [], [], []
            for K, cbv in groups:
                idx, w = lab_w[K]
                masks.append(np.ones(len(Zc), bool))
                ws.append(w)
                cbs.append(cbv)
            fits = logreg.fit_batch(Zc, yt, masks, cps, anchor=st.anchor(rho_m), anchor_c=st.C, stats=(st.zero, st.one), Cb=cbs, weights=ws)
            # fit_batch の問 p = g·len(Cs) + i．重みが全部 0 の問（K=0）は罰則だけなので出発点そのものになる
            for gi, (K, cbv) in enumerate(groups):
                f = fits[gi]
                for cpv in cps:
                    W, b = f.W[cpv], f.b[cpv]
                    if K == "0":
                        W, b = st.W_std @ st.mats[rho_m][1], st.b_std
                    cbs_ = "free" if cbv is None else cbv
                    meth, par = ("coralctl", f"rho{rho_m}cp{cpv}cb{cbs_}") if r is None else ("coral", f"r{r}rho{rho_m}cp{cpv}cb{cbs_}")
                    rec.add(student, meth, par, "-", K, logits=Zt @ W.T + b)
                    if rec.keep:                                                 # 生の特徴の上の係数（logits = X_raw Wᵀ + b）
                        Wr = (W @ Am) / st.src.sd
                        br = b - Wr @ (c_raw + st.src.mu)
                        rec.W.setdefault(student + "@coral", []).append((len(rec.rows) - 1, Wr.to(torch.float32).cpu().numpy(),
                                                                         br.to(torch.float32).cpu().numpy()))
            del Zc, Zt, fits
        rec.conv.append({**rec.tag, "student": student, "what": "coralctl" if r is None else f"coral_r{r}", "sec": time.perf_counter() - t0})
        del st, Xstd, Xte_std
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 種の扱い（仮のターゲットは種 0 で全候補，評価は種 0 で全候補 ＋ 選ばれた設定を種 1・2 で）

SEEDED = ("dann", "mme", "mapctl", "xmuda")


def strip_seed(param):
    return re.sub(r"s\d+$", "", param)


def choose_seeded(prows):
    """仮のターゲットの種 0 の値から，(生徒, 方法, K) ごとに「none を含めて選んだ設定」と「none を除いて選んだ設定」（種を除いた param）を選ぶ。"""
    d = pd.DataFrame(prows)
    if d.empty:
        return {}
    d["K"] = d["K"].astype(str)
    out = {}
    for (st, K), g in d.groupby(["student", "K"]):
        nb = g[g.method == "none"].uar.mean() if (g.method == "none").any() else -1.0
        for meth in SEEDED:
            gm = g[g.method == meth]
            if gm.empty:
                continue
            sc = gm.assign(p=gm.param.map(strip_seed)).groupby("p").uar.mean()
            best = sc.idxmax()
            out[(st, meth, K)] = {best}                                          # none を除いて選んだもの（P4 の判定に使う）．none を含めて選ぶと none になるなら，種を足す必要は無い
    return out


def problems_for(student, family, methods, pseudo, chosen):
    """仮のターゲット：種 0 の全候補．評価：種 0 の全候補と，選ばれた設定の種 1・2。"""
    if family == "map":
        allp = map_problems(methods, (SEEDS[0],))
        if pseudo:
            return allp
        extra = [c for c in map_problems(methods, SEEDS[1:]) if strip_seed(map_param(c)) in chosen.get((student, c[0], c[7]), set())]
        return allp + extra
    allp = xm_problems((SEEDS[0],))
    if pseudo:
        return allp
    pick = lambda c: f"lr{c[0]}it{c[1]}kap{c[2]}mu{c[3]}"  # noqa: E731
    want = {K: set().union(*[v for (st, m_, K_), v in chosen.items() if m_ == "xmuda" and K_ == K] or [set()]) for K in KS}
    return allp + [c for c in xm_problems(SEEDS[1:]) if pick(c) in want[c[5]]]


# ---------------------------------------------------------------- 世界と種類の 1 反復

def src_feats(st, rho_m, srcs):
    Am = st.mats[rho_m][0]
    Zs = torch.cat([((X - X.mean(0) - st.src.mu) / st.src.sd) @ Am for X, _ in srcs.values()])
    ys = np.concatenate([np.asarray(y) for _, y in srcs.values()])
    return Zs, ys


def noise_rates(teachers, ytrue, student, tag, noise):
    yt = torch.as_tensor(np.asarray(ytrue), device=DEV)
    for name, P in teachers.items():
        for K in KS:
            err = float((E77.tk(P, K).argmax(1) != yt).float().mean())
            noise.append({**tag, "student": student, "teacher": name, "K": K, "err": err})


def world_unit2(world, stu, yq, label_sets, rec, methods, srcs_of, raw_of, coral_ctx, tau_fix, diag, noise, pseudo, chosen):
    mods = WORLDS[world]
    lab = lab_of(label_sets, yq)
    srcs = {k: srcs_of(k) for k in mods + [world]}
    for k in mods + [world]:                                                     # W・b を生の特徴の上へ写すための座標（E77 と同じ）
        st = stu[k]["st"]
        rec.xform[k] = (st.mats[stu[k]["rho_m"]][0], st.src.sd, raw_of(k)[8] + st.src.mu)
    if "coral" in methods:                                                       # Student を作り直すので Prob（Xstd を消す）より先に
        for k in mods + [world]:
            Xu, te_X, src_pc, n_of, pool_s, C, gamma, rho_s, mu, lin0 = raw_of(k)
            run_coral(coral_ctx[k], Xu, te_X, src_pc, srcs[k], n_of, pool_s, C, gamma, rho_s, label_sets, yq, rec, k, lin0, mu,
                      stu[k]["cp"], stu[k]["cb"])
    prs = {k: Prob(stu[k], yq, label_sets) for k in mods + [world]}
    for k in mods + [world]:
        none = prs[k].solve([(K, None, K) for K in KS])
        for K in KS:
            rec.add(k, "none", "-", "0", K, Wb=none[K], Zte=prs[k].s["Zte"])
    for k in mods:                                                               # 融合の生徒では DANN・MME を回さない（計算量．E78 の節）
        t0 = time.perf_counter()
        Zs, ys = src_feats_from(stu[k], srcs[k])
        run_map(prs[k], Zs, ys, rec, k, lab, problems_for(k, "map", methods, pseudo, chosen))
        del Zs
        rec.conv.append({**rec.tag, "student": k, "what": "map", "sec": time.perf_counter() - t0})
    if "xmuda" in methods:
        t0 = time.perf_counter()
        run_xmuda(prs, mods, rec, lab, problems_for("*", "xmuda", methods, pseudo, chosen))
        rec.conv.append({**rec.tag, "student": "*", "what": "xmuda", "sec": time.perf_counter() - t0})
    if "cot" in methods:
        none_f = prs[world].solve([(K, None, K) for K in KS])
        pFK = {K: prs[world].probs(none_f[K], prs[world].s["Z"]) for K in KS}
        pF = tens(stu[world]["P"])
        poe0 = tens(beliefs1(stu, world)["poe"])
        lg = {K: 0 for K in KS}
        for m in mods:
            sol = prs[m].solve([(K, weak_rows(pFK[K], E77.REF_MIN), K) for K in KS])
            for K in KS:
                lg[K] = lg[K] + prs[m].probs(sol[K], prs[m].s["Z"]).clamp(min=1e-12).log()
        ref = {K: torch.softmax(lg[K], 1) for K in KS}
        if rec.keep:                                                             # 候補の上の教師の確率（E77 の teachers.npz と照合する）
            rec.teachers = {"pF": pF.cpu().numpy(), "poe0": poe0.cpu().numpy(), **{f"pFK_{K}": pFK[K].cpu().numpy() for K in KS},
                            **{f"ref_{K}": ref[K].cpu().numpy() for K in KS}}
        for k in mods + [world]:
            t0 = time.perf_counter()
            teachers = {"pF": pF, "pFK": pFK, "ref": ref} if k != world else {"poe0": poe0, "ref": ref}
            if pseudo:
                noise_rates(teachers, yq, k, rec.tag, noise)
            tf = None if pseudo else tau_fix.get(k)
            run_cot(prs[k], rec, k, lab, teachers, tau_fix=tf, ytrue=yq if pseudo else None, diag=diag)
            rec.conv.append({**rec.tag, "student": k, "what": "cot", "sec": time.perf_counter() - t0})


def src_feats_from(s, srcs):
    return src_feats(s["st_ref"], s["rho_m"], srcs)


# ---------------------------------------------------------------- ターゲットごと

def tau_fix_of(noise_rows):
    """仮のターゲットの候補で測った参照の誤りの割合の中央値（生徒 × 参照 × K ごと．pFK と ref は K 本で学び直すので K で変わる）。"""
    if not noise_rows:
        return {}
    d = pd.DataFrame(noise_rows)
    d["K"] = d["K"].astype(str)
    med = d.groupby(["student", "teacher", "K"]).err.median()
    out = {}
    for (st, te, K), v in med.items():
        out.setdefault(st, {}).setdefault(te, {})[K] = float(v)
    return out


def run_world(world, T, data_f, data_m, reps, pseudo_reps, out, cache, commit, methods, pseudo_targets=None):
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
    dat_of = lambda k, g: data_m[k][g] if k != world else data_f[g]  # noqa: E731
    prows, diag, noise, conv = [], [], [], []
    E77.save_settings(out, T, sets, {"c_fit": C_FIT, "sources": S})

    chosen = {}
    if COPY_PSEUDO and T in E77.COPY_SRC:                                        # 短縮 a：台本の仮のターゲットは即興の結果を写す
        prows, conv, diag, _ = E77.copy_pseudo(out, T, "E78", KS, commit, E77.pseudo_pairs(S, pseudo_targets, pseudo_reps),
                                               {"ks": list(KS), "methods": list(methods)})
        S_loop = []
    else:
        S_loop = S

    def unit(Tn, pool_s, pool, te, info, ev, rec, pseudo, tau_fix):
        stu = make_students(world, Tn, data_f, data_m, pool_s, pool, te, sets, info)
        for k in stu:                                                           # ソースの特徴の座標に使う（Prob が Xstd を消す前に参照を残す）
            stu[k]["st_ref"] = stu[k]["st"]
        srcs_of = lambda k: {s: (dat_of(k, s).X, dat_of(k, s).y) for s in pool_s}  # noqa: E731

        def raw_of(k):
            dat = dat_of(k, Tn)
            mu = M.rows(dat.X, pool).mean(0)
            gamma, rho_s, rho_m, cb, cp, _ = sets[k]
            src, lin, n_of = info[k]
            return M.rows(dat.X, pool) - mu, M.rows(dat.X, te) - mu, src, n_of, pool_s, C_FIT, gamma, rho_s, mu, lin
        coral_ctx = {k: dat_of(k, Tn) for k in stu}
        world_unit2(world, stu, ev.y[pool], BD.label_sets_of(rec_ctx[0], rec_ctx[1], pool), rec, methods, srcs_of, raw_of, coral_ctx,
                    tau_fix, diag, noise, pseudo, chosen)
    for c in S_loop:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        info_c = {k: M5.pool_info(dat_of_all(data_m, data_f, k, world), pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            rec = Rec({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag = conv, diag
            rec_ctx = (ctxc, r)
            unit(c, pool_c, pool, te, info_c, evc, rec, True, {})
            prows += rec.rows
            del rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    pd.DataFrame(noise).assign(commit=commit).to_csv(out / f"{T}_noise.csv", index=False)
    tau_fix = tau_fix_of(noise)
    chosen.update(choose_seeded(prows))
    pd.DataFrame([{"student": k[0], "method": k[1], "K": k[2], "param": p_} for k, v in chosen.items() for p_ in sorted(v)]).to_csv(
        out / f"{T}_chosen.csv", index=False)
    info = {k: M5.pool_info(dat_of_all(data_m, data_f, k, world), S, C_FIT, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, reps, list(KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        rec = Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag = conv, diag
        rec_ctx = (ctx, r)
        unit(T, S, pool, te, info, ev, rec, False, tau_fix)
        rec.save(out, T, r, te)
        rows += rec.rows
        del rec
        torch.cuda.empty_cache()
    pd.DataFrame(diag).assign(commit=commit).to_csv(out / f"{T}_diag.csv", index=False)
    pd.DataFrame(conv).assign(commit=commit).to_csv(out / f"{T}_conv.csv", index=False)
    pd.DataFrame(rows).assign(commit=commit).to_csv(out / f"{T}.csv", index=False)


def dat_of_all(data_m, data_f, k, world):
    return data_m[k] if k != world else data_f


def run_kind(kind, T, data, reps, pseudo_reps, out, cache, commit, methods, pseudo_targets=None):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    meths = [m for m in methods if m in ("dann", "mme", "mapctl", "coral")]
    chosen = {}
    conv, diag = [], []
    E77.save_settings(out, T, {kind: (gamma, rho_s, rho_m, cb, cp, None)}, {"c_fit_pseudo": C_FIT, "c_eval": C_T, "sources": S})

    def unit(ev, pool_s, pool, te, ctxo, r, src, lin, n_of, C, rec):
        ct = ctxo.ct[r]
        Xu = M.rows(ev.X, pool) - ct
        te_X = M.rows(ev.X, te) - ct
        srcs = {s: (data[s].X, data[s].y) for s in pool_s}
        yq = ev.y[pool]
        label_sets = BD.label_sets_of(ctxo, r, pool)
        if "coral" in meths:
            run_coral(ev, Xu, te_X, src, srcs, n_of, pool_s, C, gamma, rho_s, label_sets, yq, rec, kind, lin, ct, cp, cb)
        st = Student(ev, Xu, src, lin, n_of, pool_s, C, gamma, rho_s, [rho_m])
        Am = st.mats[rho_m][0]
        rec.xform[kind] = (Am, st.src.sd, ct + st.src.mu)
        s = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((te_X - st.src.mu) / st.src.sd) @ Am}
        Zs, ys = src_feats(st, rho_m, srcs)
        pr = Prob(s, yq, label_sets)
        none = pr.solve([(K, None, K) for K in KS])
        for K in KS:
            rec.add(kind, "none", "-", "0", K, Wb=none[K], Zte=s["Zte"])
        t0 = time.perf_counter()
        run_map(pr, Zs, ys, rec, kind, lab_of(label_sets, yq), problems_for(kind, "map", meths, rec.keep is False, chosen))
        rec.conv.append({**rec.tag, "student": kind, "what": "map", "sec": time.perf_counter() - t0})
    prows = []
    if COPY_PSEUDO and T in E77.COPY_SRC:                                        # 短縮 a
        prows, conv_c, diag_c, _ = E77.copy_pseudo(out, T, "E78", KS, commit, E77.pseudo_pairs(S, pseudo_targets, pseudo_reps),
                                                   {"ks": list(KS), "methods": list(methods)})
        conv += conv_c
        diag += diag_c
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
            rec = Rec({"pseudo": c, "rep": r}, evc.y[te], keep=False)
            rec.conv, rec.diag = conv, diag
            unit(evc, pool_c, pool, te, ctxc, r, src_c, lin_c, n_c, C_FIT, rec)
            prows += rec.rows
            del rec
            torch.cuda.empty_cache()
    pd.DataFrame(prows).assign(commit=commit).to_csv(out / f"{T}_pseudo.csv", index=False)
    chosen.update(choose_seeded(prows))
    pd.DataFrame([{"student": k[0], "method": k[1], "K": k[2], "param": p_} for k, v in chosen.items() for p_ in sorted(v)]).to_csv(
        out / f"{T}_chosen.csv", index=False)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, list(KS_LAB_ALL), center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        rec = Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.diag = conv, diag
        unit(ev, S, pool, te, ctx, r, src_pc, lin, n_of, C_T, rec)
        rec.save(out, T, r, te)
        rows += rec.rows
        del rec
        torch.cuda.empty_cache()
    pd.DataFrame(diag).assign(commit=commit).to_csv(out / f"{T}_diag.csv", index=False)
    pd.DataFrame(conv).assign(commit=commit).to_csv(out / f"{T}_conv.csv", index=False)
    pd.DataFrame(rows).assign(commit=commit).to_csv(out / f"{T}.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--world")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pseudo-targets", nargs="+", help="試走用．'none' で仮のターゲットを回さない")
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--gpu-fraction", type=float, default=0.6)
    ap.add_argument("--ks", nargs="+", help="解く K（既定は全部．DAI 論文の短縮 c は 0 5 25）．K 本の抜き方は変わらない")
    ap.add_argument("--copy-pseudo", action="store_true", help="IEMOCAP-script の仮のターゲットを同じ出力の場所の IEMOCAP-impro の結果から写す（短縮 a）")
    args = ap.parse_args()
    if bool(args.kind) == bool(args.world):
        ap.error("--kind か --world のどちらか 1 つを指定する")
    global KS, KS_LAB, COPY_PSEUDO
    if args.ks:
        E77.set_ks(args.ks)
        KS, KS_LAB = E77.KS, E77.KS_LAB
    COPY_PSEUDO = args.copy_pseudo
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True                             # 確率的な勾配法の 32 ビットの行列積だけに効く（E39 の学習器は 64 ビット）
    name = args.kind or args.world
    out = args.out_dir or ROOT / "results" / f"e78_{name}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    groups = groups_for(name)
    todo = [t for t in (args.target or groups) if not check_resume(out, t, commit)]
    if not todo:
        print(f"{name}：全ターゲット済み", flush=True)
        return 0
    ptg = ["__none__"] if args.pseudo_targets == ["none"] else args.pseudo_targets
    prov = {"script": "exp/compare_dai2.py", "experiment": "E78", "argv": sys.argv[1:], "commit": commit, "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo, "reps": args.reps, "pseudo_reps": args.pseudo_reps,
            "pseudo_targets": args.pseudo_targets, "methods": args.methods, "gpu_fraction": args.gpu_fraction, "ks": list(KS), "copy_pseudo": COPY_PSEUDO, **E77.versions(), **GRID,
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
            run_world(name, t, data_f, data_m, args.reps, args.pseudo_reps, out, cache, commit, args.methods, ptg)
        else:
            run_kind(name, t, data, args.reps, args.pseudo_reps, out, cache, commit, args.methods, ptg)
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{name} {t}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
